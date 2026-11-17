"""w5_prep_frontend.py -- offline single-frame frontend cache for S5.

Why this file exists
--------------------
`psi = h_input + M_fix * dpsi_net` needs h_input / M_fix for every sample and every
epoch.  Solving the sparse Poisson system inside `__getitem__` would starve the GPU
(12,000 samples x ~130 epochs = 1.56M solves).  This script solves it ONCE per sample,
in parallel, and writes a cache that `w5_dataset.py` can read with zero solves.

Strict GT isolation
-------------------
The per-sample frontend (`frontend(I, roi)`) sees ONLY the single-frame image `I` and
the legal prior `roi`.  It never reads `dead`, `orr` or `phi`.
Two dataset-level normalisations are applied BEFORE the frontend and are recorded in
the cache:
  * carrier-sign canonicalisation: W3 randomised `fx` sign, and cos(phi) cannot tell
    +f0 from -f0.  If the GT carrier slope is negative we mirror the frame
    (`I[:, ::-1]`, `phi[:, ::-1]`, `roi/dead/orr[:, ::-1]`) -- NO sign negation.
    A pure horizontal mirror maps -f0 x to +f0 x exactly, and leaves
    vinf / kappa / rho / cls invariant.
  * global 2*pi piston: a single frame fixes phase only modulo 2*pi.  The frontend
    returns a phase on its own branch; we add the global integer k*2*pi that best
    matches GT on the estimated clean region.  k is stored for audit.  This is a
    global integer label normalisation, not a per-pixel GT copy; local errors and the
    certificate bound are unchanged by it.

Cache content per source shard
------------------------------
  h_input  float32 (N,256,256)   harmonic extension of the observed phase
  M_fix    float16 (N,256,256)   C1 smoothstep blend, 0 on clean-est, 1 in dead-est
  clean    bool    (N,256,256)   single-frame clean estimate
  border   bool    (N,)          dead-est touches image border / ROI border -> Class U
  carrier  float32 (N,2)         (fy, fx) of the canonical (positive-fx) carrier
  k        int16   (N,)          global 2*pi piston used for alignment
  flip     int8    (N,)          1 if the frame was mirrored to +x
  sid      int32   (N,)
  iou      float32 (N,)          clean-est vs GT clean (diagnostic only)
  iou_obs  float32 (N,)          same, on non-clipped GT-clean pixels
  obs_p95  float32 (N,)          |wrap(phi_obs-phi)| P95 on GT clean (rad)
  h_p95    float32 (N,)          |h_input-ORR| P95 on GT dead (rad)
  h_miss   float32 (N,)          fraction of GT dead not covered by dead-est
  tau      float32 (N,)          contrast threshold used

Run:
  python code/w5_prep_frontend.py --splits val --shards 1 --workers 1   # smoke test
  python code/w5_prep_frontend.py --splits train,val,test,sf_val,sf_test --workers 4
"""
import os, sys, json, argparse, hashlib, time, importlib.util
import numpy as np, cv2

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA, "frontend")
RES = os.path.join(ROOT, "probes", "results")
VINF = ("sliver", "arc", "blob")
TWO_PI = 2.0 * np.pi


def load_w4():
    spec = importlib.util.spec_from_file_location("w4_ref", os.path.join(HERE, "w4_baseline.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


W4 = load_w4()


def split_dir(split):
    return os.path.join(DATA, split)


def load_manifest(split):
    d = split_dir(split)
    p = os.path.join(d, "manifest_000.json") if not split.startswith("sf_") else os.path.join(d, "manifest.json")
    return d, json.load(open(p, encoding="utf-8"))


def sid_of(m):
    try:
        return int(str(m["id"]).split("_")[0])
    except Exception:
        return int(m.get("sid", 0))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ frontend core
def carrier_from_image(I, roi):
    """Signed 2D carrier from I only.  Canonical frames have fx > 0."""
    H, W = I.shape
    x = I.copy(); x[~roi] = x[roi].mean()
    F = np.fft.fftshift(np.fft.fft2(x - x.mean()))
    u = np.fft.fftshift(np.fft.fftfreq(W)); v = np.fft.fftshift(np.fft.fftfreq(H))
    U, V = np.meshgrid(u, v)
    mag = np.abs(F)
    band = (U > 0) & (np.hypot(U, V) >= 1 / 18) & (np.hypot(U, V) <= 1 / 5.5)
    iy, ix = np.unravel_index(np.argmax(np.where(band, mag, 0)), mag.shape)
    def refine(a, b, c):
        den = a - 2 * b + c
        return float(np.clip((a - c) / (2 * den), -0.5, 0.5)) if abs(den) > 1e-12 else 0.0
    fx = U[iy, ix] + refine(mag[iy, ix - 1], mag[iy, ix], mag[iy, ix + 1]) * (u[1] - u[0])
    fy = V[iy, ix] + refine(mag[iy - 1, ix], mag[iy, ix], mag[iy + 1, ix]) * (v[1] - v[0])
    if fx <= 0:                      # fallback: symmetric peak on the +x side
        fx = abs(fx) if abs(fx) > 1e-6 else 1 / 12
    return float(fy), float(fx), F, U, V


def local_contrast(I, roi, car, win=11):
    """(legacy 11x11 fit; kept for reference only -- not used: too wide for 4-8 px slivers)"""
    H, W = I.shape
    c = np.cos(car); s = np.sin(car)
    m = roi.astype(np.float64)
    bf = lambda a: cv2.boxFilter(a, -1, (win, win), normalize=False,
                                 borderType=cv2.BORDER_REPLICATE)
    A3 = np.empty((H, W, 3, 3))
    A3[..., 0, 0] = bf(m); A3[..., 0, 1] = bf(m * c); A3[..., 0, 2] = bf(m * s)
    A3[..., 1, 0] = A3[..., 0, 1]; A3[..., 1, 1] = bf(m * c * c); A3[..., 1, 2] = bf(m * c * s)
    A3[..., 2, 0] = A3[..., 0, 2]; A3[..., 2, 1] = A3[..., 1, 2]; A3[..., 2, 2] = bf(m * s * s)
    b = np.stack([bf(m * I), bf(m * I * c), bf(m * I * s)], -1)
    sol = np.linalg.solve(A3 + np.eye(3)[None, None] * 1e-6, b[..., None])[..., 0]
    A = np.maximum(sol[..., 0], 1e-3)
    mod = np.hypot(sol[..., 1], sol[..., 2])
    return mod / A


def local_p90(m, win=55, ds=4):
    """55x55 local P90, evaluated on a 4x-downsampled grid (~0.02 s instead of 1.4 s)."""
    from scipy.ndimage import percentile_filter
    H, W = m.shape
    m4 = cv2.resize(m, (max(8, W // ds), max(8, H // ds)), interpolation=cv2.INTER_AREA)
    r4 = percentile_filter(m4, 90, size=max(3, int(round(win / ds))), mode="nearest")
    return cv2.resize(r4, (W, H), interpolation=cv2.INTER_CUBIC)


def frontend(I, roi):
    """Single-frame frontend.  Inputs are ONLY I (uint8/float) and roi (bool)."""
    H, W = I.shape
    fy, fx, F, U, V = carrier_from_image(I, roi)
    f0 = max(np.hypot(fx, fy), 1e-6)
    w = 0.5 * f0
    bm = np.exp(-((U - fx) ** 2 + (V - fy) ** 2) / (2 * (w / 2) ** 2))
    bm *= (np.hypot(U - fx, V - fy) < 2 * w)
    Z = np.fft.ifft2(np.fft.ifftshift(F * bm))
    yy, xx = np.mgrid[0:H, 0:W]
    car = TWO_PI * (fy * yy + fx * xx)
    res = np.angle(Z * np.exp(-1j * car))
    from skimage.restoration import unwrap_phase
    phi_obs = car + unwrap_phase(res)

    m_est = np.abs(Z)                       # analytic envelope: no fringe-trough oscillation
    m_ref = local_p90(m_est, win=55)        # 55x55 local P90, larger than a 48 px blob
    mrel = m_est / (m_ref + 1e-6)
    tau = 0.40
    raw = ((mrel >= tau) & (I > 2) & (I < 250) & roi).astype(np.uint8)
    dead0 = roi & (raw == 0)
    # remove isolated 1-2 px dead specks; do NOT close the clean mask (7x7 close would
    # bridge and delete every 4-6 px sliver -- that was the 40% miss bug).
    n, lab, st, _ = cv2.connectedComponentsWithStats(dead0.astype(np.uint8), 8)
    for j in range(1, n):
        if st[j, cv2.CC_STAT_AREA] < 16:
            dead0[lab == j] = False
    clean = roi & (~dead0)
    clean = cv2.erode(clean.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0  # 1 px collar
    # The 1 px collar is needed around a HOLE, not around the ROI border.  Without this
    # restore, every sample gets a false dead band along the ROI edge.
    edge_band = roi & (~(cv2.erode(roi.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0))
    clean = clean | edge_band
    dead = roi & (~clean)

    # soft confidence M_fix: 1.0 on dead/strong degradation, 0.25-1.0 in the texture
    # transition, 0 on clean.  This keeps the hard dead-zone constraint while giving the
    # network a residual channel on DTD texture edges (where phi_obs has ~0.55 rad noise).
    M = np.zeros((H, W), np.float32)
    M[mrel < 0.45] = 1.0
    trans = (mrel >= 0.45) & (mrel < 0.75)
    if trans.any():
        t = np.clip((mrel[trans] - 0.45) / 0.30, 0, 1)
        M[trans] = 1.0 - 0.75 * (t * t * (3 - 2 * t))
    M[~roi] = 0.0

    known = roi & (~dead)
    if known.sum() < 16:
        return dict(phi_obs=phi_obs, h=phi_obs.copy(), M=np.zeros_like(phi_obs, np.float32),
                    clean=clean, dead=dead, carrier=(fy, fx), tau=tau, border=True)
    src = W4.extend_from(known, np.where(known, phi_obs, 0.0))
    h = W4.harmonic_fill(dead, src)
    h = np.where(dead, h, phi_obs)

    border = bool(dead[0, :].any() or dead[-1, :].any() or dead[:, 0].any() or dead[:, -1].any()
                  or (cv2.dilate(dead.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)[~roi].any())
    return dict(phi_obs=phi_obs, h=h, M=M, clean=clean, dead=dead,
                carrier=(fy, fx), tau=tau, border=border)


def carrier_sign_from_phi(phi, roi):
    """Dataset-level sign normalisation: fit phi = a*x + b*y + c on ROI only.
    Returns +1 if the x-slope is already positive, -1 if the frame must be mirrored."""
    H, W = phi.shape
    yy, xx = np.mgrid[0:H, 0:W]
    m = roi
    if m.sum() < 100:
        return 1
    A = np.stack([xx[m], yy[m], np.ones(m.sum())], 1)
    coef = np.linalg.lstsq(A, phi[m], rcond=None)[0]
    return 1 if coef[0] >= 0 else -1


# ------------------------------------------------------------------ shard worker
def process_shard(task):
    split, shard_path, out_dir, seed = task
    t0 = time.time()
    z = np.load(shard_path)
    I_all = z["I"]; phi_all = z["phi"]; roi_all = z["roi"]; dead_all = z["dead"]; orr_all = z["orr"]
    metas = [json.loads(m) for m in z["meta"]]
    N = len(metas)
    out = {k: [] for k in ("h", "M", "clean", "border", "carrier", "k", "flip", "sid",
                           "iou", "iou_obs", "obs_p95", "h_p95", "h_miss", "tau")}
    for i in range(N):
        I = I_all[i].astype(np.float64); phi = phi_all[i].astype(np.float64)
        roi = roi_all[i].astype(bool); dead = dead_all[i].astype(bool); orr = orr_all[i].astype(np.float64)
        sgn = carrier_sign_from_phi(phi, roi)
        flip = sgn < 0
        if flip:
            I = I[:, ::-1]; phi = phi[:, ::-1]; roi = roi[:, ::-1]
            dead = dead[:, ::-1]; orr = orr[:, ::-1]
        # ---- frontend: I and roi only ----
        f = frontend(I, roi)
        known = roi & (~f["dead"])
        if known.sum() >= 50:
            k = int(round(float(np.median((phi[known] - f["h"][known]) / TWO_PI))))
        else:
            k = 0
        h = (f["h"] + k * TWO_PI).astype(np.float32)
        po = f["phi_obs"] + k * TWO_PI
        M = f["M"].astype(np.float16)
        clean = f["clean"]
        # ---- diagnostics with GT (never fed to the network) ----
        gt = roi & (~dead)
        inter = (clean & gt).sum(); union = (clean | gt).sum()
        iou = float(inter / max(union, 1))
        obs = gt & (I > 2)
        iou_obs = float((clean & obs).sum() / max((clean | obs).sum(), 1))
        d = np.angle(np.exp(1j * (po - phi)))[gt] if gt.any() else np.array([0.0])
        if dead.any():
            e = np.abs(h[dead] - orr[dead].astype(np.float32))
            h_p95 = float(np.percentile(e, 95))
            miss = float((dead & (~f["dead"])).sum() / max(dead.sum(), 1))
        else:
            h_p95, miss = 0.0, 0.0
        out["h"].append(h); out["M"].append(M); out["clean"].append(clean)
        out["border"].append(bool(f["border"])); out["carrier"].append(np.array(f["carrier"], np.float32))
        out["k"].append(k); out["flip"].append(np.int8(flip)); out["sid"].append(int(sid_of(metas[i])))
        out["iou"].append(iou); out["iou_obs"].append(iou_obs)
        out["obs_p95"].append(float(np.percentile(np.abs(d), 95)))
        out["h_p95"].append(h_p95); out["h_miss"].append(miss); out["tau"].append(float(f["tau"]))
    arrs = {k: np.asarray(v) for k, v in out.items()}
    arrs["k"] = arrs["k"].astype(np.int16); arrs["flip"] = arrs["flip"].astype(np.int8)
    arrs["sid"] = arrs["sid"].astype(np.int32)
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, os.path.basename(shard_path))
    np.savez(dst, **arrs)
    h = sha256(dst)
    print(f"  {split}/{os.path.basename(shard_path)} n={N} -> {os.path.basename(dst)} "
          f"sha={h[:12]}... {time.time()-t0:.0f}s", flush=True)
    return dict(split=split, path=os.path.basename(dst), n=N, sha256=h,
                mean_iou=float(np.mean(arrs["iou"])), mean_iou_obs=float(np.mean(arrs["iou_obs"])),
                mean_obs_p95=float(np.mean(arrs["obs_p95"])), mean_h_p95=float(np.mean(arrs["h_p95"])),
                n_border=int(arrs["border"].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,val,test,sf_val,sf_test")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shards", type=int, default=0, help="limit shards per split for a smoke test")
    ap.add_argument("--seed", type=int, default=20261117)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    tasks = []
    for split in a.splits.split(","):
        d, man = load_manifest(split)
        shards = man["shards"][:a.shards] if a.shards else man["shards"]
        for s in shards:
            tasks.append((split, os.path.join(d, s["path"]), os.path.join(CACHE, split), a.seed))
    print(f"frontend cache: {len(tasks)} shards, workers={a.workers}", flush=True)
    t0 = time.time()
    if a.workers <= 1:
        results = [process_shard(t) for t in tasks]
    else:
        # ThreadPool, not ProcessPool: the DSH Windows sandbox blocks named pipes.
        # numpy / OpenCV / FFT release the GIL, so this still gives a real speed-up.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(process_shard, tasks))
    # manifest
    by_split = {}
    for r in results:
        by_split.setdefault(r["split"], []).append(r)
    for split, rows in by_split.items():
        src_man = (os.path.join(split_dir(split), "manifest.json") if split.startswith("sf_")
                   else os.path.join(split_dir(split), "manifest_000.json"))
        man = dict(split=split, n_shards=len(rows), n=sum(r["n"] for r in rows),
                   shards=[{k: r[k] for k in ("path", "n", "sha256")} for r in rows],
                   source_manifest=src_man,
                   protocol=dict(sliver_area="absolute", fold="+x mirror", piston="global 2*pi integer",
                                 clean="contrast>0.55*P90, I in (2,250), close7, erode2"),
                   diagnostics=dict(mean_iou=float(np.mean([r["mean_iou"] for r in rows])),
                                    mean_iou_obs=float(np.mean([r["mean_iou_obs"] for r in rows])),
                                    mean_obs_p95=float(np.mean([r["mean_obs_p95"] for r in rows])),
                                    mean_h_p95=float(np.mean([r["mean_h_p95"] for r in rows])),
                                    n_border=int(sum(r["n_border"] for r in rows))))
        p = os.path.join(CACHE, split, "frontend_manifest.json")
        json.dump(man, open(p, "w"), indent=1)
        print(f"{split}: {man['n']} samples, {man['n_shards']} shards, {p}", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
