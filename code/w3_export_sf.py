"""w3_export_sf.py -- SynthFringe external benchmark export (ZERO-SHOT test set).

Blocks 12..14 -> data/sf_val   (target-domain recalibration only)
Blocks 15..17 -> data/sf_test  (one-shot zero-shot numbers)
A 30-scene isolation buffer is dropped at every block boundary; blocks 0..11 are
NEVER touched, so the network (trained on MATLAB only) has never seen a single
SynthFringe frame.

Same schema / same orr_reference / same kappa_MLQ / same no-border rule as the
MATLAB shards.

--------------------------------------------------------------------------
2026-11-17 S5 revision (three fixes, see doc/W4 报告 §SF-扩样)
--------------------------------------------------------------------------
1) hole count is now 1..4 (was 1..3).  w3_gen.py G5 -- the frozen generator --
   draws 1..4 holes; the old export script was the odd one out.
2) sliver placement uses ABSOLUTE pixel area [130, 800] px per hole and total
   dead >= 400 px.  Why: every SF sample is a 256 crop centred on the object, so
   its ROI occupancy is 0.74-0.81, versus 0.31-0.33 for MATLAB.  The old
   relative gate (0.8-3.5% of ROI) therefore demanded 2.3x larger slivers in SF
   and kept only the largest ~6% of scenes -- a systematic domain shift, not a
   real geometric limit.  Absolute area reproduces MATLAB's realised sliver
   geometry: total dead P10/50/90 = 450/781/1186 px, versus 421/782/1133 px in
   MATLAB-val.  arc/blob/edge/cut are unchanged (their relative gates are not
   binding and their realised sizes already match MATLAB).
3) vinf samples: ONE sample per scene, 150 per family (sliver/arc/blob) = 450
   unique scenes per split, assigned by max-flow.  No scene is reused across
   the three vinf families, so the CP sample size is not inflated by repeats.
   edge/cut/degraded (80 each) are for zero-shot ERROR only and are NOT used in
   the certificate / CP analysis.

Run:
  python w3_export_sf.py --split val  --n-vinf 150 --n-other 80
  python w3_export_sf.py --split test --n-vinf 150 --n-other 80
"""
import os, sys, json, argparse, hashlib, glob
from collections import Counter
import numpy as np, cv2
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_flow
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "probes", "results")
GT = os.path.join(RES, "gt")
DATA = os.path.join(ROOT, "data")
BASE = r"F:\paper\SynthFringe"
spec = importlib.util.spec_from_file_location("w3", os.path.join(HERE, "w3_gen.py"))
W3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(W3)

BLOCK, BUFFER = 200, 30
BLOCKS = {"val": range(12, 15), "test": range(15, 18)}
VINF = ["sliver", "arc", "blob"]
OTHER = ["edge", "cut", "degraded"]
FAM = VINF + OTHER
SLIVER_LO, SLIVER_HI, SLIVER_TOTAL = 130, 800, 400


# ------------------------------------------------------------------ scene i/o
def load_scene(sid):
    """Load one 256x256 crop of a SynthFringe scene; returns (phi, roi, amp)."""
    p = os.path.join(GT, f"{sid}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    phi = d["phi64"].astype(np.float64)
    roi_full = d["m_gt"].astype(bool)
    amp = d["amp64"].astype(np.float64)
    H, W = phi.shape
    ys, xs = np.where(roi_full)
    if len(ys) < 200:
        return None
    cy, cx = int(np.median(ys)), int(np.median(xs))
    y0 = int(np.clip(cy - 128, 0, H - 256))
    x0 = int(np.clip(cx - 128, 0, W - 256))
    phi = phi[y0:y0 + 256, x0:x0 + 256]
    roi = roi_full[y0:y0 + 256, x0:x0 + 256]
    amp = amp[y0:y0 + 256, x0:x0 + 256]
    if roi.mean() < 0.12:
        return None
    return phi, roi, amp


# ------------------------------------------------------------------ hole placement
def hole_sliver_abs(rng, H, W, roi, lo=SLIVER_LO, hi=SLIVER_HI):
    """Straight thin hole with an ABSOLUTE pixel-area window (domain invariant)."""
    yy, xx = np.mgrid[0:H, 0:W]
    w = int(rng.integers(4, 9)); L = int(rng.integers(30, 61))
    c = W3._sample_center(rng, roi, w / 2 + 5)
    if c is None:
        return None
    for _ in range(24):
        cy, cx = c
        a = rng.uniform(0, np.pi)
        u = (yy - cy) * np.cos(a) + (xx - cx) * np.sin(a)
        v = -(yy - cy) * np.sin(a) + (xx - cx) * np.cos(a)
        m = (np.abs(u) <= L / 2) & (np.abs(v) <= w / 2) & roi
        ar = int(m.sum())
        if lo <= ar <= hi and W3._fits(m, roi, 1):
            return m
        c = W3._sample_center(rng, roi, w / 2 + 5)
        if c is None:
            return None
    return None


def place_holes(rng, roi, branch):
    """Place the dead zone for one non-degraded branch.  Returns a bool mask or None."""
    if branch == "sliver":
        fn = hole_sliver_abs
    else:
        fn = {"arc": W3.hole_arc, "blob": W3.hole_blob,
              "edge": W3.hole_edge, "cut": W3.hole_cut}[branch]
    n_h = 1 if branch in ("blob", "cut") else int(rng.integers(1, 5))
    f_lo, f_hi = (0.03, 0.16) if branch == "cut" else (0.008, 0.035)
    dead = np.zeros(roi.shape, bool); got = 0
    for _ in range(10):
        if got >= n_h:
            break
        m = fn(rng, roi.shape[0], roi.shape[1], roi)
        if m is None:
            continue
        if branch == "sliver":
            ar = int(m.sum())
            if not (SLIVER_LO <= ar <= SLIVER_HI):
                continue
        else:
            f = m.sum() / max(roi.sum(), 1)
            if not (f_lo <= f <= f_hi):
                continue
        dead |= m; got += 1
    if got == 0:
        return None
    if branch == "sliver":
        if dead.sum() < SLIVER_TOTAL:
            return None
    else:
        if dead.sum() / max(roi.sum(), 1) < 0.02:
            return None
    thr_img = bool(dead[0, :].any() or dead[-1, :].any() or dead[:, 0].any() or dead[:, -1].any())
    thr_roi = bool((cv2.dilate(dead.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)[~roi].any())
    if branch in ("sliver", "arc", "blob") and (thr_img or thr_roi):
        return None
    return dead


# ------------------------------------------------------------------ sample builder
def make_sf(sid, rng, branch, scene=None, dead=None):
    if scene is None:
        scene = load_scene(sid)
    if scene is None:
        return None
    phi, roi, amp = scene
    H, W = phi.shape
    A = 20.0
    I = np.clip(A + amp * np.cos(phi), 0, 255)

    if branch == "degraded":
        dead = np.zeros((H, W), bool)
        I_d = cv2.GaussianBlur(I.astype(np.float32), (0, 0), float(rng.uniform(1.5, 3.0)))
        nholes = 0
    else:
        if dead is None:
            dead = place_holes(rng, roi, branch)
        if dead is None:
            return None
        soft = np.clip(cv2.GaussianBlur(dead.astype(np.float32), (0, 0), 2.0) * 1.8, 0, 1)
        R_amb = rng.uniform(0.30, 0.80)
        I_d = (1 - soft) * I + soft * (R_amb * A)
        I_d = I_d + soft * rng.normal(0, rng.uniform(0.8, 2.5), (H, W))
        nholes = int(cv2.connectedComponents(dead.astype(np.uint8), 8)[0] - 1)

    I_u8 = np.clip(I_d, 0, 255).astype(np.uint8)
    clean = roi & (~dead)
    kp50, kp95 = W3.kappa_mlq(phi, clean)
    kmap, kok = W3.kappa_mlq_map(phi, clean)
    orr = W3.orr_reference(dead, roi, phi)
    vinf = W3.torsion_vinf(dead, roi) if branch in VINF else None

    lap_gt = cv2.Laplacian(phi.astype(np.float32), cv2.CV_32F)
    k2 = np.ones((2 * W3.COLLAR + 1,) * 2, np.uint8)
    dom = (cv2.dilate(dead.astype(np.uint8), k2) > 0) & roi
    collar = dom & (~dead)
    QS = (50, 75, 90, 95, 99)
    kps = [float(np.percentile(np.abs(lap_gt[dom]), q)) for q in QS] if dom.sum() > 20 else [0.02] * 5
    khps = [float(np.percentile(np.abs(lap_gt[collar]), q)) for q in QS] if collar.sum() > 20 else [0.02] * 5
    KAP = khps[1]

    cls = "U" if branch in ("edge", "cut") else "R"
    if branch in VINF + ["edge", "cut"]:
        cls = "P" if (vinf is not None and (KAP * vinf + W3.DELTA_G) / W3.PHI_TOL <= 1.0) else "U"
    rho = (KAP * vinf + W3.DELTA_G) / W3.PHI_TOL if vinf is not None else None

    return dict(id=f"SF{sid}_{branch}", sid=int(sid), block=int(sid) // BLOCK,
                src="SynthFringe", branch=branch, bg="sf", T=np.nan,
                high_k=False, meta=dict(kind=branch, nholes=nholes),
                cls=cls, cert=bool(rho is not None and rho <= 1.0),
                rho=None if rho is None else float(rho),
                vinf=None if vinf is None else float(vinf),
                kappa50=kp50, kappa95=kp95, kappa_oracle=kps[3], kappa_hat=khps[1],
                kappa_ps=kps, kappa_hat_ps=khps, comps=[],
                rho_oracle=None, rho_hat=None, grad_over_f0=0.0,
                roi=float(roi.mean()), dead=float(dead.sum() / max(roi.sum(), 1)),
                I=I_u8, phi=phi.astype(np.float32), roi_m=roi, dead_m=dead,
                orr=orr.astype(np.float32), kmap=np.where(kok, kmap, 0).astype(np.float16))


# ------------------------------------------------------------------ unique-scene assignment
def capability(sid, scene, seed):
    """One independent placement attempt per vinf family (no image rendering)."""
    out = {}
    for j, br in enumerate(VINF):
        rng = np.random.default_rng(seed + int(sid) * 17 + j * 7919)
        out[br] = place_holes(rng, scene[1], br)
    return out


def assign_unique(scenes, caps, n_per):
    """Max-flow: source->scene(1), scene->family(1 if capable), family->sink(n_per)."""
    N = len(scenes)
    S, T = 0, N + 4
    rows, cols, data = [], [], []
    for i in range(N):
        rows.append(S); cols.append(1 + i); data.append(1)
        for j, br in enumerate(VINF):
            if caps[i].get(br) is not None:
                rows.append(1 + i); cols.append(N + 1 + j); data.append(1)
    for j in range(3):
        rows.append(N + 1 + j); cols.append(T); data.append(n_per)
    g = sp.csr_matrix((np.asarray(data, np.int32), (rows, cols)), shape=(N + 5, N + 5))
    res = maximum_flow(g, S, T)
    if res.flow_value < 3 * n_per:
        return None, int(res.flow_value)
    flow = res.flow.toarray()
    assigned = []
    for i in range(N):
        for j, br in enumerate(VINF):
            if flow[1 + i, N + 1 + j] > 0:
                assigned.append((scenes[i], br, caps[i][br]))
    return assigned, int(res.flow_value)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n-vinf", type=int, default=150)
    ap.add_argument("--n-other", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20261117)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed + (0 if a.split == "val" else 1))
    ids = []
    for b in BLOCKS[a.split]:
        ids += [f"{i:05d}" for i in range(b * BLOCK + BUFFER // 2, (b + 1) * BLOCK - BUFFER // 2)]
    ids = [s for s in ids if os.path.exists(os.path.join(GT, s + ".npz"))]
    rng.shuffle(ids)
    print(f"[{a.split}] candidate scenes: {len(ids)}", flush=True)

    # ---- pass 1: capability (hole placement only, cheap)
    scenes, caps = [], []
    for k, sid in enumerate(ids):
        sc = load_scene(sid)
        if sc is None:
            continue
        scenes.append(sid)
        caps.append(capability(sid, sc, a.seed))
        if (k + 1) % 100 == 0:
            print(f"  capability {k + 1}/{len(ids)}", flush=True)
    cnt = {br: sum(1 for c in caps if c[br] is not None) for br in VINF}
    print(f"  capable: {cnt} / {len(scenes)} scenes", flush=True)

    assigned, flow = assign_unique(scenes, caps, a.n_vinf)
    if assigned is None:
        raise RuntimeError(f"unique-scene assignment infeasible: max-flow={flow} < {3*a.n_vinf}")
    print(f"  max-flow={flow}; assigned {len(assigned)} vinf samples "
          f"({dict(Counter(b for _, b, _ in assigned))})", flush=True)

    # ---- pass 2: render samples and write shards
    outdir = os.path.join(DATA, f"sf_{a.split}")
    os.makedirs(outdir, exist_ok=True)
    for f in glob.glob(os.path.join(outdir, "shard_*.npz")):
        os.remove(f)

    buf, shards, all_meta = [], [], []

    def flush():
        if not buf:
            return
        p = os.path.join(outdir, f"shard_{len(shards):04d}.npz")
        h = W3.shard_write(buf, p)
        shards.append(dict(path=os.path.basename(p), n=len(buf), sha256=h))
        print(f"  -> {os.path.basename(p)} n={len(buf)} sha256={h[:16]}...", flush=True)
        buf.clear()

    def add(r):
        meta = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in r.items() if not hasattr(v, "shape")}
        all_meta.append(meta)
        buf.append(r)
        if len(buf) >= 120:
            flush()

    for sid, br, dead in assigned:
        sc = load_scene(sid)
        rng_g = np.random.default_rng(a.seed + int(sid) * 29 + VINF.index(br) * 100003)
        r = make_sf(sid, rng_g, br, scene=sc, dead=dead)
        if r is None:
            raise RuntimeError(f"assigned sample failed to render: {sid} {br}")
        add(r)
    print(f"  vinf rendered: {len(all_meta)}", flush=True)

    # non-vinf: for zero-shot error only; reuse is allowed.  Prefer the scenes
    # the vinf assignment left untouched (60 of 510), then reuse the rest.
    used = {sid for sid, _, _ in assigned}
    queue0 = [s for s in scenes if s not in used] + [s for s in scenes if s in used]
    for br in OTHER:
        q = list(queue0); rng_q = np.random.default_rng(a.seed + 500 + OTHER.index(br))
        rng_q.shuffle(q)
        got, seen = 0, set()
        for sid in q:
            if got >= a.n_other:
                break
            if (sid, br) in seen:
                continue
            seen.add((sid, br))
            sc = load_scene(sid)
            if sc is None:
                continue
            rng_g = np.random.default_rng(a.seed + int(sid) * 31 + (10 + OTHER.index(br)) * 100003)
            r = make_sf(sid, rng_g, br, scene=sc)
            if r is None:
                continue
            add(r); got += 1
        print(f"  {br}: {got}/{a.n_other}", flush=True)
        if got < a.n_other:
            print(f"  WARNING: {br} only {got}/{a.n_other}", flush=True)
    flush()

    per = Counter(m["branch"] for m in all_meta)
    n_vinf = sum(1 for m in all_meta if m.get("vinf") is not None)
    n_uniq_vinf = len({m["sid"] for m in all_meta if m.get("vinf") is not None})
    n_uniq = len({m["sid"] for m in all_meta})

    man = dict(split=a.split, blocks=[int(b) for b in BLOCKS[a.split]], buffer=BUFFER,
               n_samples=len(all_meta), n_vinf=n_vinf, n_unique_vinf=n_uniq_vinf,
               n_unique_scenes=n_uniq, per_family=dict(per), shards=shards,
               src="SynthFringe 64_15 (GT-validated)",
               protocol=dict(n_vinf_per_family=a.n_vinf, n_other_per_family=a.n_other,
                             sliver_area_px=[SLIVER_LO, SLIVER_HI], sliver_total_px=SLIVER_TOTAL,
                             nholes=[1, 4], vinf_assignment="one sample per scene; max-flow"))
    json.dump(man, open(os.path.join(outdir, "manifest.json"), "w"), indent=1)
    all_meta.sort(key=lambda m: (VINF.index(m["branch"]) if m["branch"] in VINF else 3, m["id"]))
    json.dump(all_meta, open(os.path.join(RES, f"sf_{a.split}_meta.json"), "w"), indent=1)

    print(f"\n{a.split}: {len(all_meta)} samples, per-family={dict(per)}")
    print(f"  vinf={n_vinf}, unique vinf scenes={n_uniq_vinf}, unique scenes={n_uniq}")
    cs = [m for m in all_meta if m.get("rho") is not None]
    if cs:
        print(f"  Tier-1 at gamma=1: {sum(m['cert'] for m in cs)}/{len(cs)} "
              f"({100*np.mean([m['cert'] for m in cs]):.1f}%)")
    print("manifest:", os.path.join(outdir, "manifest.json"))


if __name__ == "__main__":
    main()
