"""w3_export_sf.py -- SynthFringe external benchmark export (ZERO-SHOT test set).

Blocks 12..14 -> data/sf_val   (used ONLY for target-domain recalibration of tau_c / lambda_hat)
Blocks 15..17 -> data/sf_test  (used ONLY for the final zero-shot numbers)
A 30-scene isolation buffer is dropped at every block boundary; blocks 0..11 are NEVER touched,
so the network (trained on MATLAB only) has never seen a single SynthFringe frame.

Same schema / same orr_reference / same kappa_MLQ / same no-border rule as the MATLAB shards.

Run: python w3_export_sf.py --split val   [--split test]
"""
import os, sys, json, argparse, hashlib
import numpy as np, cv2
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
FAM = ["sliver", "arc", "blob", "edge", "cut", "degraded"]


def load_seq(sid):
    p = os.path.join(BASE, "64_15", sid)
    fs = sorted(x for x in os.listdir(p) if x.endswith(".bmp"))
    out = []
    for f in fs:
        a = cv2.imdecode(np.fromfile(os.path.join(p, f), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if a.ndim == 3:
            a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2GRAY)
        out.append(a.astype(np.float64))
    return np.stack(out)


def make_sf(sid, rng, branch):
    p = os.path.join(GT, f"{sid}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    phi = d["phi64"].astype(np.float64); roi_full = d["m_gt"]; amp = d["amp64"].astype(np.float64)
    H, W = phi.shape
    ys, xs = np.where(roi_full)
    if len(ys) < 200:
        return None
    cy, cx = int(np.median(ys)), int(np.median(xs))
    y0 = int(np.clip(cy - 128, 0, H - 256)); x0 = int(np.clip(cx - 128, 0, W - 256))
    phi = phi[y0:y0 + 256, x0:x0 + 256]; roi = roi_full[y0:y0 + 256, x0:x0 + 256]
    amp = amp[y0:y0 + 256, x0:x0 + 256]
    if roi.mean() < 0.12:
        return None
    A = 20.0
    I = np.clip(A + amp * np.cos(phi), 0, 255)
    # hole
    if branch == "degraded":
        dead = np.zeros((256, 256), bool); meta = dict(kind="degraded")
        I_d = cv2.GaussianBlur(I.astype(np.float32), (0, 0), float(rng.uniform(1.5, 3.0)))
    else:
        fn = {"sliver": W3.hole_sliver, "arc": W3.hole_arc, "blob": W3.hole_blob,
              "edge": W3.hole_edge, "cut": W3.hole_cut}[branch]
        dead = np.zeros((256, 256), bool); got = 0
        f_lo, f_hi = (0.03, 0.16) if branch == "cut" else (0.008, 0.035)
        n_h = 1 if branch in ("blob", "cut") else int(rng.integers(1, 4))
        for _ in range(10):
            if got >= n_h:
                break
            m = fn(rng, 256, 256, roi)
            if m is None:
                continue
            f = m.sum() / max(roi.sum(), 1)
            if not (f_lo <= f <= f_hi):
                continue
            dead |= m; got += 1
        if got == 0 or dead.sum() / max(roi.sum(), 1) < 0.02:
            return None
        thr_img = bool(dead[0, :].any() or dead[-1, :].any() or dead[:, 0].any() or dead[:, -1].any())
        thr_roi = bool((cv2.dilate(dead.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)[~roi].any())
        if branch in ("sliver", "arc", "blob") and (thr_img or thr_roi):
            return None
        meta = dict(kind=branch, nholes=got)
        soft = np.clip(cv2.GaussianBlur(dead.astype(np.float32), (0, 0), 2.0) * 1.8, 0, 1)
        R_amb = rng.uniform(0.30, 0.80)
        I_d = (1 - soft) * I + soft * (R_amb * A)
        I_d = I_d + soft * rng.normal(0, rng.uniform(0.8, 2.5), (256, 256))
    I_u8 = np.clip(I_d, 0, 255).astype(np.uint8)
    clean = roi & (~dead)
    kp50, kp95 = W3.kappa_mlq(phi, clean)
    kmap, kok = W3.kappa_mlq_map(phi, clean)
    orr = W3.orr_reference(dead, roi, phi)
    vinf = W3.torsion_vinf(dead, roi) if branch in ("sliver", "arc", "blob") else None
    lap_gt = cv2.Laplacian(phi.astype(np.float32), cv2.CV_32F)
    k2 = np.ones((2 * W3.COLLAR + 1,) * 2, np.uint8)
    dom = (cv2.dilate(dead.astype(np.uint8), k2) > 0) & roi
    collar = dom & (~dead)
    QS = (50, 75, 90, 95, 99)
    kps = [float(np.percentile(np.abs(lap_gt[dom]), q)) for q in QS] if dom.sum() > 20 else [0.02] * 5
    khps = [float(np.percentile(np.abs(lap_gt[collar]), q)) for q in QS] if collar.sum() > 20 else [0.02] * 5
    KAP = khps[1]
    cls = "U" if branch in ("edge", "cut") else "R"
    if branch in ("sliver", "arc", "blob", "edge", "cut"):
        cls = "P" if (vinf is not None and (KAP * vinf + W3.DELTA_G) / W3.PHI_TOL <= 1.0) else "U"
    rho = (KAP * vinf + W3.DELTA_G) / W3.PHI_TOL if vinf is not None else None
    return dict(id=f"SF{sid}_{branch}", src="SynthFringe", branch=branch, bg="sf", T=np.nan,
                high_k=False, meta=meta, cls=cls, cert=bool(rho is not None and rho <= 1.0),
                rho=None if rho is None else float(rho), vinf=None if vinf is None else float(vinf),
                kappa50=kp50, kappa95=kp95, kappa_oracle=kps[3], kappa_hat=khps[1],
                kappa_ps=kps, kappa_hat_ps=khps, comps=[],
                rho_oracle=None, rho_hat=None, grad_over_f0=0.0,
                roi=float(roi.mean()), dead=float(dead.sum() / max(roi.sum(), 1)),
                I=I_u8, phi=phi.astype(np.float32), roi_m=roi, dead_m=dead,
                orr=orr.astype(np.float32), kmap=np.where(kok, kmap, 0).astype(np.float16))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--per-family", type=int, default=80)
    a = ap.parse_args()
    rng = np.random.default_rng(20261117 + (0 if a.split == "val" else 1))
    ids = []
    for b in BLOCKS[a.split]:
        lo, hi = b * BLOCK, (b + 1) * BLOCK
        ids += list(range(lo + BUFFER // 2, hi - BUFFER // 2))
    outdir = os.path.join(DATA, f"sf_{a.split}"); os.makedirs(outdir, exist_ok=True)
    quota = {b: a.per_family for b in FAM}
    fail = {b: 0 for b in FAM}
    MAX_FAIL = 300                      # a family that cannot be generated must not starve the queue
    rows, buf, shards = [], [], []
    attempt = 0
    max_attempt = 50 * sum(quota.values())
    while sum(quota.values()) > 0 and attempt < max_attempt:
        avail = [b for b in FAM if quota[b] > 0 and fail[b] < MAX_FAIL]
        if not avail:
            break
        br = avail[attempt % len(avail)]
        attempt += 1
        sid = f"{ids[int(rng.integers(0, len(ids)))]:05d}"
        r = make_sf(sid, rng, br)
        if r is None:
            fail[br] += 1
            continue
        quota[br] -= 1
        fail[br] = 0
        rows.append({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                     for k, v in r.items() if not hasattr(v, "shape")})
        buf.append(r)
        if len(buf) >= 120:
            p = os.path.join(outdir, f"shard_{len(shards):04d}.npz")
            h = W3.shard_write(buf, p)
            shards.append(dict(path=os.path.basename(p), n=len(buf), sha256=h))
            print(f"  -> {os.path.basename(p)} n={len(buf)} sha256={h[:16]}...")
            buf.clear()
    if buf:
        p = os.path.join(outdir, f"shard_{len(shards):04d}.npz")
        h = W3.shard_write(buf, p)
        shards.append(dict(path=os.path.basename(p), n=len(buf), sha256=h))
        print(f"  -> {os.path.basename(p)} n={len(buf)} sha256={h[:16]}...")
    man = dict(split=a.split, blocks=[int(b) for b in BLOCKS[a.split]], buffer=BUFFER,
               n_samples=len(rows), shards=shards, src="SynthFringe 64_15 (GT-validated)")
    json.dump(man, open(os.path.join(outdir, "manifest.json"), "w"), indent=1)
    json.dump(rows, open(os.path.join(RES, f"sf_{a.split}_meta.json"), "w"), indent=1)
    from collections import Counter
    print(f"\n{a.split}: {len(rows)} samples", dict(Counter(r["branch"] for r in rows)))
    cs = [r for r in rows if r["rho"] is not None]
    if cs:
        print(f"  Tier-1 at gamma=1: {sum(r['cert'] for r in cs)}/{len(cs)} "
              f"({100*np.mean([r['cert'] for r in cs]):.1f}%)")
    print("manifest:", os.path.join(outdir, "manifest.json"))
