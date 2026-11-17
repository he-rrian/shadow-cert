"""Probe B v2
(1) SynthFringe: mother-model stride signature, with speckle removed.
    - mask = max over 3 frames > 25, 3x3 opening, keep components with area >= 500 px
    - signature per scene: number of large components, sorted area vector, bbox aspect,
      Phi_64 span, mean |grad Phi|
    - autocorrelation of the area sequence over 1200 scenes, lags 1..600
    - explicit test: are scenes i and i+k (k = 1..6) the "same" arrangement rotated 60 deg?
(2) Real scenes: frame census, kappa_real, phi span, signed sub-pixel fringe direction,
    isolated-island count.
Run: python probeB_v2.py
"""
import os, json
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "probes", "results")
BASE = r"F:\paper\SynthFringe"
F1 = r"F:\paper\1\phase"
REAL = r"F:\paper\2\model\real\captures\proc"

def im(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

def gray(p):
    a = im(p)
    if a is None:
        return None
    if a.ndim == 3:
        a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2GRAY)
    return a.astype(np.float64)

def clean_mask(g, thr=25, amin=500):
    m = (g > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    areas = []
    for k in range(1, n):
        if st[k, 4] >= amin:
            keep[lab == k] = 1
            areas.append(int(st[k, 4]))
    return keep.astype(bool), sorted(areas, reverse=True)

def scene_sig(sid):
    p = os.path.join(BASE, "64_15", sid)
    if not os.path.isdir(p):
        return None
    fs = sorted(x for x in os.listdir(p) if x.endswith(".bmp"))[:3]
    g = np.max([gray(os.path.join(p, f)) for f in fs], 0)
    m, areas = clean_mask(g)
    if not m.any():
        return None
    ys, xs = np.where(m)
    return dict(sid=sid, area=float(m.mean()), n=len(areas), areas=areas[:8],
                aspect=float((xs.max() - xs.min() + 1) / max(ys.max() - ys.min() + 1, 1)))

def autocorr(lags, seq):
    s = np.asarray(seq, float)
    s = (s - s.mean()) / (s.std() + 1e-12)
    return [(L, float(np.mean(s[:-L] * s[L:]))) for L in lags if L < len(s) // 2]

def part1():
    N = 1200
    rows = []
    for i in range(N):
        r = scene_sig(f"{i:05d}")
        if r:
            rows.append(r)
    area = [r["area"] for r in rows]
    ncomp = [r["n"] for r in rows]
    print(f"[B1] scenes sampled = {len(rows)}")
    print(f"     cleaned object area: P5={np.percentile(area,5):.3f} P50={np.median(area):.3f} "
          f"P95={np.percentile(area,95):.3f}")
    print(f"     large components    : P50={np.median(ncomp):.0f}  P95={np.percentile(ncomp,95):.0f}  max={max(ncomp)}")
    ac = autocorr(range(1, 600), area)
    top = sorted(ac, key=lambda t: -t[1])[:10]
    print("     top autocorr lags of AREA (candidate strides):")
    for L, v in top:
        print(f"       lag {L:4d}  r={v:+.3f}")
    for L in (100, 200, 300, 400, 500, 600, 700, 720):
        hit = [v for l, v in ac if l == L]
        if hit:
            print(f"       lag {L:4d}  r={hit[0]:+.3f}")
    ac2 = autocorr(range(1, 600), ncomp)
    top2 = sorted(ac2, key=lambda t: -t[1])[:6]
    print("     top autocorr lags of COMPONENT COUNT:")
    for L, v in top2:
        print(f"       lag {L:4d}  r={v:+.3f}")
    # consecutive-6 test on the sorted area vectors
    print("\n     sorted-area-vector distance, scenes i vs i+k (normalised, lower = more alike):")
    base = {r["sid"]: r for r in rows}
    for k in (1, 2, 3, 4, 5, 6, 500):
        ds = []
        for i in range(30):
            a = base.get(f"{i:05d}"); b = base.get(f"{i+k:05d}")
            if not a or not b or not a["areas"] or not b["areas"]:
                continue
            va = np.array(a["areas"], float); vb = np.array(b["areas"], float)
            m = min(len(va), len(vb))
            va = va[:m] / va[:m].sum(); vb = vb[:m] / vb[:m].sum()
            ds.append(float(np.abs(va - vb).sum()))
        if ds:
            print(f"       k={k:3d}: mean |dA| = {np.mean(ds):.4f}  (n={len(ds)})")
    return rows

def kappa_of(phi, mask):
    lap = cv2.Laplacian(np.where(mask, phi, 0).astype(np.float32), cv2.CV_32F)
    v = np.abs(lap[mask])
    if v.size == 0:
        return None
    return (float(np.median(v)), float(np.percentile(v, 95)))

def fringe_dir(p):
    """signed sub-pixel peak of the 2-D spectrum"""
    g = gray(p)
    if g is None:
        return None
    F = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))
    H, W = g.shape
    F[H // 2 - 8:H // 2 + 9, W // 2 - 8:W // 2 + 9] = 0
    iy, ix = np.unravel_index(np.argmax(F), F.shape)
    fu = (ix - W // 2) / W
    fv = (iy - H // 2) / H
    # parabolic sub-bin refinement along u
    def refine(a, i):
        if 0 < i < len(a) - 1:
            d = 0.5 * (a[i - 1] - a[i + 1]) / (a[i - 1] - 2 * a[i] + a[i + 1] + 1e-12)
            return (i + d - len(a) // 2) / len(a)
        return (i - len(a) // 2) / len(a)
    fu = refine(F[iy, :], ix)
    fv = refine(F[:, ix], iy)
    return fu, fv

def part2():
    print("\n[B2] real scenes")
    out = []
    # ---- Train (15-step GT)
    p = os.path.join(F1, "Train")
    if os.path.isdir(p):
        gt = np.load(os.path.join(p, "gt15_rad.npy"))
        mask = np.isfinite(gt)
        du = fringe_dir(os.path.join(p, "00.bmp"))
        k = kappa_of(gt, mask)
        n = cv2.connectedComponents(mask.astype(np.uint8), 8)[0] - 1
        print(f"  Train   : 15-step GT, valid={mask.mean()*100:.1f}%  span=[{np.nanmin(gt):.1f},"
              f"{np.nanmax(gt):.1f}]  kappa P50/P95 = {k[0]:.4f}/{k[1]:.4f} rad/px^2  "
              f"f=({du[0]:+.4f},{du[1]:+.4f})  islands={n}")
        out.append(dict(name="Train", valid=float(mask.mean()), kappa=k, fu=du[0], fv=du[1], islands=n,
                        span=float(np.nanmax(gt) - np.nanmin(gt))))
    # ---- scene_01..06
    for s in sorted(os.listdir(REAL)):
        gp = os.path.join(REAL, s, "gt_phase.npy")
        if not os.path.exists(gp):
            continue
        gt = np.load(gp)
        mask = np.isfinite(gt)
        k = kappa_of(gt, mask)
        du = fringe_dir(os.path.join(REAL, s, "capture_1.png"))
        n = cv2.connectedComponents(mask.astype(np.uint8), 8)[0] - 1
        print(f"  {s:7s}: 4-step GT, valid={mask.mean()*100:.1f}%  span={np.nanmax(gt)-np.nanmin(gt):.1f}  "
              f"kappa P50/P95 = {k[0]:.4f}/{k[1]:.4f}  f=({du[0]:+.4f},{du[1]:+.4f})  islands={n}")
        out.append(dict(name=s, valid=float(mask.mean()), kappa=k, fu=du[0], fv=du[1], islands=n,
                        span=float(np.nanmax(gt) - np.nanmin(gt))))
    # ---- 实验3 / 实验4 frame census
    for d in ("实验3", "实验4"):
        p = os.path.join(F1, d)
        if not os.path.isdir(p):
            continue
        fs = sorted(os.listdir(p))
        imgs = [f for f in fs if f.lower().endswith((".jpg", ".png", ".bmp"))]
        print(f"  {d}: {len(imgs)} images")
        for f in imgs:
            g = gray(os.path.join(p, f))
            print(f"      {f:32s} {None if g is None else g.shape}")
        # try an N-step demod on the 'shice' frames
        sh = [f for f in imgs if "shice" in f and "crop" not in f]
        cr = [f for f in imgs if "crop" in f]
        for tag, lst in (("full", sh), ("crop", cr)):
            if len(lst) < 4:
                continue
            I = [gray(os.path.join(p, f)) for f in lst[:min(len(lst), 5)]]
            I = [x for x in I if x is not None]
            if len(I) < 4:
                continue
            shape = I[0].shape
            I = [x if x.shape == shape else cv2.resize(x, (shape[1], shape[0])) for x in I]
            I = np.stack(I)
            N = I.shape[0]
            d0 = 2 * np.pi * np.arange(N) / N
            s_ = np.tensordot(np.sin(d0), I, axes=(0, 0)); c_ = np.tensordot(np.cos(d0), I, axes=(0, 0))
            ph = np.arctan2(s_, c_)
            amp = np.hypot(s_, c_) / N
            mk = amp > np.percentile(amp, 40)
            print(f"      [{tag}] N={N} -> wrapped-phase range=[{ph[mk].min():.2f},{ph[mk].max():.2f}] "
                  f"amp P50={np.median(amp[mk]):.1f}  valid={mk.mean()*100:.1f}%")
    json.dump(out, open(os.path.join(RES, "probeB_v2.json"), "w"), indent=1)

if __name__ == "__main__":
    part1()
    part2()
