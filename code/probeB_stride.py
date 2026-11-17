"""Probe B -- (1) does SynthFringe reuse the same CAD models at a fixed stride?
               (2) the real scenes: fringe period T_real and surface complexity.

(1) signature = object area + mean intensity, one frame per scene, over scenes 0..399.
    If the dataset is "500 models x N rotations" the area sequence is periodic; we look for
    a peak in the normalised autocorrelation over lags 1..400.
(2) decode 1\\phase\\实测1/实测2/实验3/实验4/Train: frames, size, GT availability,
    2-D-spectrum fringe period, and the phase curvature kappa (ppm of a rad/px^2).
Run: python probeB_stride.py
"""
import os, json
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "probes", "results")
BASE = r"F:\paper\SynthFringe"
F1 = r"F:\paper\1\phase"

def im(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

def gray(p):
    a = im(p)
    if a is None:
        return None
    if a.ndim == 3:
        a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2GRAY)
    return a.astype(np.float64)

def sig(sid):
    p = os.path.join(BASE, "64_15", sid)
    if not os.path.isdir(p):
        return None
    fs = sorted(x for x in os.listdir(p) if x.endswith(".bmp"))
    g = gray(os.path.join(p, fs[0]))
    obj = g > 25
    return dict(sid=sid, area=float(obj.mean()), mean=float(g[obj].mean()) if obj.any() else 0.0,
                ncomp=int(cv2.connectedComponents(obj.astype(np.uint8), 8)[0] - 1))

def autocorr_peak(seq, lags):
    s = np.asarray(seq, float)
    s = (s - s.mean()) / (s.std() + 1e-12)
    out = []
    for L in lags:
        if L >= len(s) // 2:
            break
        out.append((L, float(np.mean(s[:-L] * s[L:]))))
    return out

if __name__ == "__main__":
    N = 400
    rows = []
    for i in range(N):
        r = sig(f"{i:05d}")
        if r:
            rows.append(r)
    area = [r["area"] for r in rows]
    comp = [r["ncomp"] for r in rows]
    print(f"scenes sampled: {len(rows)}")
    print(f"  area  : P5={np.percentile(area,5):.3f} P50={np.median(area):.3f} P95={np.percentile(area,95):.3f}")
    print(f"  ncomp : P50={np.median(comp):.0f} max={max(comp)}")
    ac = autocorr_peak(area, range(1, 200))
    ac.sort(key=lambda t: -t[1])
    print("  top autocorrelation lags of the AREA sequence (candidate model strides):")
    for L, v in ac[:8]:
        print(f"     lag {L:4d}  r = {v:+.3f}")
    ac2 = autocorr_peak(comp, range(1, 200))
    ac2.sort(key=lambda t: -t[1])
    print("  top lags for the COMPONENT-COUNT sequence:")
    for L, v in ac2[:5]:
        print(f"     lag {L:4d}  r = {v:+.3f}")

    print("\n--- real scenes ---")
    real = []
    for d in ("Train", "实测1", "实测2", "实验3", "实验4"):
        p = os.path.join(F1, d)
        if not os.path.isdir(p):
            continue
        fs = sorted(os.listdir(p))
        frames = [f for f in fs if f.lower().endswith((".bmp", ".png", ".jpg"))]
        mats = [f for f in fs if f.endswith(".mat")]
        npy = [f for f in fs if f.endswith(".npy")]
        g = None
        for f in frames:
            g = gray(os.path.join(p, f))
            if g is not None:
                break
        rec = dict(dir=d, frames=len(frames), mats=len(mats), npy=npy,
                   shape=None if g is None else list(g.shape))
        if g is not None:
            F2 = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))
            H, W = g.shape
            F2[H // 2 - 8:H // 2 + 9, W // 2 - 8:W // 2 + 9] = 0
            iy, ix = np.unravel_index(np.argmax(F2), F2.shape)
            fu = (ix - W // 2) / W
            fv = (iy - H // 2) / H
            T = 1 / max(np.hypot(fu, fv), 1e-6)
            rec["T_px"] = float(T)
            rec["fringe_axis"] = "u" if abs(fu) > abs(fv) else "v"
        real.append(rec)
        print(f"  {d:6s} frames={len(frames):3d} mats={len(mats):2d} npy={len(npy):2d} "
              f"shape={rec['shape']} T={rec.get('T_px', 0):.2f} px axis={rec.get('fringe_axis','-')}")
    json.dump(dict(synth=rows, real=real), open(os.path.join(RES, "probeB.json"), "w"), indent=1)
    print("\nsaved", os.path.join(RES, "probeB.json"))
