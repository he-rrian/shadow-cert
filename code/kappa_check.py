"""Verification of the kappa anomaly reported in probe B v2.

Hypothesis (from the review): scene_01..06 are 4-step JPEG images with 0.06-2.1 % of pixels
having |grad gt| > pi (unwrap tears).  A bare discrete Laplacian amplifies phase noise by
sqrt(20) ~ 4.5x, which can turn 0.05-0.10 rad of noise into a fake 0.2-0.4 rad/px^2 curvature.

We recompute kappa three ways over SynthFringe (clean, 15-step) and the real scenes:
    (a) raw discrete Laplacian
    (b) Savitzky-Golay (quadratic, window 5 / 7) + Laplacian
    (c) (b) with the tear pixels (|grad gt| > pi) removed
Run: python kappa_check.py
"""
import os
import numpy as np, cv2
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GT = os.path.join(ROOT, "probes", "results", "gt")
REAL = r"F:\paper\2\model\real\captures\proc"

def grad_mag(phi):
    gy, gx = np.gradient(phi)
    return np.hypot(gx, gy)

def kappa(phi, mask, win=None):
    z = np.where(mask, phi, 0.0).astype(np.float64)
    if win:
        z = savgol_filter(z, win, 2, mode="nearest")
    lap = cv2.Laplacian(z.astype(np.float32), cv2.CV_32F)
    v = np.abs(lap[mask])
    if v.size < 50:
        return None
    return float(np.median(v)), float(np.percentile(v, 95))

def report(name, phi, mask):
    g = grad_mag(phi)
    tear = mask & (g > np.pi)
    m2 = mask & (~tear)
    out = []
    for tag, w in (("raw", None), ("SG5", 5), ("SG7", 7)):
        r = kappa(phi, m2, w)
        out.append(f"{tag}: {r[0]:.4f}/{r[1]:.4f}" if r else f"{tag}: -")
    print(f"{name:10s} valid={m2.mean()*100:5.1f}%  tears={tear.sum()/max(mask.sum(),1)*100:4.2f}%  "
          + "  ".join(out))

if __name__ == "__main__":
    print("kappa P50/P95 (rad/px^2), tears (|grad|>pi) removed, three estimators\n")
    for sid in ("00000", "00360", "01500"):
        p = os.path.join(GT, f"{sid}.npz")
        if os.path.exists(p):
            d = np.load(p)
            report(f"SF {sid}", d["phi64"].astype(np.float64), d["m_gt"])
    for s in sorted(os.listdir(REAL)):
        p = os.path.join(REAL, s, "gt_phase.npy")
        if not os.path.exists(p):
            continue
        g = np.load(p)
        m = np.isfinite(g)
        report(s, np.where(m, g, 0.0), m)
    # Train
    tr = r"F:\paper\1\phase\Train\gt15_rad.npy"
    if os.path.exists(tr):
        g = np.load(tr)
        m = np.isfinite(g)
        report("Train", np.where(m, g, 0.0), m)
