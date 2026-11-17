"""W2/D1 fix -- lock the theoretical step and ratios, and cure the hidden inter-component
2*pi branch error in psi_1.

Dataset truth (SFNet, arXiv:2402.00977; confirmed here by two independent checks):
    projector 912 px wide, wavelengths lam = [912, 304, 114, 15] projector px
    ratios 1 : 3 : 8/3 : 7.6   (NOT 1:3:8:64 -- 912/64 = 14.25 was rounded to 15)
    cross-check: camera period 610 px for lam=912 => magnification 1.495 ;
                 camera period  10 px for lam= 15 => magnification 1.500   (consistent)

Why 8->64 failed: if a connected component sits on the other side of the +-pi branch of
psi_1, Phi_1 has an error 2*pi*m.  That error is INVISIBLE at 1->3 (3*2pi*m = 6pi*m = 0 mod 2pi)
and at 3->8 ((8/3)*6pi*m = 16pi*m = 0 mod 2pi), but at 8->64 it becomes
7.6*16pi*m mod 2pi = -0.4*pi*m  ->  1.26 rad (m=1), 2.51 rad (m=2), matching the measured P50.

Cure: unwrap psi_1 with a PLANE GUIDE so that every component shares one global branch.
Run: python w2_d1_fix.py
"""
import os, json
import numpy as np, cv2
from skimage.restoration import unwrap_phase

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "probes", "results"); os.makedirs(RES, exist_ok=True)
BASE = r"F:\paper\SynthFringe"
SUBS = [("1_12", 12), ("3_16", 16), ("8_6", 6), ("64_15", 15)]
LAM = {"1_12": 912.0, "3_16": 304.0, "8_6": 114.0, "64_15": 15.0}
RATIO = {("1_12", "3_16"): LAM["1_12"] / LAM["3_16"],
         ("3_16", "8_6"): LAM["3_16"] / LAM["8_6"],
         ("8_6", "64_15"): LAM["8_6"] / LAM["64_15"]}

def im(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

def seq(sub, sid):
    p = os.path.join(BASE, sub, sid)
    fs = sorted(x for x in os.listdir(p) if x.endswith(".bmp"))
    out = []
    for f in fs:
        a = im(os.path.join(p, f))
        if a.ndim == 3:
            a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2GRAY)
        out.append(a.astype(np.float64))
    return np.stack(out)

def phase_theory(I):
    """LOCKED step: delta_k = 2*pi*k/N"""
    N = I.shape[0]
    d = 2 * np.pi * np.arange(N) / N
    s = np.tensordot(np.sin(d), I, axes=(0, 0))
    c = np.tensordot(np.cos(d), I, axes=(0, 0))
    return np.arctan2(s, c), np.hypot(s, c) / N

def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def poly_guided_unwrap(psi, roi, deg=3, iters=8):
    """Phi = psi + 2*pi*round((P - psi)/2pi), with P a robust polynomial fit to psi.
    All connected components get the SAME global branch (no per-component constants).
    A polynomial of degree `deg` is used because a deep object's phase deviates from a
    plane by more than pi, which makes a plane guide fail for multi-component scenes."""
    H, W = psi.shape
    yy, xx = np.mgrid[0:H, 0:W]
    xs = (xx - W / 2) / (W / 2)
    ys = (yy - H / 2) / (H / 2)
    basis = []
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            basis.append((xs ** j) * (ys ** i))
    B = np.stack([b[roi] for b in basis], 1)          # n x m
    r = np.zeros_like(psi)
    P = np.zeros_like(psi)
    for _ in range(iters):
        resid = wrap(psi - P)
        coef, *_ = np.linalg.lstsq(B, resid[roi], rcond=None)
        P = P + sum(c * b for c, b in zip(coef, basis))
    return P + wrap(psi - P)

def plane_guided_unwrap(psi, roi):
    """kept for reference (degree-1 guide)"""
    H, W = psi.shape
    yy, xx = np.mgrid[0:H, 0:W]
    xs = (xx - W / 2) / (W / 2); ys = (yy - H / 2) / (H / 2)
    basis = [np.ones_like(psi, dtype=float), xs.astype(float), ys.astype(float)]
    B = np.stack([b[roi] for b in basis], 1)
    P = np.zeros_like(psi)
    for _ in range(6):
        resid = wrap(psi - P)
        coef, *_ = np.linalg.lstsq(B, resid[roi], rcond=None)
        P = P + sum(c * b for c, b in zip(coef, basis))
    return P + wrap(psi - P)

def run(sid, use_theory=True, per_component=False):
    P, MX = {}, None
    for sub, N in SUBS:
        I = seq(sub, sid)
        P[sub], amp = phase_theory(I)
        MX = I.max(0) if MX is None else np.maximum(MX, I.max(0))
    roi = MX >= 25
    out = dict(sid=sid, roi=float(roi.mean()))

    phi = {"1_12": poly_guided_unwrap(P["1_12"], roi)}
    for lo, hi in (("1_12", "3_16"), ("3_16", "8_6"), ("8_6", "64_15")):
        a = RATIO[(lo, hi)]
        d = a * phi[lo] - P[hi]
        if per_component:
            off = np.zeros_like(d)
            m8 = roi.astype(np.uint8)
            n, lab = cv2.connectedComponents(m8, 8)
            for k in range(1, n):
                mm = (lab == k) & roi
                if mm.sum() < 50:
                    continue
                ok = float(np.angle(np.mean(np.exp(1j * wrap(d)[mm]))))
                off[mm] = ok
        else:
            off = float(np.angle(np.mean(np.exp(1j * wrap(d)[roi]))))
        k = np.round((d - off) / (2 * np.pi))
        phi[hi] = 2 * np.pi * k + P[hi] + off
        e = np.abs(wrap(d - off)[roi])
        out[f"{lo}->{hi}"] = dict(a=a, p50=float(np.median(e)), p99=float(np.percentile(e, 99)),
                                  mx=float(e.max()), frac=float((e > np.pi / 2).mean()))
    p = phi["64_15"]
    dx = np.abs(np.diff(p, axis=1))[roi[:, :-1] & roi[:, 1:]]
    dy = np.abs(np.diff(p, axis=0))[roi[:-1, :] & roi[1:, :]]
    out["tear"] = int((dx > np.pi).sum() + (dy > np.pi).sum())
    out["n_comp"] = int(cv2.connectedComponents(roi.astype(np.uint8), 8)[0] - 1)
    out["_phi"] = p
    out["_roi"] = roi
    return out

if __name__ == "__main__":
    ids = ["00000", "01200", "02399", "00360", "00600", "00900", "01500", "01800", "02400", "03000"]
    rows = []
    print(f"{'scene':7s}{'comp':>5s}{'roi%':>6s} | {'1->3 P99':>9s}{'f>pi/2':>8s} | {'3->8 P99':>9s}{'f>pi/2':>8s} "
          f"| {'8->64 P99':>10s}{'f>pi/2':>8s}{'tear':>6s}")
    for sid in ids:
        try:
            r = run(sid)
        except Exception as ex:
            print(f"{sid}: ERR {type(ex).__name__} {ex}"); continue
        a1, a2, a3 = r["1_12->3_16"], r["3_16->8_6"], r["8_6->64_15"]
        rows.append({k: v for k, v in r.items() if not k.startswith("_")})
        print(f"{sid:7s}{r['n_comp']:5d}{r['roi']*100:6.1f} | {a1['p99']:9.4f}{a1['frac']*100:7.2f}% "
              f"| {a2['p99']:9.4f}{a2['frac']*100:7.2f}% | {a3['p99']:10.4f}{a3['frac']*100:7.2f}%{r['tear']:6d}")
    json.dump(rows, open(os.path.join(RES, "w2_d1_fix.json"), "w"), indent=1)
    if rows:
        print("\ntear total = %d px over %d scenes" % (sum(r["tear"] for r in rows), len(rows)))

