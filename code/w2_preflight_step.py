"""W2 pre-flight: the four SynthFringe subsets are NOT all equal-step 2*pi/N sequences.

The textbook N-step formula assumes delta_k = 2*pi*k/N.  Fitting the step per subset gives
    1_12  -> 30.04 deg  (2pi/12)   OK
    3_16  -> 22.42 deg  (2pi/16)   OK
    8_6   -> 39.98 deg  (2pi/9)    <-- NOT 2pi/6 !
    64_15 -> 24.10 deg  (2pi/15)   OK
so the naive N=6 formula silently returns a wrong phase for 8_6 and the cascade collapses.

Fix: fit the step, then recover the phase by 3-parameter linear least squares
     I_k = A + u cos(delta_k) - v sin(delta_k),   phi = atan2(-v, u)
which is valid for ANY known (even non-uniform) step sequence.

This script re-estimates the frequency ratios after the fix.
Run: python w2_preflight_step.py
"""
import os
import numpy as np, cv2

BASE = r"F:\paper\SynthFringe"
SUBS = [("1_12", 12), ("3_16", 16), ("8_6", 6), ("64_15", 15)]

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

def fit_step(I, lo=10.0, hi=90.0, step=0.02):
    N = I.shape[0]
    d0 = 2 * np.pi * np.arange(N) / N
    s = np.tensordot(np.sin(d0), I, axes=(0, 0)); c = np.tensordot(np.cos(d0), I, axes=(0, 0))
    amp = np.hypot(s, c) / N
    good = np.argsort(amp.ravel())[-400:]
    Y = I.reshape(N, -1)[:, good]
    best = None
    for ddeg in np.arange(lo, hi, step):
        dd = np.deg2rad(ddeg) * np.arange(N)
        M = np.vstack([np.ones(N), np.cos(dd), np.sin(dd)]).T
        coef, *_ = np.linalg.lstsq(M, Y, rcond=None)
        sse = float(((Y - M @ coef) ** 2).sum())
        if best is None or sse < best[1]:
            best = (ddeg, sse)
    return best[0]

def phase_from_step(I, ddeg):
    """3-parameter LSQ phase recovery for a known (possibly non-uniform) step"""
    N = I.shape[0]
    dd = np.deg2rad(ddeg) * np.arange(N)
    M = np.vstack([np.ones(N), np.cos(dd), np.sin(dd)]).T
    Minv = np.linalg.pinv(M)                       # 3 x N
    coef = np.tensordot(Minv, I, axes=(1, 0))       # 3 x H x W
    u, v = coef[1], coef[2]
    phi = np.arctan2(-v, u)
    amp = np.hypot(u, v)
    return phi, amp

if __name__ == "__main__":
    sid = "00000"
    P, MX = {}, None
    print(f"{'sub':6s}{'N':>3s}{'step(deg)':>11s}{'2pi/':>7s}{'ampP50':>8s}{'|grad|':>10s}")
    for sub, N in SUBS:
        I = seq(sub, sid)
        dg = fit_step(I)
        phi, amp = phase_from_step(I, dg)
        m = I.max(0) >= 25
        MX = I.max(0) if MX is None else np.maximum(MX, I.max(0))
        gx = np.abs(np.gradient(phi, axis=1))[m]
        P[sub] = phi
        print(f"{sub:6s}{N:3d}{dg:11.2f}{360/dg:7.2f}{np.median(amp[m]):8.1f}{np.median(gx):10.5f}")
    roi = MX >= 25
    g = {s: float(np.median(np.abs(np.gradient(P[s], axis=1))[roi])) for s in P}
    print("\nmeasured frequency ratios (|grad| ratio, sign-corrected):")
    for lo, hi in (("1_12", "3_16"), ("3_16", "8_6"), ("8_6", "64_15"), ("1_12", "64_15")):
        print(f"  {lo:6s}->{hi:6s}  {g[hi]/g[lo]:7.4f}")
    print("\nnominal: 3.0000 / 2.6667 / 8.0000 / 64.0000")
