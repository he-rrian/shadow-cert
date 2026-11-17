"""W2 pre-flight 2 -- why the cascade failed: masked unwrapping gives each connected
component an arbitrary 2*pi multiple, so a per-component offset pollutes every diagnostic.
The base (1-period) phase must be unwrapped on the FULL image (it is a near plane wave,
so the full-image unwrap is trivially reliable) and the frequency ratios must be MEASURED,
not taken from the folder labels (measured 1 : 3.015 : 8.03 : 61.24, not 1:3:8:64).

Run: python w2_preflight_global.py
"""
import os
import numpy as np, cv2
from skimage.restoration import unwrap_phase

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
    amp = np.hypot(np.tensordot(np.sin(d0), I, axes=(0, 0)),
                   np.tensordot(np.cos(d0), I, axes=(0, 0))) / N
    good = np.argsort(amp.ravel())[-400:]
    Y = I.reshape(N, -1)[:, good]
    best = None
    for ddeg in np.arange(lo, hi, step):
        dd = np.deg2rad(ddeg) * np.arange(N)
        M = np.vstack([np.ones(N), np.cos(dd), np.sin(dd)]).T
        c, *_ = np.linalg.lstsq(M, Y, rcond=None)
        sse = float(((Y - M @ c) ** 2).sum())
        if best is None or sse < best[1]:
            best = (ddeg, sse)
    return best[0]

def phase_from_step(I, ddeg):
    N = I.shape[0]
    dd = np.deg2rad(ddeg) * np.arange(N)
    M = np.vstack([np.ones(N), np.cos(dd), np.sin(dd)]).T
    coef = np.tensordot(np.linalg.pinv(M), I, axes=(1, 0))
    return np.arctan2(-coef[2], coef[1]), np.hypot(coef[1], coef[2])

def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

if __name__ == "__main__":
    sid = "00000"
    P, MX = {}, None
    for sub, N in SUBS:
        I = seq(sub, sid)
        phi, amp = phase_from_step(I, fit_step(I))
        P[sub] = phi
        MX = I.max(0) if MX is None else np.maximum(MX, I.max(0))
    roi = MX >= 25

    # global unwrap of the 1-period phase (full image, no mask)
    U1 = unwrap_phase(P["1_12"])
    print(f"psi_1 full-image unwrap: range [{U1.min():.2f}, {U1.max():.2f}] rad over the frame")
    g = {s: float(np.median(np.abs(np.gradient(P[s], axis=1))[roi])) for s in P}
    ratios = {"3_16": g["3_16"] / g["1_12"], "8_6": g["8_6"] / g["1_12"], "64_15": g["64_15"] / g["1_12"]}
    print("measured ratios vs 1_12:", {k: round(v, 4) for k, v in ratios.items()}, " nominal 3 / 8 / 64")

    # cascade with measured ratios, all from the GLOBAL base
    cur, cur_ratio = U1, 1.0
    for sub in ("3_16", "8_6", "64_15"):
        a = ratios[sub] / cur_ratio
        d = wrap(a * cur - P[sub])
        off = float(np.angle(np.mean(np.exp(1j * d[roi]))))
        e = np.abs(wrap(d - off)[roi])
        k = np.round((d - off) / (2 * np.pi))
        cur = 2 * np.pi * k + P[sub] + off
        cur_ratio = ratios[sub]
        print(f"  step ->{sub:6s} a={a:8.4f} offset={off:+.3f}  residual P50={np.median(e):.4f} "
              f"P99={np.percentile(e,99):.4f}  frac>pi/2={np.mean(e>np.pi/2):.3f}")

    phi64 = cur
    dx = np.abs(np.diff(phi64, axis=1))[roi[:, :-1] & roi[:, 1:]]
    dy = np.abs(np.diff(phi64, axis=0))[roi[:-1, :] & roi[1:, :]]
    print(f"\nspatial tearing |dPhi|>pi : {int((dx>np.pi).sum()+(dy>np.pi).sum())} px "
          f"(grad P99={np.percentile(np.concatenate([dx,dy]),99):.3f} rad/px)")
