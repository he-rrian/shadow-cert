"""W2/D1 quick validation of the corrected cascade on a couple of scenes.

Corrections vs the first attempt:
  (1) the order k must come from the UNWRAPPED difference a*Phi_low - psi_high
      (wrapping it first forces k=0 and destroys the cascade)
  (2) the base phase needs no spatial unwrap when psi_1 spans < 2*pi (it is already continuous),
      and masked unwrapping must be avoided (per-component 2*pi offsets)
  (3) the frequency ratios must be MEASURED, not taken from the folder labels
Run: python w2_check_cascade.py
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

def fit_step(I, lo=10.0, hi=90.0, st=0.02):
    N = I.shape[0]
    d0 = 2 * np.pi * np.arange(N) / N
    amp = np.hypot(np.tensordot(np.sin(d0), I, axes=(0, 0)),
                   np.tensordot(np.cos(d0), I, axes=(0, 0))) / N
    good = np.argsort(amp.ravel())[-400:]
    Y = I.reshape(N, -1)[:, good]
    best = None
    for dg in np.arange(lo, hi, st):
        dd = np.deg2rad(dg) * np.arange(N)
        M = np.vstack([np.ones(N), np.cos(dd), np.sin(dd)]).T
        c, *_ = np.linalg.lstsq(M, Y, rcond=None)
        sse = float(((Y - M @ c) ** 2).sum())
        if best is None or sse < best[1]:
            best = (dg, sse)
    return best[0]

def phase_lsq(I, dg):
    N = I.shape[0]
    dd = np.deg2rad(dg) * np.arange(N)
    M = np.vstack([np.ones(N), np.cos(dd), np.sin(dd)]).T
    co = np.tensordot(np.linalg.pinv(M), I, axes=(1, 0))
    return np.arctan2(-co[2], co[1]), np.hypot(co[1], co[2])

def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def est_ratio(phi_low, psi_high, roi, guess):
    """concentration search + slope refinement"""
    g = np.gradient(phi_low, axis=1)[roi]
    g2 = np.gradient(psi_high, axis=1)[roi]
    # coarse: circular concentration over +-8% of the guess
    best = None
    for a in np.linspace(guess * 0.92, guess * 1.08, 1201):
        c = np.abs(np.mean(np.exp(1j * wrap(a * phi_low - psi_high)[roi])))
        if best is None or c > best[0]:
            best = (c, a)
    a = best[1]
    # refine: unwrap the residual and regress on phi_low
    for _ in range(3):
        d = wrap(a * phi_low - psi_high)
        dd = d.copy(); dd[~roi] = 0
        Du = unwrap_phase(dd)
        m = roi
        A = np.vstack([phi_low[m], np.ones(m.sum())]).T
        slope = np.linalg.lstsq(A, Du[m], rcond=None)[0][0]
        a = a - slope
        if abs(slope) < 1e-4:
            break
    c = np.abs(np.mean(np.exp(1j * wrap(a * phi_low - psi_high)[roi])))
    return float(a), float(c)

def run(sid):
    P, MX, steps = {}, None, {}
    for sub, N in SUBS:
        I = seq(sub, sid)
        dg = fit_step(I)
        steps[sub] = dg
        P[sub], _ = phase_lsq(I, dg)
        MX = I.max(0) if MX is None else np.maximum(MX, I.max(0))
    roi = MX >= 25
    # base: no unwrap needed if continuous
    span = np.ptp(P["1_12"][roi])
    phi = {"1_12": P["1_12"].copy()}
    base_mode = "raw" if span < 2 * np.pi else "unwrap"
    if base_mode == "unwrap":
        phi["1_12"] = unwrap_phase(P["1_12"])
    out = dict(sid=sid, span1=float(span), base=base_mode)
    for lo, hi in (("1_12", "3_16"), ("3_16", "8_6"), ("8_6", "64_15")):
        nom = {("1_12", "3_16"): 3.0, ("3_16", "8_6"): 8.0 / 3.0, ("8_6", "64_15"): 8.0}[(lo, hi)]
        a, conc = est_ratio(phi[lo], P[hi], roi, nom)
        d_raw = a * phi[lo] - P[hi]
        off = float(np.angle(np.mean(np.exp(1j * wrap(d_raw)[roi]))))
        k = np.round((d_raw - off) / (2 * np.pi))
        phi[hi] = 2 * np.pi * k + P[hi] + off
        e = np.abs(wrap(d_raw - off)[roi])
        out[f"{lo}->{hi}"] = dict(a=a, conc=conc, off=off, p50=float(np.median(e)),
                                  p99=float(np.percentile(e, 99)),
                                  frac=float((e > np.pi / 2).mean()))
    p64 = phi["64_15"]
    dx = np.abs(np.diff(p64, axis=1))[roi[:, :-1] & roi[:, 1:]]
    dy = np.abs(np.diff(p64, axis=0))[roi[:-1, :] & roi[1:, :]]
    out["tear"] = int((dx > np.pi).sum() + (dy > np.pi).sum())
    out["grad_p99"] = float(np.percentile(np.concatenate([dx, dy]), 99))
    return out

if __name__ == "__main__":
    for sid in ("00000", "01200", "02399"):
        try:
            r = run(sid)
        except Exception as ex:
            print(sid, "ERR", ex); continue
        print(f"=== {sid}  span(psi1)={r['span1']:.2f} base={r['base']} ===")
        for k_ in ("1_12->3_16", "3_16->8_6", "8_6->64_15"):
            v = r[k_]
            print(f"  {k_:12s} a={v['a']:9.4f} conc={v['conc']:.3f} off={v['off']:+.3f} "
                  f"P50={v['p50']:.4f} P99={v['p99']:.4f} frac>pi/2={v['frac']:.4f}")
        print(f"  tearing={r['tear']} px   grad P99={r['grad_p99']:.3f} rad/px")
