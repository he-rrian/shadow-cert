"""W2/D1 -- cascade with a SPATIALLY VARYING ratio field.

Finding: the local ratio dpsi64/dpsi8 ranges 7.11-7.80 (+-5%) across the field, so no single
constant a can satisfy  Phi_64 = a*Phi_8  pointwise; a constant a leaves a drift up to ~1.4 rad
and the order decision fails.  Fix: estimate a smooth ratio FIELD a(x,y) from block-wise
gradient medians, use it in the order decision, then check the residual.

Run: python w2_cascade_ratiofield.py [sid ...]
"""
import sys
import numpy as np, cv2, os
from skimage.restoration import unwrap_phase

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "w2_check_cascade.py"),
           encoding="utf-8").read().split("if __name__")[0]
exec(src)


def ratio_field(low, high, roi, nb=16, smooth=9):
    """smooth field r(x,y) = dpsi_high/dpsi_low, from block-wise gradient medians"""
    H, W = roi.shape
    gL = np.gradient(low, axis=1)
    gH = np.gradient(high, axis=1)
    ys, xs = np.where(roi)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    grid = np.full((nb, nb), np.nan)
    for i in range(nb):
        for j in range(nb):
            a0 = y0 + (y1 - y0) * i // nb; a1 = y0 + (y1 - y0) * (i + 1) // nb
            b0 = x0 + (x1 - x0) * j // nb; b1 = x0 + (x1 - x0) * (j + 1) // nb
            m = roi[a0:a1, b0:b1]
            if m.sum() < 60:
                continue
            gl = np.median(gL[a0:a1, b0:b1][m]); gh = np.median(gH[a0:a1, b0:b1][m])
            if abs(gl) > 1e-6:
                grid[i, j] = gh / gl
    # fill nan with the global median, then upsample + smooth
    med = np.nanmedian(grid)
    grid = np.where(np.isnan(grid), med, grid)
    F = cv2.resize(grid, (W, H), interpolation=cv2.INTER_CUBIC)
    F = cv2.GaussianBlur(F, (0, 0), smooth)
    return F, med, float(np.nanstd(grid))


def run(sid):
    P, MX = {}, None
    for sub, N in SUBS:
        I = seq(sub, sid)
        dg = fit_step(I)
        P[sub], _ = phase_lsq(I, dg)
        MX = I.max(0) if MX is None else np.maximum(MX, I.max(0))
    roi = MX >= 25
    out = dict(sid=sid, roi=float(roi.mean()))

    phi = {"1_12": P["1_12"].copy()}
    if np.ptp(P["1_12"][roi]) >= 2 * np.pi:
        phi["1_12"] = unwrap_phase(P["1_12"])
    for lo, hi, nom in (("1_12", "3_16", 3.0), ("3_16", "8_6", 8.0 / 3.0)):
        a, conc = est_ratio(phi[lo], P[hi], roi, nom)
        d = a * phi[lo] - P[hi]
        off = float(np.angle(np.mean(np.exp(1j * wrap(d)[roi]))))
        phi[hi] = 2 * np.pi * np.round((d - off) / (2 * np.pi)) + P[hi] + off
        e = np.abs(wrap(d - off)[roi])
        out[f"{lo}->{hi}"] = dict(a=a, conc=conc, p50=float(np.median(e)),
                                  p99=float(np.percentile(e, 99)), frac=float((e > np.pi / 2).mean()))

    # ---- last step with a ratio FIELD
    F, med, sd = ratio_field(phi["8_6"], P["64_15"], roi)
    d = F * phi["8_6"] - P["64_15"]
    off = float(np.angle(np.mean(np.exp(1j * wrap(d)[roi]))))
    phi["64_15"] = 2 * np.pi * np.round((d - off) / (2 * np.pi)) + P["64_15"] + off
    e = np.abs(wrap(d - off)[roi])
    out["8_6->64_15"] = dict(a_med=med, a_sd=sd, p50=float(np.median(e)),
                             p99=float(np.percentile(e, 99)), frac=float((e > np.pi / 2).mean()))
    p = phi["64_15"]
    dx = np.abs(np.diff(p, axis=1))[roi[:, :-1] & roi[:, 1:]]
    dy = np.abs(np.diff(p, axis=0))[roi[:-1, :] & roi[1:, :]]
    out["tear"] = int((dx > np.pi).sum() + (dy > np.pi).sum())
    return out


if __name__ == "__main__":
    sids = sys.argv[1:] or ["00000", "01200", "02399", "03599"]
    for sid in sids:
        try:
            r = run(sid)
        except Exception as ex:
            print(f"{sid}: ERR {type(ex).__name__} {ex}"); continue
        print(f"=== {sid}  roi={r['roi']*100:.1f}% ===")
        for k in ("1_12->3_16", "3_16->8_6"):
            v = r[k]
            print(f"  {k:12s} a={v['a']:8.4f} conc={v['conc']:.3f} P50={v['p50']:.4f} "
                  f"P99={v['p99']:.4f} frac>pi/2={v['frac']:.4f}")
        v = r["8_6->64_15"]
        print(f"  8_6->64_15   a_field med={v['a_med']:.3f} sd={v['a_sd']:.3f} P50={v['p50']:.4f} "
              f"P99={v['p99']:.4f} frac>pi/2={v['frac']:.4f}   tearing={r['tear']} px")
