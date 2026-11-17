"""D3-D4  --  SynthFringe 3D reconstruction: ray / projector-plane intersection.

Unknowns solved JOINTLY (grid search):
    extrinsic convention   X_p = R X_c + t      (A)
                           X_c = R X_p + t      (B)
    fringe axis            u_p   or  v_p
    phase sign             +1 / -1
    zero-phase offset      u_p0 in [0, 912)

Hard constants (SFNet, see plan sec 4.1):
    W_p = 912 projector px,  lam = [912, 304, 114, 15],  P_proj = 15 px/period

Selection criterion:  minimise the median high-pass residual of Z over M_gt
(a plane is reproduced exactly by a Gaussian, any wrong u_p0 "bends" the plane),
subject to  Z > 0  and  u_p in [0, W_p).

Run: python d34_reconstruct.py [--scene 00000]
"""
import os, json, argparse
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "probes", "results")
GT = os.path.join(RES, "gt")
CAL = r"F:\paper\SynthFringe\calibData"
W_P = 912.0
P_PROJ = 15.0          # projector px per period (NOT 912/64 = 14.25)

def load_cal():
    Kc = np.loadtxt(os.path.join(CAL, "CamIntrinsicMatrix.txt"))
    Kp = np.loadtxt(os.path.join(CAL, "ProjIntrinsicMatrix.txt"))
    R = np.loadtxt(os.path.join(CAL, "ProjRotationMatrix.txt"))
    t = np.loadtxt(os.path.join(CAL, "ProjTranslationVector.txt")).ravel()
    return Kc, Kp, R, t

def reconstruct(phi, mask, Kc, Kp, Re, te, axis, sign, u0):
    """returns Z (mm) and the projector coordinate"""
    H, W = phi.shape
    vv, uu = np.mgrid[0:H, 0:W]
    # camera ray
    dc = np.stack([(uu - Kc[0, 2]) / Kc[0, 0],
                   (vv - Kc[1, 2]) / Kc[1, 1],
                   np.ones_like(uu, float)], -1)             # H x W x 3
    coord = sign * P_PROJ * phi / (2 * np.pi) + u0           # projector column (or row)
    # projector plane for a fixed column u:   fx*x_p - (u-cx)*z_p = 0 ,  X_p = Re*X + te
    #   ->  A . X = D  with  A = fx*Re[0,:] - (u-cx)*Re[2,:] ,  D = (u-cx)*te[2] - fx*te[0]
    if axis == "u":
        A = Kp[0, 0] * Re[0, :] - (coord - Kp[0, 2])[..., None] * Re[2, :]
        Dd = (coord - Kp[0, 2]) * te[2] - Kp[0, 0] * te[0]
    else:
        A = Kp[1, 1] * Re[1, :] - (coord - Kp[1, 2])[..., None] * Re[2, :]
        Dd = (coord - Kp[1, 2]) * te[2] - Kp[1, 1] * te[1]
    nd = A[..., 0] * dc[..., 0] + A[..., 1] * dc[..., 1] + A[..., 2] * dc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = Dd / nd
    z = np.where(np.isfinite(z), z, np.nan)
    return z, coord

def flatness(z, mask):
    """median |Z - GaussianBlur(Z)| / IQR(Z).  A plane gives 0."""
    zz = np.where(mask, z, 0.0).astype(np.float32)
    sm = cv2.GaussianBlur(zz, (0, 0), 4.0)
    res = np.abs(zz - sm)[mask]
    iqr = np.subtract(*np.percentile(zz[mask], [75, 25]))
    return float(np.median(res) / max(iqr, 1e-9)), float(np.median(res))

def local_plane_pct(z, mask, win=15, pct=5.0):
    """percentile of the LOCAL PLANE-FIT residual (mm) over M_gt.
    A wrong u_p0 bends even a true plane, so the flattest patches stop being flat.
    Uses box-filtered moments -> O(1) per window."""
    k = (win, win)
    zz = np.where(mask, z, 0.0).astype(np.float32)
    H, W = zz.shape
    xs = np.arange(W, dtype=np.float32)[None, :].repeat(H, 0)
    ys = np.arange(H, dtype=np.float32)[:, None].repeat(W, 1)
    m = mask.astype(np.float32)
    bf = lambda a: cv2.boxFilter(a, -1, k, normalize=True, borderType=cv2.BORDER_REPLICATE)
    n = bf(m)
    mx, my = bf(xs * m) / n, bf(ys * m) / n
    mz = bf(zz * m) / n
    cxx = bf(xs * xs * m) / n - mx * mx
    cxy = bf(xs * ys * m) / n - mx * my
    cyy = bf(ys * ys * m) / n - my * my
    cxz = bf(xs * zz * m) / n - mx * mz
    cyz = bf(ys * zz * m) / n - my * mz
    czz = bf(zz * zz * m) / n - mz * mz
    det = cxx * cyy - cxy * cxy
    ok = (np.abs(det) > 1e-6) & (n > 0.95)
    a = np.where(ok, (cxz * cyy - cyz * cxy) / np.where(ok, det, 1), 0)
    b = np.where(ok, (cyz * cxx - cxz * cxy) / np.where(ok, det, 1), 0)
    res2 = np.maximum(czz - a * cxz - b * cyz, 0.0)
    rms = np.sqrt(res2)
    good = mask & ok
    if good.sum() < 200:
        return np.nan
    return float(np.percentile(rms[good], pct))

def best_for_scene(sid, coarse=6.0, fine=0.5):
    d = np.load(os.path.join(GT, f"{sid}.npz"))
    phi = d["phi64"].astype(np.float64)
    mask = d["m_gt"]
    Kc, Kp, R, t = load_cal()
    out = []
    for conv in ("A", "B"):
        if conv == "A":
            Re, te = R, t
        else:                                    # X_p = R^T (X_c - t)
            Re, te = R.T, -R.T @ t
        for axis in ("u", "v"):
            for sign in (1.0, -1.0):
                # coarse scan of u_p0
                best = None
                for u0 in np.arange(0.0, W_P, coarse):
                    z, coord = reconstruct(phi, mask, Kc, Kp, Re, te, axis, sign, u0)
                    good = mask & np.isfinite(z) & (z > 0) & (coord >= 0) & (coord < W_P)
                    if good.sum() < 0.5 * mask.sum():
                        continue
                    sc = local_plane_pct(z, good)
                    if not np.isfinite(sc):
                        continue
                    if best is None or sc < best[0]:
                        best = (sc, u0)
                if best is None:
                    continue
                # fine refine
                for u0 in np.arange(best[1] - coarse, best[1] + coarse, fine):
                    z, coord = reconstruct(phi, mask, Kc, Kp, Re, te, axis, sign, u0)
                    good = mask & np.isfinite(z) & (z > 0) & (coord >= 0) & (coord < W_P)
                    if good.sum() < 0.5 * mask.sum():
                        continue
                    sc = local_plane_pct(z, good)
                    if not np.isfinite(sc):
                        continue
                    if sc < best[0]:
                        best = (sc, u0, float(np.median(z[good])),
                                float(np.percentile(z[good], 5)), float(np.percentile(z[good], 95)),
                                float(good.sum() / max(mask.sum(), 1)))
                if len(best) == 2:
                    continue
                out.append(dict(conv=conv, axis=axis, sign=sign, u0=best[1],
                                score=best[0], z50=best[2],
                                z5=best[3], z95=best[4], cover=best[5]))
    out.sort(key=lambda r: r["score"])
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="00000")
    a = ap.parse_args()
    rows = best_for_scene(a.scene)
    print(f"=== scene {a.scene} : top candidates (sorted by local-plane P5 residual) ===")
    print(f"{'conv':>4s}{'axis':>5s}{'sign':>5s}{'u_p0':>9s}{'P5 mm':>9s}{'Z50':>8s}{'Z5':>8s}{'Z95':>8s}{'cover':>7s}")
    for r in rows[:8]:
        print(f"{r['conv']:>4s}{r['axis']:>5s}{r['sign']:>+5.0f}{r['u0']:9.2f}{r['score']:9.4f}"
              f"{r['z50']:8.1f}{r['z5']:8.1f}{r['z95']:8.1f}{r['cover']*100:6.1f}%")
    json.dump(rows, open(os.path.join(RES, "d34_scan.json"), "w"), indent=1)
