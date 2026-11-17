"""D3-D4 (v3) -- resolve the extrinsic convention with an independent FOV constraint.

Four conventions are tested:
    A   X_p = R X_c + t
    A2  X_p = R (X_c - t)          (t = projector centre in camera coords)
    B   X_c = R X_p + t            ->  X_p = R^T (X_c - t)
    B2  X_c = R (X_p - t)          ->  X_p = R^T X_c + R^T t   (unused, redundant with A)

Independent check that does NOT depend on flatness: every reconstructed point, projected into
the projector, must land inside the projector image  u_p in [0,912), v_p in [0,569).
A wrong convention scatters the points out of the image or clips them at the border.
Run: python d34_v3.py
"""
import os, json, itertools
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location("d34", os.path.join(HERE, "d34_reconstruct.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

W_P, H_P = 912.0, 569.0
sid = "00000"
d = np.load(os.path.join(m.GT, f"{sid}.npz"))
phi = d["phi64"].astype(np.float64); mask = d["m_gt"]
Kc, Kp, R, t = m.load_cal()

def project_to_projector(P_c, Re, te):
    """P_c : N x 3 points in camera coords ; X_p = Re X_c + te  (all conventions reduce to this)"""
    P_p = P_c @ Re.T + te
    uvp = P_p @ Kp.T
    with np.errstate(divide="ignore", invalid="ignore"):
        u = uvp[:, 0] / uvp[:, 2]; v = uvp[:, 1] / uvp[:, 2]
    return u, v, P_p[:, 2]

def recon(phi, mask, Re, te, axis, sign, u0, pproj=15.0):
    """delegate to the (fixed) implementation in d34_reconstruct.py"""
    return m.reconstruct(phi, mask, Kc, Kp, Re, te, axis, sign, u0)

CONV = {
    "A":  (R, t),
    "A2": (R, -R @ t),
    "B":  (R.T, -R.T @ t),
    "B2": (R.T, R.T @ t),
}

rows = []
for cname, (Re, te) in CONV.items():
    for axis, sign, u0 in itertools.product(("u", "v"), (1.0, -1.0), (0.0, 228.0, 456.0, 684.0)):
        z, coord = recon(phi, mask, Re, te, axis, sign, u0)
        H, W = phi.shape
        vv, uu = np.mgrid[0:H, 0:W]
        Pc = np.stack([(uu - Kc[0,2])/Kc[0,0], (vv - Kc[1,2])/Kc[1,1], np.ones_like(uu,float)],-1) * np.nan_to_num(z)[...,None]
        good = mask & np.isfinite(z) & (z > 0)
        if good.sum() < 0.3 * mask.sum():
            continue
        P = Pc[good]
        u, v, zp = project_to_projector(P, Re, te)
        inside = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < W_P) & (v >= 0) & (v < H_P) & (zp > 0)
        frac_in = float(inside.mean())
        sc = m.local_plane_pct(z, good, win=31, pct=20.0)
        rows.append(dict(conv=cname, axis=axis, sign=sign, u0=u0, frac_inside=frac_in,
                         score=float(sc) if np.isfinite(sc) else 1e9,
                         z50=float(np.median(z[good])),
                         cover=float(good.sum() / mask.sum())))
rows.sort(key=lambda r: (-r["frac_inside"], r["score"]))
print(f"{'conv':>4s}{'axis':>5s}{'sign':>5s}{'u_p0':>7s}{'in-FOV%':>9s}{'score mm':>10s}{'Z50':>8s}{'cover':>7s}")
for r in rows[:12]:
    print(f"{r['conv']:>4s}{r['axis']:>5s}{r['sign']:>+5.0f}{r['u0']:7.0f}{r['frac_inside']*100:8.1f}%"
          f"{r['score']:10.3f}{r['z50']:8.1f}{r['cover']*100:6.1f}%")
json.dump(rows, open(os.path.join(m.RES, "d34_v3.json"), "w"), indent=1)
