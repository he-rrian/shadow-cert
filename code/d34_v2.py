"""D3-D4 (v2) -- restrict u_p0 to the physically admissible discrete set.

Reasoning: Phi_1 is the *pattern* phase (the polynomial-guide unwrap does not add an arbitrary
real constant; it only fixes the 2*pi branch), hence
        u_p = 912 * Phi_1 / (2*pi) + u_p0 ,   u_p0 in {0, 228, 456, 684}
(228 = a quarter of 912, i.e. the four cos/sin phase conventions of the rendered pattern).
Axis must be u_p (the fringes vary along the camera x axis); axis = v_p is kept only as a control.

Also scans P_proj in {14.25, 15.0} to expose the 5.26 % effect the calibration note warns about.
Run: python d34_v2.py
"""
import os, json, itertools
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location("d34", os.path.join(HERE, "d34_reconstruct.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

W_P = m.W_P
sid = "00000"
d = np.load(os.path.join(m.GT, f"{sid}.npz"))
phi = d["phi64"].astype(np.float64); mask = d["m_gt"]
Kc, Kp, R, t = m.load_cal()

def recon(phi, mask, Re, te, axis, sign, u0, pproj):
    H, W = phi.shape
    vv, uu = np.mgrid[0:H, 0:W]
    dc = np.stack([(uu - Kc[0, 2]) / Kc[0, 0], (vv - Kc[1, 2]) / Kc[1, 1], np.ones_like(uu, float)], -1)
    coord = sign * pproj * phi / (2 * np.pi) + u0
    if axis == "u":
        a = (coord - Kp[0, 2])[..., None] * Re[2, :] - Kp[0, 0] * Re[0, :]
        dd = (coord - Kp[0, 2]) * te[2] - Kp[0, 0] * te[0]
    else:
        a = (coord - Kp[1, 2])[..., None] * Re[2, :] - Kp[1, 1] * Re[1, :]
        dd = (coord - Kp[1, 2]) * te[2] - Kp[1, 1] * te[1]
    nd = a[..., 0] * dc[..., 0] + a[..., 1] * dc[..., 1] + a[..., 2] * dc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = dd / nd
    return np.where(np.isfinite(z), z, np.nan), coord

rows = []
combos = list(itertools.product(("A", "B"), ("u", "v"), (1.0, -1.0), (0.0, 228.0, 456.0, 684.0), (15.0, 14.25)))
for conv, axis, sign, u0, pp in combos:
    Re, te = (R, t) if conv == "A" else (R.T, -R.T @ t)
    z, coord = recon(phi, mask, Re, te, axis, sign, u0, pp)
    good = mask & np.isfinite(z) & (z > 0) & (coord >= 0) & (coord < W_P)
    if good.sum() < 0.6 * mask.sum():
        continue
    sc = m.local_plane_pct(z, good, win=31, pct=20.0)
    if not np.isfinite(sc):
        continue
    rows.append(dict(conv=conv, axis=axis, sign=sign, u0=u0, pproj=pp, score=float(sc),
                     z50=float(np.median(z[good])), z5=float(np.percentile(z[good], 5)),
                     z95=float(np.percentile(z[good], 95)), cover=float(good.sum() / mask.sum())))
rows.sort(key=lambda r: r["score"])
print(f"{'conv':>4s}{'axis':>5s}{'sign':>5s}{'u_p0':>7s}{'P_proj':>8s}{'score mm':>10s}{'Z50':>8s}{'Z5':>8s}{'Z95':>8s}{'cover':>7s}")
for r in rows[:10]:
    print(f"{r['conv']:>4s}{r['axis']:>5s}{r['sign']:>+5.0f}{r['u0']:7.0f}{r['pproj']:8.2f}"
          f"{r['score']:10.4f}{r['z50']:8.1f}{r['z5']:8.1f}{r['z95']:8.1f}{r['cover']*100:6.1f}%")
json.dump(rows, open(os.path.join(m.RES, "d34_v2.json"), "w"), indent=1)

# visualise the best 5
fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
for ax, r in zip(axes.ravel(), rows[:5]):
    Re, te = (R, t) if r["conv"] == "A" else (R.T, -R.T @ t)
    z, coord = recon(phi, mask, Re, te, r["axis"], r["sign"], r["u0"], r["pproj"])
    good = mask & np.isfinite(z) & (z > 0)
    zz = np.where(good, z, np.nan)
    lo, hi = np.nanpercentile(zz, [2, 98])
    ax.imshow(zz, cmap="turbo", vmin=lo, vmax=hi)
    ax.set_title(f"{r['conv']} {r['axis']} sign{r['sign']:+.0f} u0={r['u0']:.0f} P={r['pproj']:.2f}\n"
                 f"score {r['score']:.3f} mm  Z {lo:.0f}-{hi:.0f}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
axes.ravel()[-1].axis("off")
fig.suptitle(f"SynthFringe {sid}: discrete-offset candidates", fontsize=11)
fig.savefig(os.path.join(m.RES, "d34_v2.png"), dpi=130, bbox_inches="tight")
print("saved", os.path.join(m.RES, "d34_v2.png"))
