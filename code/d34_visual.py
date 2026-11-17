"""D3-D4 diagnostic: render Z maps for the candidate (conv, axis, sign, u_p0) combos."""
import os
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location("d34", os.path.join(HERE, "d34_reconstruct.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

sid = "00000"
d = np.load(os.path.join(m.GT, f"{sid}.npz"))
phi = d["phi64"].astype(np.float64); mask = d["m_gt"]
Kc, Kp, R, t = m.load_cal()

CANDS = [
    ("A", "u", +1, 590.0),
    ("A", "u", -1, 891.5),
    ("B", "u", +1, 28.0),
    ("B", "v", +1, 728.5),
    ("B", "v", -1, 884.0),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
for ax, (conv, axis, sign, u0) in zip(axes.ravel(), CANDS):
    Re, te = (R, t) if conv == "A" else (R.T, -R.T @ t)
    z, coord = m.reconstruct(phi, mask, Kc, Kp, Re, te, axis, sign, u0)
    good = mask & np.isfinite(z) & (z > 0)
    zz = np.where(good, z, np.nan)
    lo, hi = np.nanpercentile(zz, [2, 98])
    ax.imshow(zz, cmap="turbo", vmin=lo, vmax=hi)
    ax.set_title(f"conv {conv}  axis {axis}  sign {sign:+d}  u_p0={u0:.0f}\n"
                 f"Z P2/P50/P98 = {lo:.0f}/{np.nanmedian(zz):.0f}/{hi:.0f} mm  "
                 f"cover={good.sum()/mask.sum()*100:.0f}%", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
axes.ravel()[-1].axis("off")
fig.suptitle(f"SynthFringe {sid}: Z map for each extrinsic/axis/offset candidate", fontsize=11)
out = os.path.join(m.RES, "d34_candidates.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
