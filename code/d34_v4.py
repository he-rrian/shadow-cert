"""D3-D4 (v4) -- discriminate conventions by the LARGEST PLANAR PATCH.

The median/P5 local-plane residual saturates at 0 for any smooth field, so it cannot tell a
correct reconstruction from a smoothly distorted one.  A mechanical part, however, has FLAT
faces: for the correct (conv, axis, sign, u_p0) those faces are large connected regions whose
single-plane fit residual is < 0.1 mm.  A wrong u_p0 bends them and no large patch survives.

Score = area (px) of the largest connected component with local-plane residual < thresh.
Run: python d34_v4.py
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

Kc, Kp, R, t = m.load_cal()
W_P, H_P = 912.0, 569.0
THR = 0.10          # mm

def local_res(z, mask, win=25):
    zz = np.where(mask, z, 0.0).astype(np.float32)
    H, W = zz.shape
    xs = np.arange(W, dtype=np.float32)[None, :].repeat(H, 0)
    ys = np.arange(H, dtype=np.float32)[:, None].repeat(W, 1)
    mm = mask.astype(np.float32)
    bf = lambda a: cv2.boxFilter(a, -1, (win, win), normalize=True, borderType=cv2.BORDER_REPLICATE)
    n = np.maximum(bf(mm), 1e-6)
    mx, my = bf(xs * mm) / n, bf(ys * mm) / n
    mz = bf(zz * mm) / n
    cxx = bf(xs * xs * mm) / n - mx * mx
    cxy = bf(xs * ys * mm) / n - mx * my
    cyy = bf(ys * ys * mm) / n - my * my
    cxz = bf(xs * zz * mm) / n - mx * mz
    cyz = bf(ys * zz * mm) / n - my * mz
    czz = bf(zz * zz * mm) / n - mz * mz
    det = cxx * cyy - cxy * cxy
    ok = np.abs(det) > 1e-6
    a = np.where(ok, (cxz * cyy - cyz * cxy) / np.where(ok, det, 1), 0)
    b = np.where(ok, (cyz * cxx - cxz * cxy) / np.where(ok, det, 1), 0)
    res = np.sqrt(np.maximum(czz - a * cxz - b * cyz, 0))
    return np.where(ok, res, 1e9)

def largest_flat(z, mask):
    res = local_res(z, mask)
    flat = (mask & (res < THR)).astype(np.uint8)
    flat = cv2.morphologyEx(flat, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(flat, 8)
    area = int(stats[1:, 4].max()) if n > 1 else 0
    return area, float(np.median(res[mask & (res < 1e8)])) if (mask & (res < 1e8)).any() else 1e9

CONV = {"A": (R, t), "A2": (R, -R @ t), "B": (R.T, -R.T @ t), "B2": (R.T, R.T @ t)}
sid = "00000"
d = np.load(os.path.join(m.GT, f"{sid}.npz"))
phi = d["phi64"].astype(np.float64); mask = d["m_gt"]

rows = []
for cname, (Re, te) in CONV.items():
    for axis, sign in itertools.product(("u",), (1.0, -1.0)):
        for u0 in (0.0, 228.0, 456.0, 684.0):
            z, coord = m.reconstruct(phi, mask, Kc, Kp, Re, te, axis, sign, u0)
            good = mask & np.isfinite(z) & (z > 0)
            if good.sum() < 0.5 * mask.sum():
                continue
            area, mr = largest_flat(z, good)
            rows.append(dict(conv=cname, axis=axis, sign=sign, u0=u0, flat_area=area, med_res=mr,
                             z50=float(np.median(z[good])), cover=float(good.sum() / mask.sum())))
rows.sort(key=lambda r: -r["flat_area"])
print(f"{'conv':>4s}{'axis':>5s}{'sign':>5s}{'u0':>7s}{'largest flat (px)':>19s}{'median res mm':>15s}{'Z50':>8s}{'cover':>7s}")
for r in rows[:10]:
    print(f"{r['conv']:>4s}{r['axis']:>5s}{r['sign']:>+5.0f}{r['u0']:7.0f}{r['flat_area']:19d}"
          f"{r['med_res']:15.4f}{r['z50']:8.1f}{r['cover']*100:6.1f}%")
json.dump(rows, open(os.path.join(m.RES, "d34_v4.json"), "w"), indent=1)

if rows:
    best = rows[0]
    Re, te = CONV[best["conv"]]
    z, coord = m.reconstruct(phi, mask, Kc, Kp, Re, te, "u", best["sign"], best["u0"])
    good = mask & np.isfinite(z) & (z > 0)
    zz = np.where(good, z, np.nan)
    lo, hi = np.nanpercentile(zz, [2, 98])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    im = ax[0].imshow(zz, cmap="turbo", vmin=lo, vmax=hi)
    plt.colorbar(im, ax=ax[0], fraction=0.046, label="Z (mm)")
    ax[0].set_title(f"best: {best['conv']} u sign{best['sign']:+.0f} u0={best['u0']:.0f}  Z50={best['z50']:.0f} mm")
    res = local_res(z, good)
    ax[1].imshow(np.where(good, np.minimum(res, 0.5), np.nan), cmap="magma", vmin=0, vmax=0.5)
    ax[1].set_title(f"local plane residual (mm)  largest flat patch = {best['flat_area']} px")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.savefig(os.path.join(m.RES, "d34_v4.png"), dpi=130, bbox_inches="tight")
    print("saved", os.path.join(m.RES, "d34_v4.png"))
