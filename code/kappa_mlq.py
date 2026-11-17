"""Masked local quadratic kappa  (kappa_MLQ).

Instead of "fill outside the mask then convolve with a fixed SG7 kernel" (which blows up at
irregular mask borders: P95 = 11-28 rad/px^2), fit a quadratic surface
        z = a + b x + c y + d x^2 + e xy + f y^2
to the VALID pixels inside a 7x7 window and read the Laplacian at the centre directly:
        Laplacian z(0,0) = 2d + 2f
Pixels with fewer than `min_pts` valid neighbours are skipped.

Run: python kappa_mlq.py
"""
import os
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "probes", "results")
GT = os.path.join(RES, "gt")
REAL = r"F:\paper\2\model\real\captures\proc"

def kappa_mlq(z, mask, win=7, min_pts=12):
    """returns (laplacian map, valid map)"""
    H, W = z.shape
    r = win // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    x = xx.ravel().astype(np.float64); y = yy.ravel().astype(np.float64)
    basis = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], 0)   # 6 x P
    m = mask.astype(np.float64)
    zz = np.where(mask, z, 0.0).astype(np.float64)
    Bk = [b.reshape(win, win) for b in basis]          # 6 kernels, each 7x7
    def corr(field, ker):
        # sum_p ker[p] * field(x+p)
        return cv2.filter2D(field.astype(np.float32), -1, ker[::-1].copy(),
                            borderType=cv2.BORDER_CONSTANT)
    G = np.empty((H, W, 6, 6), np.float32)
    for k in range(6):
        for l in range(6):
            G[..., k, l] = corr(m, (Bk[k] * Bk[l]))
    b = np.empty((H, W, 6), np.float32)
    for k in range(6):
        b[..., k] = corr(m * zz, Bk[k])
    npts = corr(m, np.ones((win, win)))
    sel = (mask & (npts >= min_pts)).ravel()
    lap = np.full(H * W, np.nan, np.float32)
    if sel.sum() > 0:
        Gs = G.reshape(-1, 6, 6)[sel].astype(np.float64)
        bs = b.reshape(-1, 6)[sel].astype(np.float64)
        # ridge on the diagonal for conditioning
        Gs = Gs + np.eye(6)[None] * 1e-3
        sol = np.linalg.solve(Gs, bs[..., None])[..., 0]
        lap[sel] = (2 * sol[:, 3] + 2 * sol[:, 5]).astype(np.float32)
    lap = lap.reshape(H, W)
    ok = mask & np.isfinite(lap)
    return lap, ok

def report(name, phi, mask):
    lap, ok = kappa_mlq(phi, mask)
    v = np.abs(lap[ok])
    if v.size < 50:
        print(f"{name:10s} too few valid px"); return None
    print(f"{name:10s} valid={ok.sum()/mask.sum()*100:5.1f}%  "
          f"kappa_MLQ P50={np.median(v):.4f} P95={np.percentile(v,95):.4f}")
    return float(np.median(v)), float(np.percentile(v, 95))

if __name__ == "__main__":
    print("kappa from a MASKED LOCAL QUADRATIC fit (7x7, >=12 pts)\n")
    for sid in ("00000", "00360", "01500"):
        p = os.path.join(GT, f"{sid}.npz")
        if os.path.exists(p):
            d = np.load(p)
            report(f"SF {sid}", d["phi64"].astype(np.float64), d["m_gt"])
    for s in sorted(os.listdir(REAL)):
        p = os.path.join(REAL, s, "gt_phase.npy")
        if os.path.exists(p):
            g = np.load(p); m = np.isfinite(g)
            report(s, np.where(m, g, 0.0), m)
    tr = r"F:\paper\1\phase\Train\gt15_rad.npy"
    if os.path.exists(tr):
        g = np.load(tr); m = np.isfinite(g)
        report("Train", np.where(m, g, 0.0), m)
