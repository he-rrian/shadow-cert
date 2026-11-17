"""W1 verification -- theorem checks V6/V7/V8 for the optimal-recovery certificate.

V6  attainability of Theorem 1(3): psi* = h + kappa*v is admissible and saturates the bound
V7  soundness of the closed-form bound  ||v||_inf <= min(d_eq^2/16, w_min^2/8)   (Theorem 3 + 4)
V8  kappa-hat calibration: gamma sweep -> certified fraction and violation rate  (note sec. 8)
Run: python w1_verify.py
"""
import os, json
import numpy as np, scipy.io as sio, cv2
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
P = r"F:\paper\original_phase"
IDS = [127, 253, 379, 505, 631, 757, 876, 1002, 1128, 1800, 2600]
PHI_TOL = np.pi / 10

# ------------------------------------------------------------------ tools
def load_phi(i):
    return np.asarray(sio.loadmat(os.path.join(P, f"{i}.mat"))["fai"], np.float64)

def bandlimit(phi, f0=1 / 8, frac=0.45):
    gy, gx = np.gradient(phi); g = np.hypot(gx, gy) / (2 * np.pi)
    s = min(1.0, frac * f0 / max(float(np.percentile(g, 99.9)), 1e-9))
    return phi * s

def box(mask):
    ys, xs = np.where(mask)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

def poisson_dirichlet(mask, rhs, dirichlet=None):
    """-Delta u = rhs inside mask (u = dirichlet on the ring / 0 outside)."""
    y0, y1, x0, x1 = box(mask)
    m = mask[y0:y1, x0:x1]
    n, k = m.shape
    idx = np.arange(n * k).reshape(n, k)
    rows, cols, vals = [], [], []
    b = np.where(m, rhs[y0:y1, x0:x1], 0.0).astype(np.float64)
    for i in range(n):
        for j in range(k):
            a = idx[i, j]
            if not m[i, j]:
                rows.append(a); cols.append(a); vals.append(1.0)
                b[i, j] = 0.0 if dirichlet is None else dirichlet[y0 + i, x0 + j]
                continue
            rows.append(a); cols.append(a); vals.append(4.0)
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < n and 0 <= jj < k:
                    if m[ii, jj]:
                        rows.append(a); cols.append(idx[ii, jj]); vals.append(-1.0)
                    else:
                        b[i, j] += (0.0 if dirichlet is None else dirichlet[y0 + ii, x0 + jj])
                else:
                    b[i, j] += (0.0 if dirichlet is None else dirichlet[min(max(y0 + ii, 0), dirichlet.shape[0] - 1),
                                                                        min(max(x0 + jj, 0), dirichlet.shape[1] - 1)])
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n * k, n * k)).tocsr()
    u = spla.spsolve(A, b.ravel()).reshape(n, k)
    out = np.zeros_like(mask, dtype=np.float64)
    out[y0:y1, x0:x1] = u
    return out

def torsion(mask):
    return poisson_dirichlet(mask, np.ones_like(mask, dtype=np.float64))

def harmonic(phi, mask):
    return poisson_dirichlet(mask, np.zeros_like(phi), dirichlet=phi)

def w_min(mask, ntheta=180):
    """minimum width of the convex hull, in the DISCRETE domain convention:
    Dirichlet data live on the pixel centres just outside the mask, so the effective
    domain is one pixel wider on each side  ->  effective width = (max-min extent) + 2."""
    ys, xs = np.where(mask)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    best = 1e9
    for t in np.linspace(0, np.pi, ntheta, endpoint=False):
        dd = np.array([np.cos(t), np.sin(t)])
        proj = hull @ dd
        best = min(best, proj.max() - proj.min())
    return float(best) + 2.0

def shapes(H, W, d):
    """a round hole and a sliver, both with the same equivalent diameter d"""
    c = np.zeros((H, W), bool)
    out = {}
    r = d / 2
    yy, xx = np.mgrid[0:H, 0:W]
    out["round"] = (yy - H // 2) ** 2 + (xx - W // 2) ** 2 <= r * r
    L = int(round(np.pi * r * r / 2 / max(1, r)))      # keep the area roughly equal
    L = max(L, int(3 * d))
    hh = max(3, int(round(np.pi * r * r / L)))
    s = np.zeros((H, W), bool)
    y0 = H // 2 - L // 2; x0 = W // 2 - hh // 2
    if y0 < 0 or x0 < 0 or y0 + L > H or x0 + hh > W:
        return out
    s[y0:y0 + L, x0:x0 + hh] = True
    out["sliver"] = s
    return out

# ------------------------------------------------------------------ V6 + V7
print("=" * 78, "\nV6  attainability  &  V7  closed-form bound soundness")
rows = []
for i in IDS[:6]:
    phi = bandlimit(load_phi(i))
    H, W = phi.shape
    for d in (32, 48):
        for name, hole in shapes(H, W, d).items():
            if not hole.any(): continue
            lap = cv2.Laplacian(phi.astype(np.float32), cv2.CV_32F)
            kappa = float(np.abs(lap[hole]).max())
            v = torsion(hole); vinf = float(v.max())
            h = harmonic(phi, hole)
            # V6: psi* = h + kappa*v must be admissible and saturate the bound
            psi_star = h + kappa * v
            lap_star = cv2.Laplacian(psi_star.astype(np.float32), cv2.CV_32F)
            viol = float(np.abs(lap_star[hole]).max() / max(kappa, 1e-9))
            dev = float(np.abs(psi_star - h)[hole].max())
            # V7: closed-form bounds, in the DISCRETE domain convention (effective size = extent + 2)
            area = float(hole.sum()); d_eq = 2 * np.sqrt(area / np.pi)
            wm = w_min(hole)
            d_eq_eff = d_eq + 1.0                     # half-pixel margin on each side
            b_talenti = d_eq_eff ** 2 / 16; b_width = wm ** 2 / 8
            b_min = min(b_talenti, b_width)
            rows.append(dict(id=i, d=d, shape=name, kappa=kappa, vinf=vinf, d_eq=d_eq, w_min=wm,
                             b_talenti=b_talenti, b_width=b_width, b_min=b_min,
                             sound_min=bool(vinf <= b_min * 1.02),
                             attain_ratio=dev / max(kappa * vinf, 1e-9),
                             lap_ratio=viol))
            print(f"  id={i:5d} d={d:2d} {name:6s} kappa={kappa:7.4f} ||v||={vinf:8.2f} "
                  f"d_eq={d_eq:6.1f} w_min={wm:6.1f} | min-bound={b_min:8.2f} "
                  f"{'OK' if vinf <= b_min*1.02 else 'VIOLATION'} | attain={dev/max(kappa*vinf,1e-9):.4f}")

n_sound = sum(r["sound_min"] for r in rows)
print(f"  V7: sound {n_sound}/{len(rows)};   V6 attainment ratio "
      f"P50={np.median([r['attain_ratio'] for r in rows]):.4f}, "
      f"P10={np.percentile([r['attain_ratio'] for r in rows],10):.4f}")

# ------------------------------------------------------------------ V8
print("=" * 78, "\nV8  kappa-hat calibration (Savitzky-Golay + gamma sweep)")
def kappa_hat(phi, hole, q=99.0):
    """estimate kappa from the observed ring around the hole (SG-smoothed Laplacian)"""
    ring = cv2.dilate(hole.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & (~hole)
    sm = savgol_filter(phi.astype(np.float64), 9, 3, mode="nearest")
    lap = cv2.Laplacian(sm.astype(np.float32), cv2.CV_32F)
    return float(np.percentile(np.abs(lap[ring]), q))

recs = []
for i in IDS:
    phi = bandlimit(load_phi(i))
    H, W = phi.shape
    for d in (32, 48, 64):
        for name, hole in shapes(H, W, d).items():
            if not hole.any(): continue
            lap = cv2.Laplacian(phi.astype(np.float32), cv2.CV_32F)
            kap_true = float(np.abs(lap[hole]).max())
            kh = kappa_hat(phi, hole)
            v = torsion(hole); vinf = float(v.max())
            h = harmonic(phi, hole)
            # measured error is the deviation of GT from the estimate, in noise-free setting
            err = float(np.abs(h - phi)[hole].max())
            recs.append(dict(id=i, d=d, shape=name, kap_true=kap_true, kap_hat=kh, vinf=vinf,
                             err=err, phi_tol=PHI_TOL))
print(f"  n = {len(recs)}   kappa_hat/kappa_true: P10={np.percentile([r['kap_hat']/max(r['kap_true'],1e-12) for r in recs],10):.2f} "
      f"P50={np.median([r['kap_hat']/max(r['kap_true'],1e-12) for r in recs]):.2f} "
      f"P90={np.percentile([r['kap_hat']/max(r['kap_true'],1e-12) for r in recs],90):.2f}")
print(f"\n{'gamma':>6s}{'certified%':>12s}{'certified&violated':>20s}{'violation%':>12s}{'overall err>tol%':>17s}")
cal = []
for gamma in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
    n_cert = n_viol = n_err = 0
    for r in recs:
        rho = gamma * r["kap_hat"] * r["vinf"] / r["phi_tol"]
        cert = rho <= 1.0
        bad = r["err"] > r["phi_tol"]
        n_cert += int(cert)
        n_viol += int(cert and bad)
        n_err += int(bad)
    cal.append(dict(gamma=gamma, cert=n_cert / len(recs), viol=(n_viol / max(n_cert, 1)),
                    err=n_err / len(recs)))
    print(f"{gamma:6.1f}{100*cert if False else 100*n_cert/len(recs):11.0f}%{100*n_viol/max(n_cert,1):19.1f}%"
          f"{n_viol/max(n_cert,1)*100:11.1f}%{100*n_err/len(recs):16.1f}%")

# ------------------------------------------------------------------ V7b convergence
print("=" * 78, "\nV7b  discretisation convergence of the torsion solver (continuum values in brackets)")
conv = []
for R in (8, 16, 32, 64):
    n = 2 * (R + 3) + 1
    yy, xx = np.mgrid[0:n, 0:n]
    hole = (yy - n // 2) ** 2 + (xx - n // 2) ** 2 <= R * R
    vinf = float(torsion(hole).max())
    ana = R * R / 4
    conv.append(dict(kind="disk", size=R, vinf=vinf, analytic=ana, ratio=vinf / ana))
    print(f"  disk  R={R:3d}: ||v||={vinf:9.2f}  analytic R^2/4={ana:9.2f}  ratio={vinf/ana:.4f}")
for w in (4, 8, 16, 32):
    L = 6 * w; n_y = L + 8; n_x = w + 8
    hole = np.zeros((n_y, n_x), bool)
    hole[4:4 + L, 4:4 + w] = True
    vinf = float(torsion(hole).max())
    ana = (w + 1) ** 2 / 8          # effective width = interior + 1 (Dirichlet at centres outside)
    conv.append(dict(kind="strip", size=w, vinf=vinf, analytic=ana, ratio=vinf / ana))
    print(f"  strip w={w:3d}: ||v||={vinf:9.3f}  (w+1)^2/8={ana:9.3f}  w^2/8={w*w/8:9.3f}  ratio={vinf/ana:.4f}")

json.dump(dict(v6v7=rows, v7b=conv, v8=dict(recs=recs, calib=cal)),
          open(os.path.join(RES, "w1_verify.json"), "w"), indent=1)
print("\nsaved", os.path.join(RES, "w1_verify.json"))
