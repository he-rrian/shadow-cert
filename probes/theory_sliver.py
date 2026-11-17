"""P2 theory check 3 -- the certificate scale is the WIDTH, not the area.

Exact optimal-recovery bound for an arbitrary dead zone Omega:
        max |psi - h| <= kappa * ||v_Omega||_inf ,      -Delta v = 1,  v|dOmega = 0
For a sliver of width w:  ||v||inf -> w^2/8, independent of its length.
For the equal-area disk:  ||v||inf = d_eq^2/16.
So an elongated (shadow-like) dead zone is certified far more often than its
equivalent diameter suggests -- verified here on real phase maps.
Run: python theory_sliver.py
"""
import os, json
import numpy as np, scipy.io as sio, cv2
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
P = r"F:\paper\original_phase"
IDS = [127, 253, 379, 505, 631, 757, 876, 1002, 1128, 1800, 2600]
PHI_TOL = np.pi / 10

def load_phi(i):
    return np.asarray(sio.loadmat(os.path.join(P, f"{i}.mat"))["fai"], np.float64)

def bandlimit(phi, f0=1 / 8, frac=0.45):
    gy, gx = np.gradient(phi); g = np.hypot(gx, gy) / (2 * np.pi)
    s = min(1.0, frac * f0 / max(float(np.percentile(g, 99.9)), 1e-9))
    return phi * s

def harmonic_dirichlet(phi, hole, iters=4000):
    z = phi.copy(); z[hole] = np.median(phi[~hole]); m = hole.astype(np.float32)
    k = np.array([[0, .25, 0], [.25, 0, .25], [0, .25, 0]], np.float32)
    for _ in range(iters):
        sm = cv2.filter2D(z.astype(np.float32), -1, k, borderType=cv2.BORDER_REPLICATE)
        z = z * (1 - m) + sm * m
    return z

def torsion_max(mask):
    """||v||inf with -Delta v = 1, v = 0 outside mask (sparse solve on the mask bbox)"""
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = mask[y0:y1, x0:x1]
    n, k = m.shape[0], m.shape[1]
    N = n * k
    idx = np.arange(N).reshape(n, k)
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(k):
            a = idx[i, j]
            if not m[i, j]:
                rows.append(a); cols.append(a); vals.append(1.0); continue
            rows.append(a); cols.append(a); vals.append(4.0)
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                b = idx[ii, jj] if (0 <= ii < n and 0 <= jj < k) else -1
                if b >= 0:
                    rows.append(a); cols.append(b); vals.append(-1.0)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    b = np.where(m, 1.0, 0.0).ravel()
    v = spla.spsolve(A, b)
    return float(v.max())

shapes = {
    "round_d28": (28, 28),
    "round_d48": (48, 48),
    "ellipse_2:1_A28x56": (28, 56),
    "sliver_10x200": (10, 200),
    "sliver_14x200": (14, 200),
    "sliver_20x200": (20, 200),
    "sliver_24x120": (24, 120),
}

rows = []
print(f"{'shape':20s}{'area':>7s}{'d_eq':>7s}{'w_eff':>7s}{'Vinf':>9s}{'bound_tors':>11s}{'bound_deq':>10s}"
      f"{'meas_max':>10s}{'cert_tors':>10s}{'cert_deq':>9s}")
for name, (h, w) in shapes.items():
    acc = []
    for i in IDS:
        phi = bandlimit(load_phi(i))
        H, W = phi.shape
        hole = np.zeros((H, W), bool)
        y0 = (H - h) // 2; x0 = (W - w) // 2
        hole[y0:y0 + h, x0:x0 + w] = True
        lap = cv2.Laplacian(phi.astype(np.float32), cv2.CV_32F)
        kappa = float(np.abs(lap[hole]).max())
        Vin = torsion_max(hole)
        h_ext = harmonic_dirichlet(phi, hole, iters=3000)
        meas = float(np.abs(h_ext - phi)[hole].max())
        area = float(hole.sum()); d_eq = 2 * np.sqrt(area / np.pi)
        bt = kappa * Vin; bd = kappa * d_eq ** 2 / 16
        acc.append((area, d_eq, np.sqrt(8 * Vin), Vin, bt, bd, meas,
                    float(bt <= PHI_TOL), float(bd <= PHI_TOL)))
    M = np.median(np.array(acc), 0)
    rows.append(dict(shape=name, area=M[0], d_eq=M[1], w_eff=M[2], Vinf=M[3], bound_torsion=M[4],
                     bound_deq=M[5], measured_max=M[6], cert_torsion=M[7], cert_deq=M[8]))
    print(f"{name:20s}{M[0]:7.0f}{M[1]:7.1f}{M[2]:7.1f}{M[3]:9.1f}{M[4]:11.3f}{M[5]:10.3f}"
          f"{M[6]:10.3f}{M[7]*100:9.0f}%{M[8]*100:8.0f}%")

print("\n--- soundness (measured_max <= bound_torsion ?) ---")
bad = [r for r in rows if r["measured_max"] > r["bound_torsion"] * 1.02]
print(f"  violations: {len(bad)} / {len(rows)}")
print("--- tightness of the torsion bound vs the equivalent-diameter bound ---")
for r in rows:
    print(f"  {r['shape']:20s} bound_torsion={r['bound_torsion']:7.3f}  bound_deq={r['bound_deq']:8.3f}"
          f"  gain={r['bound_deq']/max(r['bound_torsion'],1e-9):5.1f}x  measured={r['measured_max']:6.3f}")
json.dump(rows, open(os.path.join(RES, "theory_sliver.json"), "w"), indent=1)
print("\nsaved", os.path.join(RES, "theory_sliver.json"))
