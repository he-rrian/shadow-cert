"""P2 theory check -- is the optimal-recovery certificate sound?

Lemma (paper Thm 1): for a dead zone Omega with |Delta psi| <= kappa and Dirichlet boundary data,
     |psi(x) - h(x)| <= (kappa/4)(R^2 - |x-xc|^2) <= kappa*d^2/16
with h the harmonic extension of the true boundary values.

We verify it numerically on the actual phase maps: pick a disk, compute the discrete harmonic
extension from the true boundary, take kappa = sup|Delta phi| inside the disk, and check
     max|h - phi| <= kappa*d^2/16 .
Also reports how conservative the bound is (ratio bound/measured), which drives how much of the
domain the certificate can cover.
Run: python theory_check.py
"""
import os, json
import numpy as np, scipy.io as sio, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
P = r"F:\paper\original_phase"
IDS = [1, 127, 253, 379, 505, 631, 757, 876, 1002, 1128, 1800, 2600]

def load_phi(i):
    return np.asarray(sio.loadmat(os.path.join(P, f"{i}.mat"))["fai"], np.float64)

def bandlimit(phi, f0=1 / 8, frac=0.45):
    gy, gx = np.gradient(phi); g = np.hypot(gx, gy) / (2 * np.pi)
    s = min(1.0, frac * f0 / max(float(np.percentile(g, 99.9)), 1e-9))
    return phi * s

def harmonic_dirichlet(phi, hole, iters=6000):
    """solve Laplace inside `hole` with Dirichlet data = phi on the ring just outside it"""
    z = phi.copy()
    z[hole] = np.median(phi[~hole])
    m = hole.astype(np.float32)
    k = np.array([[0, .25, 0], [.25, 0, .25], [0, .25, 0]], np.float32)
    for _ in range(iters):
        sm = cv2.filter2D(z.astype(np.float32), -1, k, borderType=cv2.BORDER_REPLICATE)
        z = z * (1 - m) + sm * m
    return z

def disk(shape, c, r):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (yy - c[0]) ** 2 + (xx - c[1]) ** 2 <= r * r

rows = []
print(f"{'id':>5s}{'d':>5s}{'kappa':>10s}{'bound':>10s}{'meas_max':>10s}{'ratio':>8s}  sound?")
for i in IDS:
    phi = bandlimit(load_phi(i))
    for d in (24, 32, 48, 64, 96):
        r = d / 2.0
        # keep the disk strictly inside the image
        cy, cx = 128, 128
        hole = disk(phi.shape, (cy, cx), r)
        if hole.sum() < 20: continue
        # curvature: sup|Delta phi| inside the disk (discrete 5-point Laplacian)
        lap = cv2.Laplacian(phi.astype(np.float32), cv2.CV_32F)
        kappa = float(np.abs(lap[hole]).max())
        h = harmonic_dirichlet(phi, hole)
        meas = float(np.abs(h - phi)[hole].max())
        bound = kappa * d * d / 16.0
        ratio = bound / max(meas, 1e-9)
        sound = meas <= bound * 1.02          # 2% slack for the discrete operator
        rows.append(dict(id=i, d=d, kappa=kappa, bound=bound, meas=meas, ratio=ratio, sound=bool(sound)))
        print(f"{i:5d}{d:5d}{kappa:10.3f}{bound:10.2f}{meas:10.2f}{ratio:8.2f}  "
              f"{'OK' if sound else 'VIOLATION'}")

R = np.array([[r["d"], r["kappa"], r["bound"], r["meas"], r["ratio"], float(r["sound"])] for r in rows])
print("\n--- summary ---")
print(f"  n = {len(R)},  violations = {int((R[:,5]==0).sum())}")
for d in (24, 32, 48, 64, 96):
    S = R[R[:,0] == d]
    if len(S):
        print(f"  d={d:3d}: bound/measured median = {np.median(S[:,4]):6.2f}  "
              f"(P10 {np.percentile(S[:,4],10):5.2f}, P90 {np.percentile(S[:,4],90):6.2f})")

# how much of a domain would a certificate cover at a given tolerance?
print("\n--- certificate coverage: fraction of pixels with rho = kappa*d^2/(16*phi_tol) <= 1 ---")
for phi_tol in (0.314, 0.157, 0.063):
    for d in (24, 48, 96):
        S = R[R[:,0] == d]
        rho = S[:,1] * d * d / (16 * phi_tol)
        print(f"  phi_tol={phi_tol:.3f} rad  d={d:3d}: certified {np.mean(rho <= 1)*100:5.1f}% "
              f"(rho P50={np.median(rho):5.1f})")
json.dump(rows, open(os.path.join(RES, "theory_check.json"), "w"), indent=1)
print("\nsaved", os.path.join(RES, "theory_check.json"))
