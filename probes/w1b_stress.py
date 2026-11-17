"""W1b stress tests -- the three counter-examples the external review demands.

T1  curved shadow (banana): does the convex-hull width bound w_min^2/8 blow up?
    compare  ||v||_inf  vs  d_eq^2/16 (Talenti)  vs  w_min^2/8 (strip)  vs  r_ins^2/2
T2  noisy boundary data: perturbed Lemma 1, ||h_ghat - h_g||_inf <= delta_g ?
T3  through-cut topology: a shadow that severs the observed region -> unknown 2*pi*k offset
Run: python w1b_stress.py
"""
import os, json
import numpy as np, scipy.io as sio, cv2
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
P = r"F:\paper\original_phase"

def load_phi(i):
    return np.asarray(sio.loadmat(os.path.join(P, f"{i}.mat"))["fai"], np.float64)

def bandlimit(phi, f0=1 / 8, frac=0.45):
    gy, gx = np.gradient(phi); g = np.hypot(gx, gy) / (2 * np.pi)
    s = min(1.0, frac * f0 / max(float(np.percentile(g, 99.9)), 1e-9))
    return phi * s

def box(mask):
    ys, xs = np.where(mask)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

def solve_laplace(mask, rhs, dirichlet=None, pad=3):
    """-Delta u = rhs inside mask, u = dirichlet on the ring (0 if None).
    The bbox is padded so that a Dirichlet ring always exists inside it."""
    H, W = mask.shape
    y0, y1, x0, x1 = box(mask)
    y0 = max(0, y0 - pad); y1 = min(H, y1 + pad)
    x0 = max(0, x0 - pad); x1 = min(W, x1 + pad)
    m = mask[y0:y1, x0:x1]
    if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
        raise RuntimeError("mask touches the padded bbox border -- keep shapes away from the image edge")
    n, k = m.shape
    N = n * k
    e = np.ones(N)
    # NOTE: the flattening stride is k (columns), not n
    A = sp.diags([-e[:-1], -e[:-k], 4 * e, -e[k:], -e[1:]], [-1, -k, 0, k, 1],
                 shape=(N, N), format="lil")
    # kill the row wrap-around introduced by the +/-1 diagonals
    for i in range(n):
        if i > 0:
            A[i * k, i * k - 1] = 0.0
        if i * k + k < N:
            A[i * k + (k - 1), i * k + k] = 0.0
    b = np.where(m, rhs[y0:y1, x0:x1], 0.0).astype(np.float64).ravel().copy()
    D = np.zeros((n, k)) if dirichlet is None else dirichlet[y0:y1, x0:x1]
    for i in range(n):
        for j in range(k):
            a = i * k + j
            if not m[i, j]:
                A.rows[a] = [a]; A.data[a] = [1.0]
                b[a] = D[i, j]
    # interior rows keep their 4-point stencil; exterior unknowns are pinned to D above,
    # so no extra Dirichlet term may be added to b (that would double-count).
    u = spla.spsolve(A.tocsr(), b).reshape(n, k)
    out = np.zeros_like(mask, dtype=np.float64)
    out[y0:y1, x0:x1] = u
    return out

def torsion(mask):
    return solve_laplace(mask, np.ones_like(mask, dtype=np.float64))

def r_ins(mask):
    return float(cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5).max())

def w_min(mask, ntheta=360):
    ys, xs = np.where(mask)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    best = 1e9
    for t in np.linspace(0, np.pi, ntheta, endpoint=False):
        dd = np.array([np.cos(t), np.sin(t)])
        pr = hull @ dd
        best = min(best, pr.max() - pr.min())
    return float(best) + 2.0

# =================================================================== T1
print("=" * 92, "\nT1  curved shadow: which closed-form bound survives?")
H = W = 256
yy, xx = np.mgrid[0:H, 0:W]
shapes = {}

r = 30; shapes["disk_d60"] = (yy - 128) ** 2 + (xx - 128) ** 2 <= r * r
sq = np.zeros((H, W), bool); sq[98:158, 98:158] = True; shapes["square_60"] = sq
el = ((yy - 128) / 20.0) ** 2 + ((xx - 128) / 60.0) ** 2 <= 1; shapes["ellipse_40x120"] = el
sl = np.zeros((H, W), bool); sl[116:126, 20:236] = True; shapes["sliver_10x216"] = sl
# banana: arc of radius 100, span +-60 deg, thickness 10
d_arc = np.abs(np.hypot(yy - 128, xx - 128) - 100.0)
ang = np.arctan2(yy - 128, xx - 128)
ban = (d_arc <= 5.0) & (np.abs(np.cos(ang)) >= np.cos(1.0472))   # +-60 deg around x-axis
shapes["banana_arc"] = ban
# L-shape and cross
L = np.zeros((H, W), bool); L[60:200, 60:70] = True; L[190:200, 60:200] = True
shapes["L_shape"] = L
C = np.zeros((H, W), bool); C[40:216, 122:134] = True; C[122:134, 40:216] = True
shapes["cross"] = C

rows = []
print(f"{'shape':18s}{'area':>7s}{'||v||':>9s}{'d_eq^2/16':>10s}{'wmin^2/8':>10s}{'r_ins^2/2':>10s}"
      f"{'  sound: deq / wmin / rins(+1)'}")
# extra random blobs to test the r_ins conjecture
rngb = np.random.default_rng(7)
for t in range(8):
    m = np.zeros((H, W), bool)
    cy, cx = rngb.integers(90, 166, 2)
    for _ in range(rngb.integers(3, 6)):
        ry, rx = rngb.integers(6, 45, 2)
        yy0, xx0 = rngb.integers(60, 196, 2)
        m |= (((yy - yy0) / max(ry, 3)) ** 2 + ((xx - xx0) / max(rx, 3)) ** 2) <= 1
    if 200 < m.sum() < 12000:
        shapes[f"blob{t}"] = m
for name, m in shapes.items():
    v = torsion(m); vinf = float(v.max())
    area = float(m.sum()); d_eq = 2 * np.sqrt(area / np.pi)
    b_deq = (d_eq + 1) ** 2 / 16
    b_wmin = w_min(m) ** 2 / 8
    r_in = r_ins(m)
    b_rins = (r_in + 1.0) ** 2 / 2              # discrete convention: +1
    ok = [vinf <= b_deq * 1.01, vinf <= b_wmin * 1.01, vinf <= b_rins * 1.01]
    rows.append(dict(shape=name, area=area, vinf=vinf, b_deq=b_deq, b_wmin=b_wmin, r_ins=r_in,
                     b_rins=b_rins, ok_deq=bool(ok[0]), ok_wmin=bool(ok[1]), ok_rins=bool(ok[2]),
                     ratio_wmin=b_wmin / max(vinf, 1e-9), ratio_rins=b_rins / max(vinf, 1e-9)))
    print(f"{name:18s}{area:7.0f}{vinf:9.2f}{b_deq:10.2f}{b_wmin:10.1f}{b_rins:10.2f}"
          f"   {'Y' if ok[0] else 'N'} / {'Y' if ok[1] else 'N'} / {'Y' if ok[2] else 'N'}")
nw = sum(1 for r in rows if not r["ok_wmin"]); nr = sum(1 for r in rows if not r["ok_rins"])
print(f"\n  sound: d_eq {len(rows)-sum(1 for r in rows if not r['ok_deq'])}/{len(rows)}"
      f"   w_min {len(rows)-nw}/{len(rows)}   r_ins(+1) {len(rows)-nr}/{len(rows)}")
print("  looseness on curved shapes  w_min^2/8 / ||v||_inf : "
      + ", ".join(f"{r['shape']}={r['ratio_wmin']:.0f}x" for r in rows if r['ratio_wmin'] > 50))

# =================================================================== T2
print("=" * 92, "\nT2  noisy boundary: perturbed Lemma 1")
phi = bandlimit(load_phi(505))
hole = np.zeros((H, W), bool); hole[98:158, 98:158] = True
h_clean = solve_laplace(hole, np.zeros_like(phi), dirichlet=phi)
lap = cv2.Laplacian(phi.astype(np.float32), cv2.CV_32F)
kappa = float(np.abs(lap[hole]).max())
vinf = float(torsion(hole).max())
rng = np.random.default_rng(0)
print(f"  kappa={kappa:.4f}  ||v||={vinf:.2f}  model bound kappa*||v||={kappa*vinf:.3f} rad")
print(f"{'delta_g':>9s}{'||h_ghat-h_g||_inf':>20s}{'ratio':>8s}{'total bound':>13s}{'certified@pi/10':>17s}")
t2 = []
for dg in (0.02, 0.05, 0.1, 0.2, 0.5):
    errs = []
    for trial in range(5):
        g = phi + rng.uniform(-dg, dg, phi.shape) * (np.abs(cv2.dilate(hole.astype(np.uint8),
            np.ones((3, 3), np.uint8)).astype(bool) & ~hole)).astype(float)
        h_n = solve_laplace(hole, np.zeros_like(phi), dirichlet=g)
        errs.append(float(np.abs(h_n - h_clean)[hole].max()))
    e = float(np.max(errs))
    tot = kappa * vinf + e
    t2.append(dict(delta_g=dg, err_max=e, ratio=e / dg, total=tot, cert=bool(tot <= np.pi / 10)))
    print(f"{dg:9.3f}{e:20.3f}{e/dg:8.2f}{tot:13.3f}{'YES' if tot <= np.pi/10 else 'NO':>17s}")

# =================================================================== T3
print("=" * 92, "\nT3  through-cut shadow: unknown 2*pi*k offset between the two sides")
ph0 = bandlimit(load_phi(757))
PAD = 24
ph = np.pad(ph0, PAD, mode="edge")                 # so the band does not touch the image border
Hp, Wp = ph.shape
yyp, xxp = np.mgrid[0:Hp, 0:Wp]
band = np.zeros((Hp, Wp), bool); band[122 + PAD:134 + PAD, PAD:PAD + 256] = True
top = (yyp < 122 + PAD); bot = (yyp >= 134 + PAD)
band &= (xxp >= PAD) & (xxp < PAD + 256)
print(f"  band covers {band.mean()*100:.1f}% of the padded canvas; the observed region is split "
      f"into an upper and a lower component (no path around the cut inside the object)")
for koff in (0, 1, 2, 3):
    g = ph.copy()
    g[bot] += 2 * np.pi * koff            # simulate independent unwrapping of the lower part
    h = solve_laplace(band, np.zeros_like(ph), dirichlet=g)
    err = float(np.abs(h - ph)[band].max())
    vinf3 = float(torsion(band).max())
    kap3 = float(np.abs(cv2.Laplacian(ph.astype(np.float32), cv2.CV_32F)[band]).max())
    bound = kap3 * vinf3
    print(f"  k={koff}: measured max err = {err:8.3f} rad | model bound = {bound:6.3f} rad | "
          f"{'bound<=tol (would certify)' if bound <= np.pi/10 else 'bound>tol'} "
          f"| {'VIOLATION' if err > bound else 'ok'}")
print("  -> a 2*pi*k offset on one boundary component is invisible to the model bound:")
print("     the certificate is CONDITIONAL on a globally consistent boundary (Omega_roi\\Omega connected).")

json.dump(dict(t1=rows, t2=t2), open(os.path.join(RES, "w1b_stress.json"), "w"), indent=1)
print("\nsaved", os.path.join(RES, "w1b_stress.json"))

