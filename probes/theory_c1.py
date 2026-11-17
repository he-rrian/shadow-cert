"""P2 theory check 2 -- how much does C^1 (value + normal derivative) tighten the certificate?

Class K = { psi : |Delta psi| <= kappa } on a disk of radius R.

  C^0 data : two admissible surfaces share the boundary VALUE.
             w = psi1 - psi2 :  w|d = 0, |Delta w| <= 2kappa
             => max |w(0)| = kappa R^2 / 2 = kappa d^2 / 8     (diameter)
             => one-sided certificate radius = kappa d^2 / 16

  C^1 data : they also share the normal DERIVATIVE.
             w|d = dn w|d = 0, |Delta w| <= 2kappa  (Gauss makes int Delta w = 0 automatic)
             radial extremal: Delta w = -2kappa on r < rho, +2kappa on r > rho,
             with rho = R/sqrt(2), hence
             max |w(0)| = kappa R^2 ln2 / 2 = kappa d^2 ln2 / 8   (diameter)
             => one-sided certificate radius = kappa d^2 ln2 / 16 = kappa d^2 / 23.08

  => tightening = 1/ln2 = 1.4427 (NOT 4x); certified DIAMETER grows by sqrt(1/ln2) = 1.201

Torsion-function form (exact for an ARBITRARY region Omega):
             max |psi - h| over K = kappa * ||v||_inf,   -Delta v = 1, v|dOmega = 0
   and by Talenti's comparison principle the disk maximises ||v||_inf among sets of equal
   area, so d_eq = 2*sqrt(A/pi) is a SOUND substitute: bound = kappa d_eq^2/16.
Run: python theory_c1.py
"""
import os, json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)

def radial_solution(R, f_of_r, nr=20001):
    """solve (r w')' = r f, enforcing w'(R)=0 and w(R)=0; returns (r, w, w'(R))"""
    r = np.linspace(1e-9, R, nr)
    f = f_of_r(r)
    # r w' = int_0^r s f ds   (trapezoid)
    integ = np.concatenate([[0.0], np.cumsum((r[1:] * f[1:] + r[:-1] * f[:-1]) / 2 * np.diff(r))])
    wp = integ / r
    w = np.concatenate([[0.0], np.cumsum((wp[1:] + wp[:-1]) / 2 * np.diff(r))]) + 0.0
    wp_R = wp[-1]
    w -= w[-1]                      # enforce w(R) = 0 by a constant shift
    return r, w, wp_R

def solve_dirichlet(n, rhs, mask):
    """(-Delta) u = rhs inside mask, u = 0 outside (Dirichlet on the region boundary)"""
    N = n * n
    e = np.ones(N)
    L = sp.diags([-e[:-1], -e[:-n], 4 * e, -e[n:], -e[1:]], [-1, -n, 0, n, 1], format="lil")
    border = ~mask
    rows = np.where(border.ravel())[0]
    for k in rows:
        L.rows[k] = [k]; L.data[k] = [1.0]
    b = np.where(mask, rhs, 0.0).ravel().copy()
    b[rows] = 0.0
    u = spla.spsolve(L.tocsr(), b)
    return u.reshape(n, n)

# ---------------- C^0 vs C^1 (both as DIFFERENCE of two admissible surfaces, |Delta w|<=2k) ----
n, c, R, kappa = 129, 64, 50.0, 1.0
yy, xx = np.mgrid[0:n, 0:n]
r = np.hypot(yy - c, xx - c)

# C^0: w|d = 0, |Delta w| <= 2k  ->  w(0) = kappa R^2 / 2   (diameter)
r0, w0, _ = radial_solution(R, lambda rr: np.full_like(rr, -2 * kappa))
pred0 = kappa * R**2 / 2
# C^1: w|d = dn w|d = 0, |Delta w| <= 2k  ->  w(0) = kappa R^2 ln2 / 2
rho = R / np.sqrt(2.0)
r1, w1, wpR = radial_solution(R, lambda rr: np.where(rr <= rho, -2 * kappa, 2 * kappa))
pred1 = kappa * R**2 * np.log(2) / 2

out = {
    "C0": dict(diameter=float(w0[0]), predicted=float(pred0), ratio=float(w0[0] / pred0),
               radius_constant_over_d2=float(w0[0] / 2 / (2 * R) ** 2)),
    "C1": dict(diameter=float(w1[0]), predicted=float(pred1), ratio=float(w1[0] / pred1),
               radius_constant_over_d2=float(w1[0] / 2 / (2 * R) ** 2),
               domega_slope_at_R=float(wpR), rho_over_R=float(rho / R)),
    "tightening": float(w0[0] / w1[0]),
    "diameter_gain_px": float(np.sqrt(w0[0] / w1[0])),
}

# ---------------- torsion function: Talenti + SLIVER scaling ----------------
n2, c2 = 161, 80
yy2, xx2 = np.mgrid[0:n2, 0:n2]
disk = (yy2 - c2) ** 2 + (xx2 - c2) ** 2 <= 30 ** 2
A = disk.sum()
shapes = {"disk": disk}
side = int(round(np.sqrt(A)))
shapes["square_equal_area"] = (np.abs(yy2 - c2) <= side // 2) & (np.abs(xx2 - c2) <= side // 2)
shapes["ellipse_2:1_equal_area"] = (((xx2 - c2) / (side / np.sqrt(2))) ** 2 +
                                    ((yy2 - c2) / (side / (2 * np.sqrt(2)))) ** 2) <= 1.0
# slivers of equal area with aspect 4:1, 8:1, 16:1  (shadow-like)
for ar in (4, 8, 16):
    w = int(round(np.sqrt(A / ar))); L = int(round(A / max(w, 1)))
    shapes[f"sliver_{ar}:1_equal_area"] = (np.abs(yy2 - c2) <= w // 2) & (np.abs(xx2 - c2) <= L // 2)
tors = {}
for k_, m in shapes.items():
    if m.sum() < 20: continue
    v = solve_dirichlet(n2, np.ones_like(yy2, float), m)
    d_eq = 2 * np.sqrt(m.sum() / np.pi)
    tors[k_] = dict(area=float(m.sum()), d_eq=float(d_eq), vmax=float(v.max()),
                    disk_bound=float(d_eq ** 2 / 16),
                    ratio_vmax_over_diskbound=float(v.max() / (d_eq ** 2 / 16)),
                    w_eff=float(np.sqrt(8 * v.max())))
out["torsion"] = tors

print("=" * 78)
print("C^0 (diameter): w(0) = %.2f   predicted kappa R^2/2 = %.2f   ratio %.4f"
      % (out["C0"]["diameter"], out["C0"]["predicted"], out["C0"]["ratio"]))
print("C^1 (diameter): w(0) = %.2f   predicted kappa R^2 ln2/2 = %.2f   ratio %.4f"
      % (out["C1"]["diameter"], out["C1"]["predicted"], out["C1"]["ratio"]))
print("  dn w at r=R (must be ~0) = %.2e ;  rho/R = %.4f (pred %.4f)"
      % (out["C1"]["domega_slope_at_R"], out["C1"]["rho_over_R"], 1 / np.sqrt(2)))
print("  tightening C^0 -> C^1 radius = %.4f   (1/ln2 = %.4f)" % (out["tightening"], 1 / np.log(2)))
print("  certified DIAMETER gain      = %.4f   (sqrt(1/ln2) = %.4f)"
      % (out["diameter_gain_px"], np.sqrt(1 / np.log(2))))
print("-" * 78)
print("torsion function, equal-area shapes (Talenti: the disk must be the maximum):")
for k_, v in tors.items():
    print("   %-24s area=%6.0f d_eq=%5.2f  ||v||inf=%.3f  diskbound=%.3f  ratio=%.3f"
          % (k_, v["area"], v["d_eq"], v["vmax"], v["disk_bound"], v["ratio_vmax_over_diskbound"]))
json.dump(out, open(os.path.join(RES, "theory_c1.json"), "w"), indent=1)
print("saved", os.path.join(RES, "theory_c1.json"))
