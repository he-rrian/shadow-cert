"""W4 -- baselines, ORR / ORR_noisy, Oracle violation check, joint (kappa, delta_g) calibration.

The task isolated here is HOLE FILLING given the observed phase on the clean region.
Unwrapping is not part of the comparison: on Omega_clean the observed phase is taken as the
GT plus the probe-A boundary noise, so every method sees exactly the same boundary data.

Baselines
  Identity      : fill the hole with the mean boundary value (piston only)
  Nearest       : nearest-boundary copy (cv2 distance transform)
  Harmonic      : harmonic extension of the OBSERVED boundary  (= the practical optimum)
  Telea / NS    : skimage inpainting of the observed phase
  ORR           : harmonic extension of the EXACT boundary  (oracle reference, from the shard)
  ORR_noisy     : harmonic extension of the OBSERVED boundary (= Harmonic, kept for the table)

Certificates
  Oracle : bound = kappa_oracle * ||v|| + delta_g     (kappa_oracle = P95 inside the hole)
  Deploy : bound = gamma * kappa_hat * ||v|| + delta_g  (kappa_hat = P75 on the annulus)
  Violation rate must be <= eps; gamma* is the smallest gamma on a grid that achieves it.

Run: python w4_baseline.py --shard ../../data/train/shard_099_0000.npz
"""
import os, json, argparse, glob
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "probes", "results")
DELTA_G = 0.13
PHI_TOL = float(np.pi / 10)

# ------------------------------------------------------------------ POISSON (harmonic fill)
def poisson_dirichlet(domain, dirichlet):
    """-Delta u = 0 inside `domain`, u = dirichlet on its boundary ring"""
    import scipy.sparse as sp, scipy.sparse.linalg as spla
    ys, xs = np.where(domain); pad = 3
    y0, y1 = max(0, ys.min() - pad), min(domain.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(domain.shape[1], xs.max() + pad + 1)
    m = domain[y0:y1, x0:x1]; n, k = m.shape; N = n * k; e = np.ones(N)
    A = sp.diags([-e[:-1], -e[:-k], 4 * e, -e[k:], -e[1:]], [-1, -k, 0, k, 1],
                 shape=(N, N), format="lil")
    for i in range(n):
        if i > 0: A[i * k, i * k - 1] = 0.0
        if i * k + k < N: A[i * k + k - 1, i * k + k] = 0.0
    bb = np.zeros(N)
    D = dirichlet[y0:y1, x0:x1]
    for i in range(n):
        for j in range(k):
            if not m[i, j]:
                a = i * k + j; A.rows[a] = [a]; A.data[a] = [1.0]; bb[a] = D[i, j]
    u = spla.spsolve(A.tocsr(), bb).reshape(n, k)
    out = np.zeros_like(domain, float); out[y0:y1, x0:x1] = u
    return out

def extend_from(mask, src):
    m = mask.astype(np.uint8)
    _, lab = cv2.distanceTransformWithLabels(1 - m, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    flat = src.ravel().astype(np.float64)
    acc = np.zeros(lab.max() + 1); cnt = np.zeros(lab.max() + 1)
    np.add.at(acc, lab.ravel()[mask.ravel()], flat[mask.ravel()])
    np.add.at(cnt, lab.ravel()[mask.ravel()], 1.0)
    acc = acc / np.maximum(cnt, 1)
    return acc[lab].reshape(src.shape)

def harmonic_fill(dead, boundary_src, collar=2):
    dom = cv2.dilate(dead.astype(np.uint8), np.ones((2 * collar + 1,) * 2, np.uint8)) > 0
    ring = dom & (~dead)
    if ring.sum() < 8:
        return np.zeros_like(boundary_src, dtype=float)
    D = extend_from(ring, np.where(ring, boundary_src, 0.0))
    u = poisson_dirichlet(dead & dom, D)
    return np.where(dead, u, 0.0)

# ------------------------------------------------------------------ baselines
def nearest_fill(dead, src):
    """nearest-known-pixel fill.  NOTE: cv2.distanceTransform measures the distance to the
    nearest ZERO pixel, so the mask must be `dead` (1 = unknown), NOT `~dead` -- passing the
    inverse reads the label of a dead pixel and produces garbage (that was the 99-225 rad bug)."""
    if not dead.any():
        return np.zeros_like(src, dtype=float)
    _, lab = cv2.distanceTransformWithLabels(dead.astype(np.uint8), cv2.DIST_L2, 5,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    known = (~dead).ravel()
    flat = src.ravel().astype(np.float64)
    acc = np.zeros(lab.max() + 1)
    cnt = np.zeros(lab.max() + 1)
    np.add.at(acc, lab.ravel()[known], flat[known])
    np.add.at(cnt, lab.ravel()[known], 1.0)
    acc = acc / np.maximum(cnt, 1)
    return np.where(dead, acc[lab], 0.0)

def identity_fill(dead, src):
    ring = (cv2.dilate(dead.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (~dead)
    v = float(np.mean(src[ring])) if ring.sum() else float(np.mean(src[~dead]))
    return np.where(dead, v, 0.0)

def biharmonic_fill(dead, src):
    """thin-plate (biharmonic) inpainting -- the classical 'smooth extrapolation' baseline"""
    from skimage.restoration import inpaint_biharmonic
    base = float(np.mean(src[~dead]))
    img = np.where(dead, 0.0, src) - base
    return inpaint_biharmonic(img.astype(np.float64), dead.astype(np.uint8)) + base

def poly_fill(dead, src, deg=2):
    """quadratic surface fitted on a band around the hole, evaluated inside it"""
    band = (cv2.dilate(dead.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0) & (~dead)
    ys, xs = np.where(band)
    if len(ys) < 30:
        return identity_fill(dead, src)
    H, W = src.shape
    yy, xx = np.mgrid[0:H, 0:W]
    def basis(x, y):
        x = (x - W / 2) / (W / 2); y = (y - H / 2) / (H / 2)
        cols = []
        for i in range(deg + 1):
            for j in range(deg + 1 - i):
                cols.append((x ** j) * (y ** i))
        return np.stack(cols, -1)
    B = basis(xx[ys, xs], yy[ys, xs])
    coef, *_ = np.linalg.lstsq(B, src[ys, xs], rcond=None)
    Bd = basis(xx[dead], yy[dead])
    out = np.zeros_like(src, dtype=float)
    out[dead] = Bd @ coef
    return out

# ------------------------------------------------------------------ metrics
def stats(e):
    return dict(p50=float(np.median(e)), p95=float(np.percentile(e, 95)),
                mean=float(np.mean(e)), frac_tol=float(np.mean(e > PHI_TOL)))

def run_shard(path, n_max=64):
    d = np.load(path)
    metas = [json.loads(s) for s in d["meta"]]
    rows = []
    for i in range(min(len(metas), n_max)):
        m = metas[i]
        dead = d["dead"][i]; roi = d["roi"][i]
        if dead.sum() < 30:
            continue
        phi = d["phi"][i].astype(np.float64)
        orr = d["orr"][i].astype(np.float64)
        vinf = m.get("vinf")
        if vinf is None:
            continue
        rng = np.random.default_rng(i)
        # observed phase on the clean region = GT + probe-A boundary-level noise
        src = phi + rng.normal(0, DELTA_G / 2.0, phi.shape) * (~dead)
        out = {}
        fills = {
            "Identity": identity_fill(dead, src),
            "Nearest": nearest_fill(dead, src),
            "Harmonic": harmonic_fill(dead, src),
            "Biharm": biharmonic_fill(dead, src),
            "Poly2": poly_fill(dead, src),
            "ORR": orr,
        }
        for k, f in fills.items():
            e = np.abs(f[dead] - phi[dead])
            out[k] = stats(e)
        rel = {k: (out[k]["mean"] / max(out["ORR"]["mean"], 1e-9)) for k in out}
        # certificates
        k_or = float(m["kappa_oracle"]); k_hat = float(m["kappa_hat"])
        bound_or = k_or * vinf + DELTA_G
        err_or = out["Harmonic"]["p95"]        # the realised error of the practical optimum
        rows.append(dict(id=m["id"], branch=m["branch"], cls=m["cls"], vinf=vinf,
                         k_or=k_or, k_hat=k_hat, err_orr=out["ORR"]["mean"],
                         err_har=out["Harmonic"]["mean"], rel_orr=rel["Harmonic"],
                         **{f"{k}_p95": out[k]["p95"] for k in out},
                         **{f"rel_{k}": rel[k] for k in out},
                         bound_or=bound_or, viol_or=bool(err_or > bound_or)))
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=os.path.join(ROOT, "data", "train", "shard_099_0000.npz"))
    ap.add_argument("--glob", default=None)
    a = ap.parse_args()
    paths = sorted(glob.glob(a.glob)) if a.glob else [a.shard]
    rows = []
    for p in paths:
        try:
            rows += run_shard(p)
        except Exception as ex:
            print(f"{os.path.basename(p)}: {type(ex).__name__} {ex}")
    if not rows:
        raise SystemExit("no rows")
    print(f"samples evaluated: {len(rows)}\n")
    keys = ["Identity", "Nearest", "Harmonic", "Biharm", "Poly2", "ORR"]
    print(f"{'method':10s}{'P95 err (rad)':>16s}{'Rel-ORR':>10s}")
    for k in keys:
        print(f"{k:10s}{np.median([r[f'{k}_p95'] for r in rows]):16.4f}"
              f"{np.median([r[f'rel_{k}'] for r in rows]):10.3f}")
    print(f"\nOracle bound: kappa_oracle*||v|| + delta_g")
    vo = np.mean([r["viol_or"] for r in rows])
    print(f"  violation rate = {vo*100:.1f}%  (n={len(rows)})")
    # gamma* calibration on the DEPLOY bound
    print("\ngamma* calibration (deploy bound = gamma * kappa_hat * ||v|| + delta_g), eps = 0.10:")
    for eps in (0.05, 0.10, 0.20):
        best = None
        for g in np.arange(0.5, 6.01, 0.1):
            bad = np.mean([r["err_har"] > (g * r["k_hat"] * r["vinf"] + DELTA_G) for r in rows])
            if bad <= eps:
                best = (g, bad); break
        print(f"  eps={eps:.2f}: gamma*={'-' if best is None else f'{best[0]:.2f}'} "
              f"(violation {0 if best is None else best[1]*100:.0f}%)")
    json.dump(rows, open(os.path.join(RES, "w4_baseline.json"), "w"), indent=1, default=float)
    print("\nsaved", os.path.join(RES, "w4_baseline.json"))
