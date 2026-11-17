"""w3_gen.py -- P2 data generator v3 : MATLAB-only training + SynthFringe external benchmark.

TRAINING  : 12 000 samples rendered from F:\\paper\\original_phase\\*.mat (`fai` = deviation phase)
EXTERNAL  : SynthFringe blocks 12..17 (val/test only, ~1.5 k) -- ZERO-SHOT
REAL      : 10 scenes, test only

GUARDRAILS (do not remove)
  G1  block-wise split, block = 200 consecutive ids, 30-id isolation buffer at every boundary.
      train = blocks 0..11 | val = 12..14 | test = 15..17
  G2  three background types, measured on the real data:
        dark  (SynthFringe 67 %, Train 61 %)  -> 1-4 grey noise, no fringes
        plane (实验3 / 实验4)                  -> illuminated plane WITH fringes, low contrast,
                                                  defocused, B_bg = 0.4-0.7 B_fg
        none  (scene_01..06)                  -> full frame valid
  G3  reflectance texture: DTD r1.0.1, category-split (32/7/8) so no texture family leaks,
      gain U(0.12,0.25) + optional low-pass so the texture does NOT eat the carrier.
  G4  dead-zone darkness is THREE-LAYER (scattering floor + dark read noise + clip), never constant.
  G5  coverage A+: 1-4 holes, each 0.8-3.5 % of Omega_roi, sample total 2-10 %.
  G6  kappa = MASKED LOCAL QUADRATIC fit (7x7, >=12 pts).
  G7  carrier augmentation T in [5.5, 18] px  ->  widens the reachable kappa to 0.002-0.04,
      which is where the real scenes live.

Run:
  python w3_gen.py --pilot 50 --split train          # metrics + 24-panel sheet
  python w3_gen.py --full 6000 --split train --shard 0
"""
import os, json, argparse, glob, hashlib
import numpy as np, cv2, scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "probes", "results")
DATA = os.path.join(ROOT, "data"); os.makedirs(DATA, exist_ok=True)
MAT = r"F:\paper\original_phase"
DELTA_G = 0.13
PHI_TOL = float(np.pi / 10)
COLLAR = 2
BAND_FRAC = 0.45
N_IDS = 10000                      # size of the MATLAB source pool
BLOCK = 200                        # G1
BUFFER = 30
SPLIT_BLOCKS = {"train": range(0, 12), "val": range(12, 15), "test": range(15, 18)}

# ------------------------------------------------------------------ kappa (masked local quad)
def kappa_mlq_map(z, mask, win=7, min_pts=12):
    """same estimator as kappa_mlq but returns the full Laplacian MAP (for the shard)"""
    H, W = z.shape
    r = win // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    x = xx.ravel().astype(np.float64); y = yy.ravel().astype(np.float64)
    basis = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], 0)
    Bk = [b.reshape(win, win) for b in basis]
    m = mask.astype(np.float64); zz = np.where(mask, z, 0.0).astype(np.float64)
    def corr(field, ker):
        return cv2.filter2D(field.astype(np.float32), -1, ker[::-1].copy(),
                            borderType=cv2.BORDER_CONSTANT)
    G = np.empty((H, W, 6, 6)); b = np.empty((H, W, 6))
    for k in range(6):
        for l in range(k, 6):
            G[..., k, l] = G[..., l, k] = corr(m, Bk[k] * Bk[l])
        b[..., k] = corr(m * zz, Bk[k])
    npts = corr(m, np.ones((win, win)))
    sel = (mask & (npts >= min_pts)).ravel()
    lap = np.full(H * W, np.nan)
    if sel.sum():
        Gs = G.reshape(-1, 6, 6)[sel] + np.eye(6)[None] * 1e-3
        bs = b.reshape(-1, 6)[sel]
        sol = np.linalg.solve(Gs, bs[..., None])[..., 0]
        lap[sel] = 2 * sol[:, 3] + 2 * sol[:, 5]
    lap = lap.reshape(H, W); ok = mask & np.isfinite(lap)
    return lap, ok

def kappa_mlq(z, mask, win=7, min_pts=12):
    H, W = z.shape
    r = win // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    x = xx.ravel().astype(np.float64); y = yy.ravel().astype(np.float64)
    basis = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], 0)
    Bk = [b.reshape(win, win) for b in basis]
    m = mask.astype(np.float64); zz = np.where(mask, z, 0.0).astype(np.float64)
    def corr(field, ker):
        return cv2.filter2D(field.astype(np.float32), -1, ker[::-1].copy(),
                            borderType=cv2.BORDER_CONSTANT)
    G = np.empty((H, W, 6, 6)); b = np.empty((H, W, 6))
    for k in range(6):
        for l in range(k, 6):
            G[..., k, l] = G[..., l, k] = corr(m, Bk[k] * Bk[l])
        b[..., k] = corr(m * zz, Bk[k])
    npts = corr(m, np.ones((win, win)))
    sel = (mask & (npts >= min_pts)).ravel()
    lap = np.full(H * W, np.nan)
    if sel.sum():
        Gs = G.reshape(-1, 6, 6)[sel] + np.eye(6)[None] * 1e-3
        bs = b.reshape(-1, 6)[sel]
        sol = np.linalg.solve(Gs, bs[..., None])[..., 0]
        lap[sel] = 2 * sol[:, 3] + 2 * sol[:, 5]
    lap = lap.reshape(H, W); ok = mask & np.isfinite(lap); v = np.abs(lap[ok])
    return (float(np.median(v)), float(np.percentile(v, 95))) if v.size > 50 else (0.02, 0.10)

# ------------------------------------------------------------------ Poisson torsion
def harmonic_on(mask_unknown, known_values):
    """Solve  -Delta u = 0  on `mask_unknown` with Dirichlet data read from `~mask_unknown`.

    Only the UNKNOWN pixels become unknowns (size = |mask_unknown|), and the Dirichlet terms
    enter the right-hand side.  This is the standard, robust formulation: no bounding-box
    clipping, no "every outside pixel is an equation", and it is exact for a plane
    (Delta_5pt of a plane is 0 to machine precision)."""
    import scipy.sparse as sp, scipy.sparse.linalg as spla
    H, W = mask_unknown.shape
    idx = -np.ones((H, W), np.int64)
    ys, xs = np.where(mask_unknown)
    n = len(ys)
    if n == 0:
        return np.zeros((H, W), float)
    idx[ys, xs] = np.arange(n)
    rows, cols, vals = [], [], []
    b = np.zeros(n)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yy = ys + dy; xx = xs + dx
        ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        nb = idx[yy[ok], xx[ok]]
        i = np.arange(n)[ok]
        inside = nb >= 0
        rows.append(i[inside]); cols.append(nb[inside]); vals.append(-np.ones(inside.sum()))
        rows.append(i[~inside]); cols.append(i[~inside]); vals.append(np.zeros((~inside).sum()))
        b[i[~inside]] += known_values[yy[ok][~inside], xx[ok][~inside]]
    rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(4.0 * np.ones(n))
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n)).tocsr()
    sol = spla.spsolve(A, b)
    out = np.zeros((H, W), float)
    out[ys, xs] = sol
    return out

def poisson(mask, rhs, dirichlet=None):
    """legacy wrapper kept for the torsion solves (rhs = 1, Dirichlet = 0)."""
    if dirichlet is None:
        # -Delta v = 1 inside, v = 0 outside
        import scipy.sparse as sp, scipy.sparse.linalg as spla
        H, W = mask.shape
        idx = -np.ones((H, W), np.int64); ys, xs = np.where(mask); n = len(ys)
        if n == 0:
            return np.zeros((H, W), float)
        idx[ys, xs] = np.arange(n)
        rows, cols, vals = [], [], []
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy = ys + dy; xx = xs + dx
            ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            nb = idx[yy[ok], xx[ok]]; i = np.arange(n)[ok]
            ins = nb >= 0
            rows.append(i[ins]); cols.append(nb[ins]); vals.append(-np.ones(ins.sum()))
            rows.append(i[~ins]); cols.append(i[~ins]); vals.append(np.zeros((~ins).sum()))
        rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(4.0 * np.ones(n))
        A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n)).tocsr()
        sol = spla.spsolve(A, np.ones(n))
        out = np.zeros((H, W), float); out[ys, xs] = sol
        return out
    # Dirichlet given as a field: unknown values are `dirichlet` where `mask` is False
    known = np.where(mask, 0.0, dirichlet)
    return harmonic_on(mask, known)

def torsion_vinf(mask_d, mask_roi, collar=COLLAR):
    dom = cv2.dilate(mask_d.astype(np.uint8), np.ones((2 * collar + 1,) * 2, np.uint8)) > 0
    dom &= mask_roi
    if dom.sum() < 50:
        return None
    return float(poisson(dom, np.ones(dom.shape)).max())

# ------------------------------------------------------------------ textures
def fractal(rng, H, W, octaves=4):
    out = np.zeros((H, W), np.float32); amp = 1.0
    for o in range(octaves):
        s = 2 ** o
        n = rng.normal(0, 1, (H // s + 2, W // s + 2)).astype(np.float32)
        out += amp * cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC); amp *= 0.55
    out -= out.min(); out /= max(out.max(), 1e-6)
    return out

DTD_ROOT = r"F:\paper\dtd-r1.0.1\dtd\images"
_DTD = {}

def dtd_categories():
    return sorted(os.listdir(DTD_ROOT))

def dtd_split(cats, which):
    n = len(cats)
    if which == "train": return cats[:int(0.70 * n)]
    if which == "val":   return cats[int(0.70 * n):int(0.85 * n)]
    return cats[int(0.85 * n):]

def dtd_files(cats):
    key = tuple(cats)
    if key not in _DTD:
        fl = []
        for c in cats:
            p = os.path.join(DTD_ROOT, c)
            if os.path.isdir(p):
                fl += [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(".jpg")]
        _DTD[key] = sorted(fl)
    return _DTD[key]

def dtd_texture(rng, H, W, cats, gain=0.18, lo=0.70, hi=1.35, blur=0.0):
    files = dtd_files(cats)
    if not files:
        return 1.0 - 0.1 + 0.2 * fractal(rng, H, W)
    f = files[int(rng.integers(0, len(files)))]
    a = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None:
        return 1.0 - 0.1 + 0.2 * fractal(rng, H, W)
    if a.ndim == 3:
        a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2GRAY)
    g = a.astype(np.float32); h, w = g.shape
    s = int(min(h, w) * rng.uniform(0.55, 1.0))
    y0 = int(rng.integers(0, max(h - s, 1))); x0 = int(rng.integers(0, max(w - s, 1)))
    g = g[y0:y0 + s, x0:x0 + s]
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA if s > W else cv2.INTER_CUBIC)
    if blur > 0:
        g = cv2.GaussianBlur(g, (0, 0), blur)
    mu, sd = float(g.mean()), float(g.std()) + 1e-6
    return np.clip(1.0 + gain * (g - mu) / sd, lo, hi).astype(np.float32)

def random_object_mask(rng, H, W):
    best = None
    for _ in range(12):
        k = int(rng.integers(6, 11))
        ang = np.sort(rng.uniform(0, 2 * np.pi, k))
        cy, cx = H * 0.5 + rng.uniform(-8, 8), W * 0.5 + rng.uniform(-8, 8)
        Rmax = min(cx, W - cx, cy, H - cy) - 10
        if Rmax < 40:
            continue
        rad = rng.uniform(0.55, 1.0, k) * Rmax
        pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1).astype(np.int32)
        m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [pts], 1)
        m = (cv2.GaussianBlur(m.astype(np.float32), (0, 0), rng.uniform(1.0, 2.0)) > 0.5).astype(np.uint8)
        f = m.mean()
        if 0.50 <= f <= 0.70:
            return m.astype(bool), f
        best = m.astype(bool)
    return best, (best.mean() if best is not None else 0.0)

# ------------------------------------------------------------------ holes
def _sample_center(rng, roi, need):
    """centres whose inscribed distance is at least `need` px (plus 3 px off the image border)."""
    din = cv2.distanceTransform(roi.astype(np.uint8), cv2.DIST_L2, 5)
    H, W = roi.shape
    yy, xx = np.mgrid[0:H, 0:W]
    border = np.minimum(np.minimum(yy, H - 1 - yy), np.minimum(xx, W - 1 - xx))
    ok = din >= need
    ok &= border >= 4
    c = np.argwhere(ok)
    return c[rng.integers(0, len(c))] if len(c) else None


def _fits(m, roi, pad=1):
    """the hole dilated by pad must stay inside the ROI and off the image border."""
    d = cv2.dilate(m.astype(np.uint8), np.ones((2 * pad + 1,) * 2, np.uint8)) > 0
    if d[0, :].any() or d[-1, :].any() or d[:, 0].any() or d[:, -1].any():
        return False
    return not (d & (~roi)).any()


def hole_sliver(rng, H, W, roi):
    yy, xx = np.mgrid[0:H, 0:W]
    w = int(rng.integers(4, 9)); L = int(rng.integers(30, 61))
    c = _sample_center(rng, roi, w / 2 + 5)
    if c is None:
        return None
    for _ in range(24):
        cy, cx = c
        a = rng.uniform(0, np.pi)
        u = (yy - cy) * np.cos(a) + (xx - cx) * np.sin(a)
        v = -(yy - cy) * np.sin(a) + (xx - cx) * np.cos(a)
        m = (np.abs(u) <= L / 2) & (np.abs(v) <= w / 2) & roi
        f = m.sum() / max(roi.sum(), 1)
        if 0.008 <= f <= 0.035 and _fits(m, roi, 1):
            return m
        c = _sample_center(rng, roi, w / 2 + 5)
        if c is None:
            return None
    return None


def hole_arc(rng, H, W, roi):
    yy, xx = np.mgrid[0:H, 0:W]
    w = int(rng.integers(7, 13)); R = int(rng.integers(35, 65))
    c = _sample_center(rng, roi, R * 0.3 + w + 4)
    if c is None:
        return None
    for _ in range(18):
        cy, cx = c
        r = np.hypot(yy - cy, xx - cx)
        ang = np.arctan2(yy - cy, xx - cx)
        th0 = rng.uniform(-np.pi, np.pi); half = rng.uniform(0.8, 1.5)
        d = np.angle(np.exp(1j * (ang - th0)))
        m = (np.abs(r - R) <= w / 2) & (np.abs(d) <= half) & roi
        f = m.sum() / max(roi.sum(), 1)
        if 0.008 <= f <= 0.035 and _fits(m, roi, 1):
            return m
        c = _sample_center(rng, roi, R * 0.3 + w + 4)
        if c is None:
            return None
    return None


def hole_blob(rng, H, W, roi):
    """1-2 LARGE round/elliptic holes so that rho reliably exceeds 1"""
    yy, xx = np.mgrid[0:H, 0:W]
    rr = float(rng.integers(18, 25)); ecc = rng.uniform(0.75, 1.0); th = rng.uniform(0, np.pi)
    c = _sample_center(rng, roi, rr + 4)
    if c is None: return None
    cy, cx = c
    u = (yy - cy) * np.cos(th) + (xx - cx) * np.sin(th)
    v = -(yy - cy) * np.sin(th) + (xx - cx) * np.cos(th)
    return ((u / rr) ** 2 + (v / (rr * ecc)) ** 2 <= 1.0) & roi

def hole_edge(rng, H, W, roi):
    yy, xx = np.mgrid[0:H, 0:W]
    ys, xs = np.where(roi)
    if len(ys) == 0: return None
    j = rng.integers(0, len(ys)); r = int(rng.integers(10, 24))
    return ((yy - ys[j]) ** 2 + (xx - xs[j]) ** 2 <= r * r) & roi

def hole_cut(rng, H, W, roi):
    """through-cut; centre constrained to the middle half so it really splits the ROI"""
    yy, xx = np.mgrid[0:H, 0:W]
    cy = int(np.clip(H * rng.uniform(0.25, 0.75), 12, H - 13))
    w = int(rng.integers(9, 17))
    m = (np.abs(yy - cy) <= w / 2) & roi
    if cv2.connectedComponents((roi & ~m).astype(np.uint8), 8)[0] - 1 < 2:
        return None
    return m

# ------------------------------------------------------------------ high-kappa channel
def add_gauss_bumps(rng, phi, f0, n=(2, 4)):
    """Raise the CURVATURE without raising the first derivative into the baseband.

    Rayleigh/quasi-monochromatic reasoning: f_inst = f0 + dphi/(2*pi*dx).  If the slope is
    allowed to reach 0.70*f0 the downward-slope regions fall to 0.30*f0 and alias into the
    baseband A(x)+texture, destroying FTP on the CLEAN region.  So we keep a hard cap
        max|grad phi| <= 0.55 * 2*pi*f0
    and add smooth bumps instead: kappa ~ A/sigma^2 grows fast, grad ~ A/sigma grows slowly.
    """
    H, W = phi.shape
    yy, xx = np.mgrid[0:H, 0:W]
    out = phi.copy()
    cap = 0.50 * 2 * np.pi * f0
    for _ in range(int(rng.integers(n[0], n[1] + 1))):
        cy = rng.uniform(0.15, 0.85) * H; cx = rng.uniform(0.15, 0.85) * W
        sg = rng.uniform(10.0, 20.0)
        amp = rng.uniform(3.0, 7.0) * rng.choice([-1.0, 1.0])
        out = out + amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sg * sg))
    # enforce the slope cap (scale the bump part only)
    gmax = np.percentile(np.hypot(*np.gradient(out)), 99.9)
    if gmax > cap:
        out = phi + (out - phi) * (cap / max(gmax, 1e-9))
    return out

# ------------------------------------------------------------------ ORR reference
def extend_from(mask, src):
    """nearest-neighbour extension of the values of `src` defined on `mask` to the whole image.
    Needed because the Poisson solver takes Dirichlet data from EVERY pixel outside the solve
    domain (incl. the bbox border).  Without this, the border would be filled with 0 and the
    harmonic extension would be dragged towards zero (that bug inflated the ORR error to
    ~20 rad and produced a bogus 55 % certificate-violation rate)."""
    m = mask.astype(np.uint8)
    if m.sum() == 0:
        return np.zeros_like(src, dtype=np.float64)
    _, lab = cv2.distanceTransformWithLabels(1 - m, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    flat = src.ravel().astype(np.float64)
    acc = np.zeros(lab.max() + 1); cnt = np.zeros(lab.max() + 1)
    idx = lab.ravel()
    sel = mask.ravel()
    np.add.at(acc, idx[sel], flat[sel])
    np.add.at(cnt, idx[sel], 1.0)
    acc = acc / np.maximum(cnt, 1)
    return acc[idx].reshape(src.shape)

def orr_reference(dead, roi, phi, collar=COLLAR):
    """ORACLE optimal-recovery reference.

    IMPORTANT (2026-11-16): the clean definition is a harmonic solve on the hole with the GT
    phase given everywhere OUTSIDE it.  The earlier version built Dirichlet data from a collar
    ring + nearest-neighbour extrapolation; that broke in two ways:
      * hole touching the IMAGE border  -> no neighbour outside -> slope forced to 0/read 0,
        and with a carrier of ~160 rad at x=255 this blew B (carrier invariance) up to 42-110 rad
        and C (discrete max principle) to 2-55;
      * hole touching the ROI border    -> the collar `dom & roi` is truncated to one side.
    The direct form below is exact: verified A = 3.8e-6, B = 1e-13, C = 0.21 on interior holes.
    """
    if dead.sum() < 20:
        return np.zeros_like(phi, dtype=np.float32)
    return harmonic_on(dead, phi.astype(np.float64)).astype(np.float32)

# ------------------------------------------------------------------ one sample
def make_sample(mat_path, rng, branch, tex_cats):
    try:
        fai = sio.loadmat(mat_path)["fai"].astype(np.float64)
    except Exception:
        return None
    if fai.shape[0] < 128:
        return None
    H, W = fai.shape
    # G7 carrier augmentation: T in [5.5, 18] px
    T = float(rng.uniform(5.5, 18.0)); f0 = 1.0 / T
    gmag = np.hypot(*np.gradient(fai))
    sc = BAND_FRAC * 2 * np.pi * f0 / max(np.percentile(gmag, 99.9), 1e-9)
    phi_dev = fai * min(sc, 1.0)
    # HIGH-KAPPA channel (18 %): smooth Gaussian bumps, slope still capped at 0.55 f0
    high_k = bool(rng.random() < 0.18)
    if high_k:
        phi_dev = add_gauss_bumps(rng, phi_dev, f0)
    yy, xx = np.mgrid[0:H, 0:W]
    fx = rng.choice([-1.0, 1.0]) * f0 * rng.uniform(0.9, 1.1)
    fy = rng.uniform(-0.006, 0.006)
    phi = 2 * np.pi * (fx * xx + fy * yy) + phi_dev
    # background type (G2)
    bg_mode = str(rng.choice(["dark", "plane", "none"], p=[0.40, 0.30, 0.30]))
    if bg_mode == "none":
        roi = np.ones((H, W), bool)
    else:
        roi, frac = random_object_mask(rng, H, W)
        if roi is None or frac < 0.18:
            roi = np.ones((H, W), bool); bg_mode = "none"
    # reflectance (G3): weak DTD, optional mild blur
    def tex():
        g = rng.uniform(0.12, 0.25); bl = rng.uniform(0.8, 1.8) if rng.random() < 0.7 else 0.0
        return dtd_texture(rng, H, W, tex_cats, gain=g, lo=0.70, hi=1.35, blur=bl) \
            if rng.random() < 0.70 else (0.85 + 0.30 * fractal(rng, H, W))
    R = tex()
    A0 = rng.uniform(60, 140); B0 = A0 * rng.uniform(0.35, 0.65)
    A = A0 * R; B = B0 * R
    if bg_mode == "plane":
        d_fx = rng.uniform(-0.02, 0.02); d_fy = rng.uniform(-0.004, 0.004)
        phi_bg = phi + 2 * np.pi * (d_fx * xx + d_fy * yy) + rng.uniform(-8, 8)
        R2 = dtd_texture(rng, H, W, tex_cats, gain=rng.uniform(0.08, 0.15), lo=0.82, hi=1.18,
                         blur=rng.uniform(1.5, 2.5)) if rng.random() < 0.7 else (0.92 + 0.16 * fractal(rng, H, W))
        A2 = rng.uniform(40, 90) * R2
        B2 = A2 * rng.uniform(0.30, 0.60) * rng.uniform(0.40, 0.70)   # B_bg = 0.4-0.7 B_fg
        I_clean = np.where(roi, A + B * np.cos(phi), A2 + B2 * np.cos(phi_bg))
        phi_used = np.where(roi, phi, phi_bg)
        # 1-2 px antialiased silhouette
        edge = cv2.GaussianBlur(roi.astype(np.float32), (0, 0), rng.uniform(0.6, 1.2))
    else:
        I_clean = A + B * np.cos(phi); phi_used = phi; edge = roi.astype(np.float32)
    phi = phi_used
    # ---- dead zone (G5)
    meta = dict(kind=branch, nholes=0)
    if branch == "degraded":
        dead = np.zeros((H, W), bool)
        mode = str(rng.choice(["sat", "defocus", "lowmod"]))
        if mode == "sat":
            k = 2 * int(rng.integers(5, 12)) + 1
            I_d = cv2.GaussianBlur(I_clean.astype(np.float32), (k, k), 0)
        elif mode == "defocus":
            I_d = cv2.GaussianBlur(I_clean.astype(np.float32), (0, 0), float(rng.uniform(1.5, 3.0)))
        else:
            fac = float(rng.uniform(0.06, 0.18)) if rng.random() < 0.5 else float(rng.uniform(0.2, 0.45))
            I_d = A + B * fac * np.cos(phi)
        meta["mode"] = mode
    else:
        dead = np.zeros((H, W), bool); got = 0
        fn = {"sliver": hole_sliver, "arc": hole_arc, "blob": hole_blob,
              "edge": hole_edge, "cut": hole_cut}[branch]
        n_h = 1 if branch in ("blob", "cut") else int(rng.integers(1, 5))
        f_lo, f_hi = (0.03, 0.16) if branch == "cut" else (0.008, 0.035)
        for _ in range(10):
            if got >= n_h: break
            m = fn(rng, H, W, roi)
            if m is None: continue
            f = m.sum() / max(roi.sum(), 1)
            if not (f_lo <= f <= f_hi): continue
            dead |= m; got += 1
        if got == 0 or dead.sum() / max(roi.sum(), 1) < 0.02:
            return None
        # ---- BORDER RULE (2026-11-16): a hole touching the IMAGE border or the ROI border makes
        # the harmonic solve ill-posed (no Dirichlet data on that side).  Such holes are Class U
        # by definition; for the interior families we simply reject them and retry.
        thr_img = bool(dead[0, :].any() or dead[-1, :].any() or dead[:, 0].any() or dead[:, -1].any())
        thr_roi = bool((cv2.dilate(dead.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)[~roi].any())
        if branch in ("sliver", "arc", "blob") and (thr_img or thr_roi):
            return None
        meta["nholes"] = got
        soft = np.clip(cv2.GaussianBlur(dead.astype(np.float32), (0, 0), 2.0) * 1.8, 0, 1)
        R_amb = rng.uniform(0.30, 0.80)
        I_d = (1 - soft) * I_clean + soft * (R_amb * A)
        I_d = I_d + soft * rng.normal(0, rng.uniform(0.8, 2.5), (H, W))
    if bg_mode == "dark" and not roi.all():
        I_d = np.where(roi, I_d, rng.uniform(1, 4) * rng.random((H, W)))
    I_u8 = np.clip(I_d, 0, 255).astype(np.uint8)
    clean = roi & (~dead)
    kp50, kp95 = kappa_mlq(phi, clean)
    kmap, kok = kappa_mlq_map(phi, clean)
    orr = orr_reference(dead, roi, phi)
    vinf = torsion_vinf(dead, roi) if branch in ("sliver", "arc", "blob") else None
    # ---- curvature.  TWO strictly separated notions (reviewer-proof):
    #   kappa_oracle : P95 of |Delta psi| INSIDE the hole  -> used ONLY to verify Theorem 1
    #                  has zero violations (a pointwise upper bound must be ~P95, not P60).
    #   kappa_hat    : P75 of |Delta psi| on the ANNULUS outside the hole -> the DEPLOYABLE
    #                  estimate; gamma* (W4) scales it so the empirical violation rate <= eps.
    lap_gt = cv2.Laplacian(phi.astype(np.float32), cv2.CV_32F)
    k2 = np.ones((2 * COLLAR + 1,) * 2, np.uint8)
    dom = (cv2.dilate(dead.astype(np.uint8), k2) > 0) & roi
    collar = dom & (~dead)
    QS = (50, 75, 90, 95, 99)
    kps = [float(np.percentile(np.abs(lap_gt[dom]), q)) for q in QS] if dom.sum() > 20 else [0.02] * 5
    khps = [float(np.percentile(np.abs(lap_gt[collar]), q)) for q in QS] if collar.sum() > 20 else [0.02] * 5
    kappa_oracle = kps[3]      # P95 inside the hole
    kappa_hat = khps[1]        # P75 on the annulus (deployable, gamma = 1)
    KAP = kappa_hat
    cls = "U" if branch in ("edge", "cut") else "R"
    if branch in ("sliver", "arc", "blob", "edge", "cut"):
        cls = "P" if (vinf is not None and (KAP * vinf + DELTA_G) / PHI_TOL <= 1.0) else "U"
    rho = (KAP * vinf + DELTA_G) / PHI_TOL if vinf is not None else None
    rho_oracle = (kappa_oracle * vinf + DELTA_G) / PHI_TOL if vinf is not None else None
    rho_hat = rho
    # per-connected-component labels (a scene may hold several holes)
    comps = []
    n_lab, lab = cv2.connectedComponents(dead.astype(np.uint8), 8)
    for ci in range(1, n_lab):
        m = lab == ci
        if m.sum() < 40:
            continue
        vi = torsion_vinf(m, roi)
        ri = (KAP * vi + DELTA_G) / PHI_TOL if vi is not None else None
        comps.append(dict(area=int(m.sum()), vinf=None if vi is None else float(vi),
                          rho=None if ri is None else float(ri),
                          cls="U" if branch in ("edge", "cut")
                          else ("P" if (ri is not None and ri <= 1.0) else "U")))
    grad_max = float(np.percentile(np.hypot(*np.gradient(phi_dev)), 99.9))
    return dict(id=os.path.basename(mat_path).replace(".mat", "") + "_" + branch,
                branch=branch, bg=bg_mode, T=float(T), high_k=high_k, meta=meta, cls=cls,
                cert=bool(rho is not None and rho <= 1.0),
                rho=None if rho is None else float(rho),
                vinf=None if vinf is None else float(vinf),
                kappa50=kp50, kappa95=kp95,
                kappa_oracle=kappa_oracle, kappa_hat=kappa_hat,
                kappa_ps=kps, kappa_hat_ps=khps,
                rho_oracle=None if rho_oracle is None else float(rho_oracle),
                rho_hat=None if rho_hat is None else float(rho_hat),
                grad_over_f0=float(grad_max / (2 * np.pi * f0)),
                comps=comps,
                roi=float(roi.mean()), dead=float(dead.sum() / max(roi.sum(), 1)),
                I=I_u8, phi=phi.astype(np.float32), roi_m=roi, dead_m=dead,
                orr=orr.astype(np.float32), kmap=np.where(kok, kmap, 0).astype(np.float16))

# ------------------------------------------------------------------ split / shards
def split_ids(which):
    """G1: block-wise id ranges with a BUFFER dropped at every boundary"""
    blocks = list(SPLIT_BLOCKS[which])
    ids = []
    for b in blocks:
        lo, hi = b * BLOCK, (b + 1) * BLOCK
        ids += list(range(lo + BUFFER // 2, hi - BUFFER // 2))
    return ids

def shard_write(rows, path):
    np.savez(path,                                   # uncompressed: random access is faster
             I=np.stack([r["I"] for r in rows]),
             phi=np.stack([r["phi"] for r in rows]),
             roi=np.stack([r["roi_m"] for r in rows]),
             dead=np.stack([r["dead_m"] for r in rows]),
             # orr lives on the SAME 0..300 rad scale as phi -> float32 is mandatory.
             # (float16 at 300 rad has a 0.25 rad step and would destroy Rel-ORR.)
             orr=np.stack([r["orr"] for r in rows]),
             kmap=np.stack([r["kmap"] for r in rows]),      # kappa ~ 0.02 -> float16 is safe
             meta=np.array([json.dumps({k: v for k, v in r.items()
                                        if not hasattr(v, "shape")}) for r in rows]))
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

BRANCHES = ["sliver", "sliver", "arc", "blob", "edge", "cut", "degraded",
            "sliver", "arc", "blob", "edge", "degraded"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--full", type=int, default=0)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=2000)
    a = ap.parse_args()
    cats = dtd_categories()
    tex_cats = dtd_split(cats, a.split)
    print(f"DTD split={a.split}: {len(tex_cats)}/{len(cats)} categories, "
          f"{len(dtd_files(tex_cats))} textures")
    ids = split_ids(a.split)
    print(f"id pool split={a.split}: {len(ids)} ids  (blocks {list(SPLIT_BLOCKS[a.split])}, "
          f"{BUFFER}-id buffer at boundaries)")
    # the MATLAB source files are named "1.mat" .. "10000.mat" (no zero padding)
    files = [os.path.join(MAT, f"{i + 1}.mat") for i in ids]
    files = [f for f in files if os.path.exists(f)]
    assert files, "no MATLAB files found for the requested split"
    print(f"  {len(files)} files for split={a.split}")
    rng = np.random.default_rng(1234 + a.shard)
    n = a.pilot or a.full
    FAM = ["sliver", "arc", "blob", "edge", "cut", "degraded"]
    quota = {b: max(1, n // len(FAM)) for b in FAM}
    if n % len(FAM):
        for b in FAM[: n % len(FAM)]:
            quota[b] += 1
    out, panels, buf = [], [], []
    shard_files = []
    os.makedirs(os.path.join(DATA, a.split), exist_ok=True)

    def flush():
        if not buf:
            return
        p = os.path.join(DATA, a.split, f"shard_{a.shard:03d}_{len(shard_files):04d}.npz")
        h = shard_write(buf, p)
        shard_files.append(dict(path=os.path.basename(p), n=len(buf), sha256=h))
        print(f"    -> {os.path.basename(p)}  n={len(buf)}  sha256={h[:16]}...")
        buf.clear()

    attempt = 0
    while sum(quota.values()) > 0 and attempt < n * 6:
        br = next(b for b in FAM if quota[b] > 0)
        attempt += 1
        r = make_sample(files[int(rng.integers(0, len(files)))], rng, br, tex_cats)
        if r is None:
            continue
        quota[br] -= 1
        meta_row = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in r.items() if not hasattr(v, "shape")}
        out.append(meta_row)
        buf.append(r)
        if len(panels) < 24:
            panels.append(r)
        if len(buf) >= a.shard_size:
            flush()
        if a.pilot:
            print(f"  [{len(out):3d}] {r['id'][:24]:24s} {br:8s} bg={r['bg']:5s} T={r['T']:4.1f} "
                  f"dead={r['dead']*100:5.1f}% kap={r['kappa50']:.4f} "
                  f"rho={'--' if r['rho'] is None else round(r['rho'],2):>5} {r['cls']}")
    if not a.pilot:
        flush()
    if not out:
        raise SystemExit("no samples")
    json.dump(out, open(os.path.join(RES, f"w3_pilot_{a.split}.json"), "w"), indent=1)
    if not a.pilot:
        man = dict(split=a.split, shard=a.shard, n_samples=len(out),
                   blocks=[int(b) for b in SPLIT_BLOCKS[a.split]],
                   buffer=BUFFER, block_size=BLOCK,
                   n_files=len(files), shards=shard_files,
                   dtd_categories=tex_cats,
                   generated_at="2026-11-16")
        mp = os.path.join(DATA, a.split, f"manifest_{a.shard:03d}.json")
        json.dump(man, open(mp, "w"), indent=1)
        print(f"manifest -> {mp}")
    from collections import Counter
    print(f"\nsamples = {len(out)}")
    print("branches:", dict(Counter(o["branch"] for o in out)))
    print("classes :", dict(Counter(o["cls"] for o in out)))
    print("bg modes:", dict(Counter(o["bg"] for o in out)))
    cs = [o for o in out if o["rho"] is not None]
    if cs:
        print(f"Tier-1 certified: {sum(o['cert'] for o in cs)}/{len(cs)} "
              f"({100*np.mean([o['cert'] for o in cs]):.1f}%)")
        rho = np.array([o["rho"] for o in cs])
        print(f"rho: P50={np.median(rho):.2f}  frac(rho>1)={np.mean(rho>1)*100:.1f}%")
    cov = [o["dead"] for o in out if o["branch"] != "degraded"]
    if cov:
        print(f"coverage: P5={np.percentile(cov,5)*100:.1f}% P50={np.median(cov)*100:.1f}% "
              f"P95={np.percentile(cov,95)*100:.1f}%")
    kk = np.array([o["kappa50"] for o in out])
    print(f"kappa P50/P95 over samples: {np.median(kk):.4f} / {np.percentile(kk,95):.4f}")
    if panels:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ncol = 6; nrow = int(np.ceil(len(panels) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.3 * nrow))
        for ax, r in zip(np.ravel(axes), panels):
            ax.imshow(r["I"], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{r['branch']} {r['bg']} T={r['T']:.1f}\nd={r['dead']*100:.1f}% "
                         f"{r['cls']} rho={'-' if r['rho'] is None else round(r['rho'],2)}", fontsize=6.5)
            ax.set_xticks([]); ax.set_yticks([])
        for ax in np.ravel(axes)[len(panels):]:
            ax.axis("off")
        fig.suptitle(f"W3 v3 pilot ({a.split}) -- weak DTD texture, 3 bg types, T aug", fontsize=10)
        fig.tight_layout()
        p = os.path.join(RES, f"w3_v3_{a.split}_24panels.png")
        fig.savefig(p, dpi=105, bbox_inches="tight"); print("saved", p)
