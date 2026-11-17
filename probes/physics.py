"""P2 probe 1 -- physics of the phase data and of the degradation operators.

Answers (numbers reproduced in doc/题目2_可行性复核与数据方案_v1.md):
  A. phase range / gradient statistics of original_phase          -> is f0=1/16 admissible?
  B. clean-fringe demodulation error, raw vs band-limited          -> is the rendering physical?
  C. which degradation actually destroys phase: saturation / shadow / total occlusion
Run:  python physics.py
"""
import os, json
import numpy as np, scipy.io as sio, cv2
from skimage.restoration import unwrap_phase

P = r"F:\paper\original_phase"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)

# ---------------------------------------------------------------- helpers
def load_phi(i):
    return np.asarray(sio.loadmat(os.path.join(P, f"{i}.mat"))["fai"], np.float64)

def grad_cycles(phi):
    gy, gx = np.gradient(phi)
    return np.hypot(gx, gy) / (2 * np.pi)

def bandlimit(phi, f0, frac=0.45):
    """scale phi so that the local instantaneous frequency stays inside the analysis band"""
    g = grad_cycles(phi)
    s = min(1.0, frac * f0 / max(float(np.percentile(g, 99.9)), 1e-9))
    return phi * s, s

def render(phi, f0=1 / 8, seed=0, black=None, A_amb=0.04, R=0.82, a=0.50, b=0.30, noise=0.010):
    H, W = phi.shape
    X = np.arange(W)[None, :]
    rng = np.random.default_rng(seed)
    I = A_amb + R * (a + b * np.cos(2 * np.pi * f0 * X + phi)) + noise * rng.standard_normal(phi.shape)
    I = np.clip(I, 0, 1)
    if black is not None:
        I[black] = 0.0
    return np.clip(np.round(I * 255), 0, 255).astype(np.uint8)

def ftp_unwrap(I8, f0, band=0.5):
    I = I8.astype(np.float64) / 255.0
    H, W = I.shape
    F = np.fft.fftshift(np.fft.fft2(I - I.mean()))
    u = np.fft.fftshift(np.fft.fftfreq(W)); v = np.fft.fftshift(np.fft.fftfreq(H))
    U, V = np.meshgrid(u, v)
    w = band * f0
    bm = np.exp(-(np.abs(U - f0)) ** 2 / (2 * (w / 2.0) ** 2)) * (np.abs(U - f0) < 2 * w)
    Z = np.fft.ifft2(np.fft.ifftshift(F * bm))
    Z = Z * np.exp(-1j * 2 * np.pi * f0 * np.arange(W))[None, :]
    return unwrap_phase(np.angle(Z))

def harmonic(img, hole, iters=1500):
    z = img.astype(np.float32).copy()
    z[hole] = np.median(img[~hole]); m = hole.astype(np.float32)
    k = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]], np.float32)
    for _ in range(iters):
        sm = cv2.filter2D(z, -1, k, borderType=cv2.BORDER_REPLICATE)
        z = z * (1 - m) + sm * m
    return z.astype(np.float64)

def err(est, gt, mask):
    d = est - gt
    d = d - np.median(d[~mask])                       # single frame: global piston unknown
    e = np.abs(d[mask])
    return float(np.sqrt((e ** 2).mean())), float((e > np.pi).mean())

# ---------------------------------------------------------------- A
print("=" * 78, "\nA. phase statistics (original_phase, 10k files)")
allf = sorted(os.listdir(P))
idx = np.linspace(0, len(allf) - 1, 200).astype(int)
maxs, g50, g999 = [], [], []
for i in idx:
    ph = load_phi(allf[i].replace(".mat", ""))
    g = grad_cycles(ph)
    maxs.append(ph.max()); g50.append(np.median(g)); g999.append(float(np.percentile(g, 99.9)))
maxs, g50, g999 = map(np.array, (maxs, g50, g999))
print(f"  objects sampled = {len(maxs)}   max phase: P50={np.median(maxs):.0f} P90={np.percentile(maxs,90):.0f} max={maxs.max():.0f}")
print(f"  |grad|/2pi  P50={np.median(g50):.4f}  per-object P99.9 median={np.median(g999):.4f}")
for f0 in (1/16, 1/8, 1/6):
    print(f"  f0={f0:.4f}: objects violating P99.9>f0 = {np.mean(g999 > f0)*100:5.1f}%")

# ---------------------------------------------------------------- B
print("=" * 78, "\nB. clean-fringe demodulation: raw vs band-limited")
ids = [1, 127, 253, 379, 505, 631, 757, 876, 1002, 1128]
rows = []
for i in ids:
    raw = load_phi(i)
    r = {"id": i, "maxphi": float(raw.max())}
    for f0 in (1 / 16, 1 / 8, 1 / 6):
        e_raw = ftp_unwrap(render(raw, f0, seed=i), f0)
        rm_raw = float(np.sqrt((np.abs(e_raw - raw - np.median(e_raw - raw)) ** 2).mean()))
        bl, s = bandlimit(raw, f0)
        e_bl = ftp_unwrap(render(bl, f0, seed=i), f0)
        rm_bl = float(np.sqrt((np.abs(e_bl - bl - np.median(e_bl - bl)) ** 2).mean()))
        r[f"raw_{f0:.4f}"] = rm_raw; r[f"bl_{f0:.4f}"] = rm_bl; r[f"scale_{f0:.4f}"] = s
    rows.append(r)
for f0 in (1 / 16, 1 / 8, 1 / 6):
    print(f"  f0={f0:.4f}: raw RMSE median={np.median([r[f'raw_{f0:.4f}'] for r in rows]):8.2f} rad"
          f"   band-limited={np.median([r[f'bl_{f0:.4f}'] for r in rows]):.3f} rad")
json.dump(rows, open(os.path.join(OUT, "physics_render.json"), "w"), indent=1)

# ---------------------------------------------------------------- C
print("=" * 78, "\nC. which degradation destroys phase?  (Identity = FTP+unwrap)")
def degrade(phi, mode, d, seed):
    H, W = phi.shape
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    f0 = 1 / 8
    cy, cx = int(rng.integers(90, 167)), int(rng.integers(90, 167))
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    I = 0.04 + 0.82 * (0.50 + 0.30 * np.cos(2 * np.pi * f0 * xx + phi))
    if mode == "sat":                       # additive specular -> sensor clip
        I = I + (0.98 - (0.04 + 0.82 * 0.80) + 0.55) * np.exp(-r2 / (2 * (d / 2.6) ** 2))
    elif mode == "shadow":                  # multiplicative attenuation (phase preserving!)
        I = 0.04 + 0.82 * (1 - 0.985 * np.exp(-r2 / (2 * (d / 3.2) ** 2))) * (0.50 + 0.30 * np.cos(2 * np.pi * f0 * xx + phi))
    I = np.clip(I + 0.010 * rng.standard_normal(I.shape), 0, 1)
    I8 = np.clip(np.round(I * 255), 0, 255).astype(np.uint8)
    if mode == "occl":
        I8[r2 <= (d / 2) ** 2] = 0          # total occlusion: no modulated light at all
    mask = (I8 == 0) if mode == "occl" else ((I >= 0.98) if mode == "sat" else (0.82 * (1 - 0.985 * np.exp(-r2 / (2 * (d / 3.2) ** 2))) * 0.30 < 0.03))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    return I8, mask

print(f"  {'mode':7s}{'d':>4s} {'cov':>6s} {'2pi-fail':>9s} {'RMSE':>7s} {'envelope':>9s}")
mech = []
for mode in ("sat", "shadow", "occl"):
    for d in (32, 64, 96, 128):
        acc = []
        for i in (253, 505, 631, 1002, 1128):
            ph, _ = bandlimit(load_phi(i), 1 / 8)
            I8, mask = degrade(ph, mode, d, seed=i * 37 + d)
            if mask.mean() < 0.002 or mask.mean() > 0.5:
                continue
            psi = ftp_unwrap(I8, 1 / 8)
            rm, fa = err(psi, ph, mask)
            ro, fo = err(harmonic(ph, mask), ph, mask)
            acc.append((mask.mean(), rm, fa, ro))
        if not acc:
            continue
        m, rm, fa, ro = np.median(np.array(acc), 0)
        print(f"  {mode:7s}{d:4d} {m:6.3f} {fa:9.3f} {rm:7.2f} {ro:9.3f}")
        mech.append(dict(mode=mode, d=d, cov=m, id_rmse=rm, id_fail=fa, env_rmse=ro))
json.dump(mech, open(os.path.join(OUT, "physics_mechanism.json"), "w"), indent=1)
print("\nsaved to", OUT)
