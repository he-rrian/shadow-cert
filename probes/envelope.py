"""P2 probe 2 -- the curve that decides the paper.

  dead zone = TOTAL OCCLUSION (the only degradation that destroys phase, see physics.py part C)
  sweep the equivalent hole diameter d and measure
     Identity  : FTP + unwrap on the degraded fringe
     envelope  : GT outside the hole, harmonic extension inside  (boundary-information limit)
     phaseInp  : FTP + unwrap, then harmonic fill inside
  plus the 1-D geometric bound kappa*d^2/8 for comparison.
Saves results/envelope.npy  (rows: area, d, id_fail, id_rmse, env_fail, env_rmse, env_p90, pinp, geo)
Run:  python envelope.py
"""
import os
import numpy as np, scipy.io as sio, cv2
from skimage.restoration import unwrap_phase

P = r"F:\paper\original_phase"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
IDS = [1, 127, 253, 379, 505, 631, 757, 876, 1002, 1128, 1800, 2600]

def load_phi(i):
    return np.asarray(sio.loadmat(os.path.join(P, f"{i}.mat"))["fai"], np.float64)

def bandlimit(phi, f0, frac=0.45):
    gy, gx = np.gradient(phi); g = np.hypot(gx, gy) / (2 * np.pi)
    s = min(1.0, frac * f0 / max(float(np.percentile(g, 99.9)), 1e-9))
    return phi * s, s

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

def harmonic(img, hole, iters=2000):
    z = img.astype(np.float32).copy()
    z[hole] = np.median(img[~hole]); m = hole.astype(np.float32)
    k = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]], np.float32)
    for _ in range(iters):
        sm = cv2.filter2D(z, -1, k, borderType=cv2.BORDER_REPLICATE)
        z = z * (1 - m) + sm * m
    return z.astype(np.float64)

def render(phi, f0=1 / 8, seed=0):
    H, W = phi.shape
    rng = np.random.default_rng(seed)
    X = np.arange(W)[None, :]
    I = 0.04 + 0.82 * (0.50 + 0.30 * np.cos(2 * np.pi * f0 * X + phi)) + 0.010 * rng.standard_normal(phi.shape)
    return np.clip(np.round(np.clip(I, 0, 1) * 255), 0, 255).astype(np.uint8)

def stats(est, gt, mask):
    d = est - gt
    d = d - np.median(d[~mask])
    e = np.abs(d[mask])
    return float((e > np.pi).mean()), float(np.sqrt((e ** 2).mean())), float(np.percentile(e, 90))

rows = []
print(f"{'area%':>6s}{'d':>6s} | {'Id fail':>8s}{'Id RMSE':>9s} | {'Env fail':>9s}{'Env RMSE':>9s}{'Env P90':>9s} | "
      f"{'PInp':>7s} | {'k d^2/8':>8s}")
for area in (0.01, 0.02, 0.03, 0.05, 0.085, 0.12, 0.20, 0.30, 0.40):
    acc = []
    for i in IDS:
        phi, _ = bandlimit(load_phi(i), 1 / 8)
        H, W = phi.shape
        h = int(np.sqrt(area * H * W / 1.4)); w = int(h * 1.4)
        r = np.zeros((H, W), bool)
        r[(H - h) // 2:(H - h) // 2 + h, (W - w) // 2:(W - w) // 2 + w] = True
        I8 = render(phi, seed=i); I8[r] = 0
        psi = ftp_unwrap(I8, 1 / 8)
        a = stats(psi, phi, r); b = stats(harmonic(phi, r), phi, r); c = stats(harmonic(psi, r), phi, r)
        lap = np.abs(cv2.Laplacian(phi.astype(np.float32), cv2.CV_32F))
        ring = cv2.dilate(r.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & (~r)
        kappa = float(lap[ring].mean()); d = 2 * np.sqrt(r.sum() / np.pi)
        acc.append([a[0], a[1], b[0], b[1], b[2], c[1], kappa * d * d / 8.0, d])
    M = np.median(np.array(acc), 0)
    rows.append([area, M[7], M[0], M[1], M[2], M[3], M[4], M[5], M[6]])
    print(f"{area*100:6.1f}{M[7]:6.1f} | {M[0]:8.3f}{M[1]:9.2f} | {M[2]:9.3f}{M[3]:9.3f}{M[4]:9.3f} | "
          f"{M[5]:7.2f} | {M[6]:8.2f}")

arr = np.array(rows)
np.save(os.path.join(OUT, "envelope.npy"), arr)
print("\nratio kappa*d^2/8 / envelope-RMSE :",
      np.round(arr[:, 8] / np.maximum(arr[:, 5], 1e-9), 2))
print("median ratio = %.1f  -> 2-D constant is ~ kappa*d^2/%.0f"
      % (np.median(arr[:, 8] / np.maximum(arr[:, 5], 1e-9)), 8 * np.median(arr[:, 8] / np.maximum(arr[:, 5], 1e-9))))
print("saved", os.path.join(OUT, "envelope.npy"))
