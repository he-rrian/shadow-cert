"""Probe A -- single-frame boundary phase error delta_g as a function of the distance
to a shadow edge, for two demodulation front-ends:

    (1) standard global FTP      : bandpass over the WHOLE image (kernel spans the shadow,
                                   so the penumbra mixes zero-modulation pixels in)
    (2) masked local sinusoid LSQ: fit  I = A + u cos(2*pi*f*x) + v sin(2*pi*f*x)
                                   locally, using only CLEAN pixels (mask-weighted),
                                   so the fit never convolves across the shadow edge

Injection: a circular occluder with a 6 px penumbra is written into a real SynthFringe frame as
        I_new = A + (I0 - A) * s(x)          (erases the modulated component, keeps the pedestal)
which is exactly "the surface receives no projector light".

Output: delta_g P50/P95 vs distance to the shadow edge, for windows 5/7/9/11.
Run: python probeA_edge.py
"""
import os, json
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "probes", "results"); os.makedirs(RES, exist_ok=True)
GT = os.path.join(RES, "gt")
BASE = r"F:\paper\SynthFringe"

def im(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

def load_seq(sub, sid):
    p = os.path.join(BASE, sub, sid)
    fs = sorted(x for x in os.listdir(p) if x.endswith(".bmp"))
    out = []
    for f in fs:
        a = im(os.path.join(p, f))
        if a.ndim == 3:
            a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2GRAY)
        out.append(a.astype(np.float64))
    return np.stack(out)

def ftp(psi_img, f0):
    """standard global FTP"""
    H, W = psi_img.shape
    F = np.fft.fftshift(np.fft.fft2(psi_img - psi_img.mean()))
    u = np.fft.fftshift(np.fft.fftfreq(W)); v = np.fft.fftshift(np.fft.fftfreq(H))
    U, V = np.meshgrid(u, v)
    w = 0.5 * abs(f0)
    bm = np.exp(-((U - f0) ** 2) / (2 * (w / 2) ** 2)) * (np.abs(U - f0) < 2 * w)
    Z = np.fft.ifft2(np.fft.ifftshift(F * bm))
    Z = Z * np.exp(-1j * 2 * np.pi * f0 * np.arange(W))[None, :]
    return np.angle(Z)

def masked_lsq(I, mask, f0, win):
    """local 3-parameter sinusoid fit, only CLEAN pixels contribute"""
    H, W = I.shape
    xs = np.arange(W, dtype=np.float64)[None, :].repeat(H, 0)
    c = np.cos(2 * np.pi * f0 * xs); s = np.sin(2 * np.pi * f0 * xs)
    m = mask.astype(np.float64)
    bf = lambda a: cv2.boxFilter(a, -1, (win, win), normalize=False, borderType=cv2.BORDER_REPLICATE)
    S = bf(m); Sc = bf(m * c); Ss = bf(m * s)
    Scc = bf(m * c * c); Scs = bf(m * c * s); Sss = bf(m * s * s)
    S1 = bf(m * I); Sc1 = bf(m * I * c); Ss1 = bf(m * I * s)
    A3 = np.empty((H, W, 3, 3))
    A3[..., 0, 0] = S;   A3[..., 0, 1] = Sc;  A3[..., 0, 2] = Ss
    A3[..., 1, 0] = Sc;  A3[..., 1, 1] = Scc; A3[..., 1, 2] = Scs
    A3[..., 2, 0] = Ss;  A3[..., 2, 1] = Scs; A3[..., 2, 2] = Sss
    b = np.stack([S1, Sc1, Ss1], -1)
    A3 = A3 + np.eye(3)[None, None] * 1e-9
    sol = np.linalg.solve(A3, b[..., None])[..., 0]
    u_, v_ = sol[..., 1], sol[..., 2]
    phi = np.arctan2(-v_, u_)
    amp = np.hypot(u_, v_)
    return phi, amp

def make_shadow(I, A, cy, cx, R, pen=3.0):
    H, W = I.shape
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(yy - cy, xx - cx)
    t = np.clip((r - (R - pen)) / (2 * pen), 0, 1)
    sm = t * t * (3 - 2 * t)                       # C1 smoothstep penumbra, width 2*pen
    I2 = A + (I - A) * sm
    shadow = sm < 0.99
    return I2, shadow, sm

def analyse(sid, R=26, pen=3.0):
    d = np.load(os.path.join(GT, f"{sid}.npz"))
    phi_gt = d["phi64"].astype(np.float64)
    roi = d["m_gt"]
    I15 = load_seq("64_15", sid)
    A = I15.mean(0)
    I0 = I15[0]
    # ---- fringe frequency from the 2-D spectrum peak (excluding DC)
    F2 = np.abs(np.fft.fftshift(np.fft.fft2(I0 - I0.mean())))
    H, W = I0.shape
    cy0, cx0 = H // 2, W // 2
    F2[cy0 - 6:cy0 + 7, cx0 - 6:cx0 + 7] = 0
    iy, ix = np.unravel_index(np.argmax(F2), F2.shape)
    f0 = (ix - cx0) / W            # SIGNED: the analytic signal is conjugated when f0 < 0
    if abs(f0) < 0.02:
        f0 = 0.1
    # put the occluder on a high-modulation, interior patch
    yy, xx = np.mgrid[0:phi_gt.shape[0], 0:phi_gt.shape[1]]
    dist_in = cv2.distanceTransform(roi.astype(np.uint8), cv2.DIST_L2, 5)
    cand = np.argwhere((dist_in > R + 12) & roi)
    if len(cand) == 0:
        return None
    cy, cx = cand[np.argmax(dist_in[tuple(cand.T)])]
    I2, shad, sm = make_shadow(I0, A, int(cy), int(cx), R, pen)
    clean = (~shad) & roi
    out = dict(sid=sid, f0=float(f0), R=R, pen=pen, clean=float(clean.mean()))
    # distance of clean pixels to the shadow boundary
    dsh = cv2.distanceTransform((~shad).astype(np.uint8), cv2.DIST_L2, 5)
    # --- method 1: FTP
    p1 = ftp(I2, f0)
    # --- method 2..: masked LSQ
    res = {"FTP": p1}
    for win in (5, 7, 9, 11):
        p2, _ = masked_lsq(I2, clean, f0, win)
        res[f"LSQ{win}"] = p2
    tbl = {}
    for name, ph in res.items():
        # ph is the CARRIER-REMOVED phase; the GT is the TOTAL phase.
        # Restore the carrier before comparing (otherwise the difference is a 2*pi*f0*x ramp).
        ph_tot = ph + 2 * np.pi * f0 * np.arange(ph.shape[1])[None, :]
        r = np.angle(np.exp(1j * (ph_tot - phi_gt)))
        far = clean & (dsh > 12)
        off = float(np.angle(np.mean(np.exp(1j * r[far])))) if far.sum() > 50 else 0.0
        e = np.abs(np.angle(np.exp(1j * (r - off))))
        rowd = {}
        for dd in range(1, 11):
            m = clean & (dsh >= dd - 0.5) & (dsh < dd + 0.5)
            if m.sum() > 30:
                rowd[dd] = (float(np.median(e[m])), float(np.percentile(e[m], 95)))
        tbl[name] = rowd
    out["table"] = tbl
    return out

if __name__ == "__main__":
    sids = ["00000", "00360", "01500"]
    allrows = []
    for sid in sids:
        try:
            r = analyse(sid)
        except Exception as ex:
            print(f"{sid}: ERR {type(ex).__name__} {ex}"); continue
        if r is None:
            print(f"{sid}: no room for the occluder"); continue
        allrows.append(r)
        print(f"\n=== {sid}: f0={r['f0']:.4f} cyc/px (T={1/r['f0']:.1f} px) ===")
        print("  d(px) | " + " | ".join(f"{k:>13s}" for k in r["table"]))
        for dd in range(1, 11):
            cells = []
            for k, t in r["table"].items():
                cells.append(f"{t[dd][0]:.3f}/{t[dd][1]:.3f}" if dd in t else "     -      ")
            print(f"  {dd:5d} | " + " | ".join(f"{c:>13s}" for c in cells))
    json.dump(allrows, open(os.path.join(RES, "probeA.json"), "w"), indent=1)
    print("\nsaved", os.path.join(RES, "probeA.json"))
