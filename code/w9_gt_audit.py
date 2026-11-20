"""w9_gt_audit.py -- GT audit for the 11 real scenes before G7.

Tier A (independent reference): scene_01..06, Train, 实测1
Tier B (self-consistent 4-step): 实测2, 实验3, 实验4

Locked scope (2026-11-20):
  * Train's gt15_rad.npy is a deviation phase -> allow a deterministic fitted carrier
    plane in the cross-check (median residual 0.0016 rad); do not modify the data.
  * 实测1 stays Tier A but is a FAIL (mat deviation+plane 0.116/0.506 > 0.1/0.3);
    it is qualitative-only and never enters the G7 P-pixel statistics.  A failed
    validation is NOT the same as "no validation", so it is not laundered into Tier B.
  * scene_02/03/06 are GT-insufficient (old skimage.unwrap_phase 2*pi branch, badmask
    from the same old pipeline).  No automatic per-component alignment; they stay out
    of pixel-level G7 until the phase and mask gates in `reaudit_prereg` pass.
  * first=~fifth is N/A (the 5th frame is not a 2*pi repeat; fringe.m uses 4 frames).
  * Tier B canonicalises to +x and to the carrier sign, and reports raw AND canonical
    direction checks.

Tier B enters G7 only if the pre-registered self-consistency suite passes:
  1. wrap closure P95 < 0.05 rad
  2. 4/15-step model residual median < 0.10
  3. fringe-direction check (canonical): sign(median d phi/dx) == sign(carrier fx)
  4. unwrap continuity: fraction of adjacent valid pairs with |d phi| > pi < 0.5%
  5. valid mask fraction > 10%

N_eff rule (locked 2026-11-20):
  n_fail_A = Tier A scenes not pixel-evaluable (GT-insufficient or Tier A FAIL)
  n_fail_B = Tier B scenes failing the suite
  N_eff = 11 - n_fail_A - n_fail_B
  if N_eff < 10: G7 = insufficient
  else: handled >= ceil(8*N_eff/11), n_P >= 5, P_pass >= ceil(0.8*n_P)
  "P but GT-insufficient" scenes are listed separately; they enter neither the
  P_pass numerator nor the denominator.

Run:
  python code/w9_gt_audit.py
  python code/w9_gt_audit.py --scene scene_01   # debug one scene
"""
import os, sys, json, math, argparse, heapq
import numpy as np, cv2

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "probes", "results")
P1 = r"F:\paper\1\phase"
REAL = r"F:\paper\2\model\real\captures\proc"

# Pre-registered audit thresholds (written before looking at G7).
TH = dict(
    valid_frac_min=0.10,
    wrap_p95_max=0.05,
    model_resid_med_max=0.10,
    dir_sign_match=True,
    unwrap_jump_frac_max=0.005,
    cross_med_max=0.05,
    cross_p95_max=0.20,
    cross_med_max_tierA_real=0.10,   # 实测1 vs fourstep mat
    cross_p95_max_tierA_real=0.30,
)

SCENES = [
    # name, tier, path, kind, gt
    ("scene_01", "A", os.path.join(REAL, "scene_01"), "4step_proc", "gt_phase.npy"),
    ("scene_02", "A", os.path.join(REAL, "scene_02"), "4step_proc", "gt_phase.npy"),
    ("scene_03", "A", os.path.join(REAL, "scene_03"), "4step_proc", "gt_phase.npy"),
    ("scene_04", "A", os.path.join(REAL, "scene_04"), "4step_proc", "gt_phase.npy"),
    ("scene_05", "A", os.path.join(REAL, "scene_05"), "4step_proc", "gt_phase.npy"),
    ("scene_06", "A", os.path.join(REAL, "scene_06"), "4step_proc", "gt_phase.npy"),
    ("Train", "A", os.path.join(P1, "Train"), "15step", "gt15_rad.npy"),
    ("实测1", "A", os.path.join(P1, "实测1"), "4step_bmp", "fpp_shice_1_fourstep_purphase.mat"),
    ("实测2", "B", os.path.join(P1, "实测2"), "4step_bmp", None),
    ("实验3", "B", os.path.join(P1, "实验3"), "4step_crop", None),
    ("实验4", "B", os.path.join(P1, "实验4"), "4step_crop", None),
]


# ------------------------------------------------------------------ io
def im(p):
    b = np.fromfile(p, dtype=np.uint8)
    a = cv2.imdecode(b, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(p)
    if a.ndim == 3:
        a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2GRAY)
    return a.astype(np.float32)


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------------ unwrap
def unwrap_quality(phi_w, quality, valid):
    """Quality-guided BFS: highest-quality seed first, then flood by quality."""
    H, W = phi_w.shape
    phi = np.full((H, W), np.nan, np.float32)
    ys, xs = np.where(valid)
    if len(ys) == 0:
        return phi
    order = np.argsort(-quality[ys, xs])
    visited = np.zeros((H, W), bool)
    nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for oi in order:
        y0, x0 = int(ys[oi]), int(xs[oi])
        if visited[y0, x0]:
            continue
        phi[y0, x0] = phi_w[y0, x0]; visited[y0, x0] = True
        heap = []
        for dy, dx in nbrs:
            y, x = y0 + dy, x0 + dx
            if 0 <= y < H and 0 <= x < W and valid[y, x] and not visited[y, x]:
                heapq.heappush(heap, (-quality[y, x], y, x, y0, x0))
        while heap:
            _, y, x, py, px = heapq.heappop(heap)
            if visited[y, x]:
                continue
            phi[y, x] = phi[py, px] + wrap(phi_w[y, x] - phi_w[py, px])
            visited[y, x] = True
            for dy, dx in nbrs:
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and valid[yy, xx] and not visited[yy, xx]:
                    heapq.heappush(heap, (-quality[yy, xx], yy, xx, y, x))
    return phi


# ------------------------------------------------------------------ phase helpers
def phase_model_residual(frames, phi, valid, nstep):
    """Median relative residual of I_k = A + B cos(phi + 2*pi*k/N)."""
    N = nstep
    d = 2 * np.pi * np.arange(N) / N
    X = np.stack([np.ones(N), np.cos(d)], 1)  # placeholder
    # basis: 1, cos(d), sin(d); use cos(phi)cos(d) - sin(phi)sin(d)
    A = np.stack([np.ones(N), np.cos(d), -np.sin(d)], 1)  # N x 3
    pinv = np.linalg.pinv(A)                                # 3 x N
    ys, xs = np.where(valid)
    M = len(ys)
    if M == 0:
        return float("nan"), 0
    Y = np.stack([frames[k][ys, xs] for k in range(N)], 0)  # N x M
    coef = pinv @ Y                                         # 3 x M
    B = np.hypot(coef[1], coef[2]) + 1e-6
    rec = A @ coef
    res = np.sqrt(np.mean((Y - rec) ** 2, 0)) / B
    return float(np.median(res)), M


def valid_from_mod(B, frac=0.10):
    return B > frac * np.percentile(B, 95)


def wrap_closure(phi, phi_w, valid):
    if not valid.any():
        return float("nan")
    return float(np.percentile(np.abs(wrap(phi[valid] - phi_w[valid])), 95))


def unwrap_jump_frac(phi, valid):
    if valid.sum() < 10:
        return float("nan")
    f = 0.0; n = 0
    for axis in (0, 1):
        a = np.diff(phi, axis=axis)
        m = valid[:, :-1] & valid[:, 1:] if axis == 1 else valid[:-1, :] & valid[1:, :]
        if m.any():
            f += float((np.abs(a[m]) > np.pi).sum())
            n += int(m.sum())
    return f / max(n, 1)


def signed_carrier_fx(I, valid):
    """Signed dominant fx from the FFT (no positive-fx forcing)."""
    x = I.copy(); x[~valid] = x[valid].mean() if valid.any() else 0.0
    F = np.fft.fftshift(np.fft.fft2(x - x.mean()))
    H, W = I.shape
    u = np.fft.fftshift(np.fft.fftfreq(W)); v = np.fft.fftshift(np.fft.fftfreq(H))
    U, V = np.meshgrid(u, v)
    band = (np.hypot(U, V) >= 1 / 18) & (np.hypot(U, V) <= 1 / 5.5)
    iy, ix = np.unravel_index(np.argmax(np.where(band, np.abs(F), 0)), F.shape)
    return float(U[iy, ix])


def median_dx(phi, valid):
    if valid.sum() < 10:
        return float("nan")
    d = np.diff(phi, axis=1)
    m = valid[:, :-1] & valid[:, 1:]
    return float(np.median(d[m])) if m.any() else float("nan")


def align(ref, test, valid):
    """Try sign + 2*pi*k + constant offset.

    The piston can be hundreds of radians (unwrapped GT vs a fresh 4-step reference),
    and up to half the pixels can be on a different 2*pi branch.  The median of
    (ref - test) is therefore NOT a reliable piston estimate; use a coarse k scan on a
    subsample, then refine on the full valid mask.
    """
    ys, xs = np.where(valid)
    m = len(ys)
    sub = np.linspace(0, m - 1, min(5000, m)).astype(int)
    ysub, xsub = ys[sub], xs[sub]
    ks = np.arange(-160, 161)
    best = None
    for s in (1, -1):
        base = ref - s * test
        med = np.empty(len(ks), float)
        for j, k in enumerate(ks):
            d = base[ysub, xsub] - 2 * np.pi * k
            c = np.median(d)
            med[j] = np.median(np.abs(d - c))
        k0 = int(ks[int(np.argmin(med))])
        for k in range(k0 - 3, k0 + 4):
            d = base - 2 * np.pi * k
            c = float(np.median(d[valid]))
            r = np.abs(d - c)[valid]
            mm = float(np.median(r))
            if best is None or mm < best["median"]:
                best = dict(sign=s, k=int(k), offset=c, median=mm,
                            p95=float(np.percentile(r, 95)), max=float(np.max(r)),
                            mean=float(np.mean(r)), model="offset")
    return best


def align_plane(ref, test, valid):
    """Try sign + a fitted carrier plane (a + bx*x + by*y); for deviation-phase GT."""
    ys, xs = np.where(valid)
    A = np.stack([np.ones(len(ys)), xs.astype(np.float64), ys.astype(np.float64)], 1)
    best = None
    for s in (1, -1):
        d = ref - s * test
        coef, *_ = np.linalg.lstsq(A, d[valid].astype(np.float64), rcond=None)
        r = d[valid] - A @ coef
        med = float(np.median(np.abs(r)))
        if best is None or med < best["median"]:
            best = dict(sign=s, a=float(coef[0]), bx=float(coef[1]), by=float(coef[2]),
                        median=med, p95=float(np.percentile(np.abs(r), 95)),
                        max=float(np.max(np.abs(r))), mean=float(np.mean(np.abs(r))),
                        model="deviation+plane")
    return best


def first_fifth(frames, valid):
    if len(frames) < 5:
        return None
    a = frames[0]; b = frames[4]
    scale = np.percentile(a[valid], 95) - np.percentile(a[valid], 5) if valid.any() else 0.0
    if scale <= 0:
        return None
    return float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2)) / scale)


# ------------------------------------------------------------------ scene loaders
def load_frames_4step_proc(path):
    return [im(os.path.join(path, f"capture_{k}.png")) for k in range(1, 5)]


def load_frames_4step_bmp(path):
    fs = sorted(f for f in os.listdir(path) if f.lower().endswith(".bmp") and "shice" in f and "_purphase" not in f)
    fs = [f for f in fs if not f.endswith("wraptedphase.bmp")]
    return [im(os.path.join(path, f)) for f in fs], fs


def load_frames_4step_crop(path):
    fs = sorted(f for f in os.listdir(path) if f.startswith("cropped_") and f.lower().endswith((".jpg", ".png")))
    return [im(os.path.join(path, f)) for f in fs[:4]], fs


def load_train(path):
    fs = sorted(f for f in os.listdir(path) if f.endswith(".bmp") and f[:2].isdigit())
    return [im(os.path.join(path, f)) for f in fs]


def scene_4step(frames):
    I1, I2, I3, I4 = frames[:4]
    phi_w = np.arctan2(I4 - I2, I1 - I3)
    B = 0.5 * np.hypot(I4 - I2, I1 - I3)
    return phi_w.astype(np.float32), B


def scene_15step(frames):
    I = np.stack(frames, 0).astype(np.float32)
    N = I.shape[0]
    d = 2 * np.pi * np.arange(N) / N
    num = -(I * np.sin(d)[:, None, None]).sum(0)
    den = (I * np.cos(d)[:, None, None]).sum(0)
    phi_w = np.arctan2(num, den).astype(np.float32)
    B = (2.0 / N) * np.hypot(num, den)
    return phi_w, B


def audit_scene(name, tier, path, kind, gt):
    rec = dict(scene=name, tier=tier, path=path, kind=kind, gt=gt)
    if kind == "4step_proc":
        frames = load_frames_4step_proc(path); files = [f"capture_{k}.png" for k in range(1, 5)]
        phi_w, B = scene_4step(frames); nstep = 4
        valid = valid_from_mod(B, 0.05)
        ref = np.load(os.path.join(path, "gt_phase.npy")).astype(np.float32)
        ref_name = "gt_phase.npy"
        ref_valid = np.isfinite(ref) & valid
    elif kind == "15step":
        frames = load_train(path); files = ["00.bmp..14.bmp"]
        phi_w, B = scene_15step(frames); nstep = 15
        ref = np.load(os.path.join(path, "gt15_rad.npy")).astype(np.float32)
        ref_mask = np.load(os.path.join(path, "gt15_mask.npy")).astype(bool)
        valid = ref_mask & np.isfinite(ref)
        ref_name = "gt15_rad.npy + gt15_mask.npy"
        ref_valid = valid
    elif kind == "4step_bmp":
        frames, files = load_frames_4step_bmp(path); phi_w, B = scene_4step(frames); nstep = 4
        valid = valid_from_mod(B, 0.10)
        if gt and gt.endswith(".mat"):
            import scipy.io as sio
            ref = np.asarray(sio.loadmat(os.path.join(path, gt))["purePhase2"], np.float32)
            ref_name = gt
            ref_valid = np.isfinite(ref) & valid
        else:
            ref = None; ref_name = "self-consistency only"; ref_valid = None
    elif kind == "4step_crop":
        frames, files = load_frames_4step_crop(path); phi_w, B = scene_4step(frames); nstep = 4
        valid = valid_from_mod(B, 0.10)
        ref = None; ref_name = "self-consistency only"; ref_valid = None
    else:
        raise ValueError(kind)

    # ---- direction + unwrap.  Tier B is canonicalised to +x and to the carrier sign
    # before the direction check; Tier A is fixed by the GT cross-check instead.
    orig_fx = signed_carrier_fx(frames[0], valid)
    phi_raw = unwrap_quality(phi_w, B, valid)
    raw_mdx = median_dx(phi_raw, valid)
    raw_dir = bool(np.isfinite(orig_fx) and np.isfinite(raw_mdx) and np.sign(orig_fx) == np.sign(raw_mdx))
    if tier == "B":
        canon_fx = orig_fx
        if canon_fx < 0:
            frames = [f[:, ::-1].copy() for f in frames]
            phi_w = phi_w[:, ::-1].copy(); B = B[:, ::-1].copy(); valid = valid[:, ::-1].copy()
            canon_fx = -canon_fx
        phi = unwrap_quality(phi_w, B, valid)
        mdx = median_dx(phi, valid)
        if np.isfinite(mdx) and np.isfinite(canon_fx) and np.sign(mdx) != np.sign(canon_fx):
            phi_w = -phi_w; phi = -phi; mdx = -mdx
        canonical_dir = bool(np.isfinite(canon_fx) and np.isfinite(mdx) and np.sign(canon_fx) == np.sign(mdx))
        direction_ok = canonical_dir
        carrier_fx = canon_fx
    else:
        phi = phi_raw
        mdx = raw_mdx
        canonical_dir = raw_dir
        direction_ok = raw_dir
        carrier_fx = orig_fx

    rec.update(shape=list(phi_w.shape), n_frames=len(frames), files=files,
               valid_frac=float(valid.mean()), unit="rad",
               unwrapped=True,
               range=[float(np.nanmin(phi[valid])), float(np.nanmax(phi[valid]))] if valid.any() else [None, None],
               wrap_p95=wrap_closure(phi, phi_w, valid),
               jump_frac=unwrap_jump_frac(phi, valid),
               model_resid_med=phase_model_residual(frames, phi, valid, nstep)[0],
               first5=first_fifth(frames, valid),
               first5_applicable=False,
               first5_reason="5th frame is not a 2*pi repeat; fringe.m uses 4 frames; removed 2026-11-20",
               carrier_fx=float(carrier_fx),
               median_dphi_dx=float(mdx),
               median_dphi_dx_raw=float(raw_mdx),
               direction_raw=raw_dir,
               direction_canonical=canonical_dir,
               direction_ok=direction_ok)

    if ref is not None:
        g = align(ref, phi, ref_valid)
        p = align_plane(ref, phi, ref_valid)
        # 实测1's fourstep_purphase.mat is a deviation phase (sign=-1 + plane);
        # Train's gt15_rad is a deviation phase too.  The other GTs are total phase.
        use_plane = (name == "实测1") or (name == "Train" and (g["median"] > 0.5) and (p["median"] < TH["cross_med_max"]))
        a = p if use_plane else g
        rec.update(cross=dict(ref=ref_name, **a),
                   cross_alt=dict(ref=ref_name, **(g if use_plane else p)),
                   gt_convention=("deviation+plane" if use_plane else "total+offset"))
    else:
        rec["cross"] = None
        rec["cross_alt"] = None
        rec["gt_convention"] = "self-consistency only"

    # gate
    if tier == "A":
        med_max = TH["cross_med_max_tierA_real"] if name == "实测1" else TH["cross_med_max"]
        p95_max = TH["cross_p95_max_tierA_real"] if name == "实测1" else TH["cross_p95_max"]
        c = rec.get("cross")
        ok_cross = bool(c and c["median"] <= med_max and c["p95"] <= p95_max)
        ok_misc = (rec["valid_frac"] >= TH["valid_frac_min"]
                   and rec["wrap_p95"] <= TH["wrap_p95_max"])
        rec["pass"] = bool(ok_cross and ok_misc)
        rec["pixel_evaluable"] = bool(rec["pass"])
        rec["gt_status"] = "PASS" if rec["pass"] else "Tier A FAIL"
        if name in ("scene_02", "scene_03", "scene_06"):
            # Locked 2026-11-20: the old skimage.unwrap_phase GT and the fresh
            # quality-guided unwrap differ by 2*pi over a large connected region, and the
            # badmask is from the same old pipeline.  Do NOT auto-align per component;
            # keep these scenes out of pixel-level G7 until the masks are repaired.
            rec["pass"] = False
            rec["pixel_evaluable"] = False
            rec["gt_status"] = "GT-insufficient (old skimage 2pi branch; not pixel-evaluable)"
        if name == "实测1":
            rec["pass"] = False
            rec["pixel_evaluable"] = False
            rec["gt_status"] = ("Tier A FAIL (mat deviation+plane 0.116/0.506 > 0.1/0.3; "
                                "qualitative only)")
        rec["gate"] = dict(cross_med_max=med_max, cross_p95_max=p95_max,
                           wrap_p95_max=TH["wrap_p95_max"],
                           valid_frac_min=TH["valid_frac_min"],
                           ok_cross=ok_cross, ok_misc=ok_misc)
        # jump_frac is reported as a diagnostic, not a Tier-A gate.
    else:
        # first≈fifth is N/A (protocol correction 2026-11-20): the 5th frame is not a
        # 2*pi repeat and fringe.m uses only 4 frames.
        ok = (rec["valid_frac"] >= TH["valid_frac_min"]
              and rec["wrap_p95"] <= TH["wrap_p95_max"]
              and rec["model_resid_med"] <= TH["model_resid_med_max"]
              and rec["jump_frac"] <= TH["unwrap_jump_frac_max"]
              and rec["direction_ok"])
        rec["pass"] = bool(ok)
        rec["pixel_evaluable"] = bool(ok)
        rec["gt_status"] = "PASS" if ok else "Tier B FAIL"
        rec["gate"] = dict(valid_frac_min=TH["valid_frac_min"],
                           wrap_p95_max=TH["wrap_p95_max"],
                           model_resid_med_max=TH["model_resid_med_max"],
                           unwrap_jump_frac_max=TH["unwrap_jump_frac_max"],
                           direction_sign_match=TH["dir_sign_match"])
    return rec


# ------------------------------------------------------------------ report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="")
    ap.add_argument("--out", default=os.path.join(RES, "w9_gt_audit.json"))
    a = ap.parse_args()
    rows = []
    for name, tier, path, kind, gt in SCENES:
        if a.scene and name != a.scene:
            continue
        print(f"auditing {name} ({tier}, {kind}) ...", flush=True)
        rows.append(audit_scene(name, tier, path, kind, gt))

    n_fail_A = sum(1 for r in rows if r["tier"] == "A" and not r.get("pixel_evaluable", False))
    n_fail_B = sum(1 for r in rows if r["tier"] == "B" and not r["pass"])
    N_eff = len(rows) - n_fail_A - n_fail_B
    reaudit = dict(
        applies_to=["scene_02", "scene_03", "scene_06"],
        phase_gate=("per connected component k_c=round(median(Delta/(2*pi))); "
                    "residual=Delta-2*pi*k_c; P95(residual)<=0.1 rad AND "
                    "fraction(|residual|>0.5 rad)<1%"),
        mask_gate=("IoU(repaired badmask, quality-unwrap invalid mask / independent "
                   "modulation mask)>=0.95 AND ROI boundary deviation<=2 px"),
        rule="both phase and mask gates must pass; otherwise stay qualitative",
    )
    out = dict(date="2026-11-20", thresholds=TH, scenes=rows,
               n_fail_A=n_fail_A, n_fail_B=n_fail_B, N_eff=N_eff,
               p_class_gt_insufficient=[],
               g7=dict(handled_need=int(math.ceil(8 * N_eff / 11)) if N_eff >= 10 else None,
                       n_P_min=5, p_pass_need=None,
                       status="insufficient" if N_eff < 10 else "pending"),
               reaudit_prereg=reaudit,
               attribution=("3/6 captures (scene_02/03/06) fail on the same old "
                            "skimage.unwrap_phase GT pipeline; badmask and phase are "
                            "products of that same old pipeline, so the failure is "
                            "systematic, not random.  This does not exempt the scenes."))
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    # markdown table
    L = []
    L.append("# 真实 11 场景 GT audit（Tier A/B，G7 前置）\n")
    L.append("日期：2026-11-20 ｜ 代码：`code/w9_gt_audit.py` ｜ JSON：`probes/results/w9_gt_audit.json`\n")
    L.append("> 口径：Tier A 用独立 GT（scene_01–06 的 gt_phase.npy、Train 的 gt15_rad+mask、实测1 的 fourstep_purphase.mat）做符号/2π/平面（Train、实测1）对齐后的交叉验证；Tier B（实测2、实验3/4）没有独立 GT，只有通过预注册自洽套件才进 G7。**没有一个场景可以“看起来对”就进 G7。**\n")
    L.append("## 0. 预注册阈值（看 G7 前写死）\n")
    L.append("| 项 | 阈值 |\n|---|---:|")
    for k, v in TH.items():
        L.append(f"| {k} | {v} |")
    L.append("\n## 1. 审计表\n")
    L.append("| 场景 | Tier | 帧 | shape | 有效 mask | 相位范围 (rad) | 交叉验证 | median | P95 | max | wrap P95 | jump% | model resid | 1st≈5th | dir raw→canon | 判定 |")
    L.append("|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|")
    for r in rows:
        c = r.get("cross")
        cr = "—" if c is None else c["ref"]
        med = "—" if c is None else f"{c['median']:.4g}"
        p95 = "—" if c is None else f"{c['p95']:.4g}"
        mx = "—" if c is None else f"{c['max']:.4g}"
        f5 = "N/A" if not r.get("first5_applicable", False) else (f"{r['first5']:.4g}" if r["first5"] is not None else "—")
        dirc = ("%s→%s" % ("ok" if r["direction_raw"] else "BAD", "ok" if r["direction_canonical"] else "BAD"))
        p = r.get("gt_status", "FAIL")
        p = {"PASS": "**PASS**", "Tier B FAIL": "**FAIL**",
             "Tier A FAIL": "**FAIL**"}.get(p, "**%s**" % p)
        L.append("| {scene} | {tier} | {nf} | {shape} | {vf:.3f} | [{r0:.3f}, {r1:.3f}] | {cr} | {med} | {p95} | {mx} | {wp:.4g} | {jf:.3g} | {mr:.3g} | {f5} | {dirc} | {p} |".format(
            scene=r["scene"], tier=r["tier"], nf=r["n_frames"],
            shape="×".join(map(str, r["shape"])), vf=r["valid_frac"],
            r0=r["range"][0], r1=r["range"][1],
            cr=cr, med=med, p95=p95, mx=mx, wp=r["wrap_p95"], jf=r["jump_frac"] * 100,
            mr=r["model_resid_med"], f5=f5, dirc=dirc, p=p))
    L.append("\n**判定说明**：PASS = 通过预注册阈值；FAIL = 失败；GT-insufficient / Tier A FAIL = 保留原始数字但不进 G7 的 P 像素统计。\n")
    L.append("## 2. Tier A/B、N_eff 与 G7\n")
    L.append("```")
    L.append("N_eff = 11 - n_fail_A - n_fail_B")
    L.append(f"n_fail_A = {n_fail_A}  (scene_02/03/06: GT-insufficient; 实测1: Tier A FAIL)")
    L.append(f"n_fail_B = {n_fail_B}  (实验3: model residual 0.132 > 0.10)")
    L.append(f"N_eff   = {N_eff}")
    L.append("```")
    if N_eff < 10:
        L.append("- **G7 = insufficient**（N_eff < 10；不是 fail，也不是 pass；不允许靠缩小分母过关）")
    else:
        L.append(f"- G7 门槛：handled ≥ **{math.ceil(8*N_eff/11)}/{N_eff}**；n_P ≥ **5**；P_pass ≥ ceil(0.8·n_P)。")
    L.append("- “P 但 GT 不可评”场景单列 `P-class, GT-insufficient`：不进 P_pass 分子，也不进分母；不默认挂、不默认过。")
    L.append("- 若 scene_02/03/06 后续被 C1 判 U，它们可回到 N_eff 并计 handled；判 P 则只能等 badmask 修复。")
    L.append("\n## 3. 失败/风险备注\n")
    for r in rows:
        if not r.get("pixel_evaluable", False) or not r["pass"]:
            L.append(f"- **{r['scene']}**：{r.get('gt_status')}；cross={r.get('cross')}；valid={r['valid_frac']:.3f}；wrap={r['wrap_p95']:.4g}；jump={r['jump_frac']*100:.2f}%；model_resid={r['model_resid_med']:.3g}；dir raw/canon={r['direction_raw']}/{r['direction_canonical']}")
    L.append("\n## 4. scene_02/03/06 re-audit 预注册（锁定 2026-11-20）\n")
    L.append("- **相位门槛**：按连通域 `k_c=round(median(Δ/2π))`，`residual=Δ−2πk_c`；`P95(residual) ≤ 0.1 rad` 且 `|residual|>0.5 rad` 的像素 `<1%`。")
    L.append("- **mask 门槛**：修复后的 badmask 与质量导向解包 invalid mask（或独立调制度 mask）`IoU ≥ 0.95`，且 ROI 边界偏差 `≤2 px`。")
    L.append("- 两条都过才恢复像素级评价；否则维持“不可像素级评价/定性”。**不按连通域自动对齐，不用质量导向解包当 GT。**")
    L.append("\n## 5. 归因（不用于免责）\n")
    L.append("- 6 个 captures 里 3 个（scene_02/03/06）挂在同一条旧 `skimage.unwrap_phase` GT 管线上；失败不是随机分布，而是旧管线在大连通域上的系统性 2π 跳变。")
    L.append("- badmask 与相位是同一次旧管线产物，所以 mask 也不对齐；处置仍是“修好 / 降级 / insufficient”，不是免责。")
    L.append("\n> 注意：实测2 / 实验3 / 实验4 的“GT”是 4 步自洽参考，不是独立标定；只有通过上表套件才进 G7。Tier B 失败场景只做定性，并从 N_eff 中剔除。")
    open(os.path.join(RES, "w9_gt_audit.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
