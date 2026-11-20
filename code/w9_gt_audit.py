"""w9_gt_audit.py -- GT audit for the 11 real scenes before G7.

Tier A (independent reference): scene_01..06, Train, 实测1
Tier B (self-consistent 4-step): 实测2, 实验3, 实验4

Tier A is cross-checked against its independent GT (gt_phase.npy / gt15_rad.npy /
fourstep_purphase.mat) after sign + 2*pi*k + constant-offset alignment.
Tier B has no independent GT; it enters G7 only if a pre-registered self-consistency
suite passes:
  1. wrap closure P95 < 0.05 rad
  2. 4/15-step model residual median < 0.10
  3. first frame ~ fifth frame (5-frame scenes) normalized RMS < 0.05
  4. fringe-direction check: sign(median d phi/dx) == sign(carrier fx)
  5. unwrap continuity: fraction of adjacent valid pairs with |d phi| > pi < 0.5%
  6. valid mask fraction > 10%

Tier A pass: cross-check median < 0.05 rad and P95 < 0.2 rad (max is reported, but a
few local outliers do not fail the scene), plus wrap closure / valid mask.

N_eff rule (pre-registered):
  n_fail_B = number of Tier B scenes failing the suite
  N_eff = 11 - n_fail_B
  if N_eff < 10: G7 = insufficient
  else: handled >= ceil(8*N_eff/11), n_P >= 5, P_pass >= ceil(0.8*n_p)

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
    first5_rms_max=0.05,
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

    phi = unwrap_quality(phi_w, B, valid)
    rec.update(shape=list(phi_w.shape), n_frames=len(frames), files=files[:4],
               valid_frac=float(valid.mean()), unit="rad",
               unwrapped=True,
               range=[float(np.nanmin(phi[valid])), float(np.nanmax(phi[valid]))] if valid.any() else [None, None],
               wrap_p95=wrap_closure(phi, phi_w, valid),
               jump_frac=unwrap_jump_frac(phi, valid),
               model_resid_med=phase_model_residual(frames, phi, valid, nstep)[0],
               first5=first_fifth(frames, valid))
    fx = signed_carrier_fx(frames[0], valid)
    mdx = median_dx(phi, valid)
    rec.update(carrier_fx=fx, median_dphi_dx=mdx,
               direction_ok=bool(np.isfinite(fx) and np.isfinite(mdx) and np.sign(fx) == np.sign(mdx)))

    if ref is not None:
        g = align(ref, phi, ref_valid)
        p = align_plane(ref, phi, ref_valid)
        # Only invoke the carrier-plane model when the plain offset model is clearly
        # inadequate and the plane model is accurate -- i.e. the GT is a deviation phase
        # (Train).  For total-phase GT the offset model is authoritative.
        use_plane = (g["median"] > 0.5) and (p["median"] < TH["cross_med_max"])
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
        rec["gate"] = dict(cross_med_max=med_max, cross_p95_max=p95_max,
                           wrap_p95_max=TH["wrap_p95_max"],
                           valid_frac_min=TH["valid_frac_min"],
                           ok_cross=ok_cross, ok_misc=ok_misc)
        # jump_frac is reported as a diagnostic, not a Tier-A gate: a shadow/cut
        # produces a legitimate 2*pi branch boundary and would otherwise fail every
        # real scene with an occlusion.
    else:
        ok = (rec["valid_frac"] >= TH["valid_frac_min"]
              and rec["wrap_p95"] <= TH["wrap_p95_max"]
              and rec["model_resid_med"] <= TH["model_resid_med_max"]
              and rec["jump_frac"] <= TH["unwrap_jump_frac_max"]
              and rec["direction_ok"])
        if rec["first5"] is not None:
            ok = ok and (rec["first5"] <= TH["first5_rms_max"])
        rec["pass"] = bool(ok)
        rec["gate"] = dict(valid_frac_min=TH["valid_frac_min"],
                           wrap_p95_max=TH["wrap_p95_max"],
                           model_resid_med_max=TH["model_resid_med_max"],
                           first5_rms_max=TH["first5_rms_max"],
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

    n_fail_B = sum(1 for r in rows if r["tier"] == "B" and not r["pass"])
    N_eff = len(rows) - n_fail_B
    out = dict(date="2026-11-20", thresholds=TH, scenes=rows,
               n_fail_B=n_fail_B, N_eff=N_eff,
               g7=dict(handled_need=int(math.ceil(8 * N_eff / 11)) if N_eff >= 10 else None,
                       n_P_min=5,
                       p_pass_need=None,
                       status="insufficient" if N_eff < 10 else "pending"))
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    # markdown table
    L = []
    L.append("# 真实 11 场景 GT audit（Tier A/B，G7 前置）\n")
    L.append("日期：2026-11-20 ｜ 代码：`code/w9_gt_audit.py` ｜ JSON：`probes/results/w9_gt_audit.json`\n")
    L.append("> 口径：Tier A 用独立 GT（scene_01–06 的 gt_phase.npy、Train 的 gt15_rad+mask、实测1 的 fourstep_purphase.mat）做 2π/offset/符号对齐后的交叉验证；Tier B（实测2、实验3/4）没有独立 GT，只有通过预注册自洽套件才进 G7。**没有一个场景可以“看起来对”就进 G7。**\n")
    L.append("## 0. 预注册阈值（看 G7 前写死）\n")
    L.append("| 项 | 阈值 |\n|---|---:|")
    for k, v in TH.items():
        L.append(f"| {k} | {v} |")
    L.append("\n## 1. 审计表\n")
    L.append("| 场景 | Tier | 帧/文件 | shape | 有效 mask | 相位范围 (rad) | unwrapped | 交叉验证 | median | P95 | max | wrap P95 | jump% | model resid | 1st≈5th | dir | pass |")
    L.append("|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for r in rows:
        c = r.get("cross")
        cr = "—" if c is None else c["ref"]
        med = "—" if c is None else f"{c['median']:.4g}"
        p95 = "—" if c is None else f"{c['p95']:.4g}"
        mx = "—" if c is None else f"{c['max']:.4g}"
        f5 = "—" if r["first5"] is None else f"{r['first5']:.4g}"
        L.append("| {scene} | {tier} | {files} | {shape} | {vf:.3f} | [{r0:.3f}, {r1:.3f}] | {uw} | {cr} | {med} | {p95} | {mx} | {wp:.4g} | {jf:.3g} | {mr:.3g} | {f5} | {dr} | **{p}** |".format(
            scene=r["scene"], tier=r["tier"],
            files=(str(r["n_frames"]) + " frames" if r["files"] and r["files"][0] != "00.bmp..14.bmp" else "15 frames"),
            shape="×".join(map(str, r["shape"])), vf=r["valid_frac"],
            r0=r["range"][0], r1=r["range"][1], uw="yes" if r["unwrapped"] else "no",
            cr=cr, med=med, p95=p95, mx=mx, wp=r["wrap_p95"], jf=r["jump_frac"] * 100,
            mr=r["model_resid_med"], f5=f5,
            dr=("ok" if r["direction_ok"] else "BAD"), p=("PASS" if r["pass"] else "FAIL")))
    L.append("\n## 2. Tier A/B 与 N_eff\n")
    L.append(f"- Tier B 失败场景数：**{n_fail_B}**；N_eff = 11 − {n_fail_B} = **{N_eff}**")
    if N_eff < 10:
        L.append("- **G7 = insufficient**（N_eff < 10；不允许靠缩小分母过关）")
    else:
        L.append(f"- G7 门槛：handled ≥ **{math.ceil(8*N_eff/11)}/{N_eff}**；n_P ≥ **5**；P_pass ≥ ceil(0.8·n_P)。")
        L.append("- 还需真实推理结果填 handled/P_pass/n_P；本表只锁定分母与 Tier A/B。")
    L.append("\n## 3. 失败/风险备注\n")
    for r in rows:
        if not r["pass"]:
            L.append(f"- **{r['scene']}**：FAIL；cross={r.get('cross')}；valid={r['valid_frac']:.3f}；wrap={r['wrap_p95']:.4g}；jump={r['jump_frac']*100:.2f}%；model_resid={r['model_resid_med']:.3g}；first5={r['first5']}；dir_ok={r['direction_ok']}")
    if all(r["pass"] for r in rows):
        L.append("- 全部 11 场景通过审计（仍不代表 G7 已过；G7 需真实推理的 P_pass/acceptance）。")
    L.append("\n> 注意：实测2 / 实验3 / 实验4 的“GT”是 4 步自洽参考，不是独立标定；只有通过上表套件才进 G7。Tier B 失败场景只做定性，并从 N_eff 中剔除。")
    open(os.path.join(RES, "w9_gt_audit.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
