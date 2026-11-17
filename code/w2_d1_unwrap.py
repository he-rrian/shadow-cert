"""W2 / D1-D2  --  SynthFringe cascaded temporal phase unwrapping, 50-scene pilot.

There is NO ground-truth phase file in SynthFringe (census: 176,400 .bmp + 8 .txt).
The residual is therefore defined physically, by three independent criteria:

  R1  order-decision residual   e_k = wrap(ratio*Phi_low - psi_high)   [rad]
      -> the quantity that decides whether a 2*pi error happens; |e|>pi/2 means "close to wrong"
  R2  spatial tearing           count(|dPhi_64| > pi) on 4-neighbours   [px]
      -> a 2*pi order error shows up as a hard step of 2*pi
  R3  independent cross-check   spatial unwrap of psi_64 ALONE (no coarse frequency),
      restricted to the largest connected component, compared with the cascade [rad]

Pre-flight checks (user-mandated):
  P1  does psi_1 already contain a -pi -> +pi wrap line?  count & report
  P2  sign convention + per-level phase offset  (circular mean of ratio*Phi_low - psi_high)
  P3  phi_GT availability  -> NONE, hence R1-R3

Run: python w2_d1_unwrap.py [--n 50]
"""
import os, sys, json, argparse
import numpy as np, cv2
from skimage.restoration import unwrap_phase

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # 2\P2
RES = os.path.join(ROOT, "probes", "results"); os.makedirs(RES, exist_ok=True)
BASE = r"F:\paper\SynthFringe"
SUBS = [("1_12", 12, 1.0), ("3_16", 16, 3.0), ("8_6", 6, 8.0), ("64_15", 15, 64.0)]

def im(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

def load_seq(sub, sid):
    p = os.path.join(BASE, sub, sid)
    fs = sorted(f for f in os.listdir(p) if f.lower().endswith(".bmp"))
    out = []
    for f in fs:
        a = im(os.path.join(p, f))
        if a is None: continue
        if a.ndim == 3: a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2GRAY)
        out.append(a.astype(np.float32))
    return np.stack(out)

def nstep(I, sign=+1):
    """I_k = A + B cos(phi - delta_k), delta_k = 2*pi*k/N  ->  phi = atan2(sum I sin, sum I cos)"""
    N = I.shape[0]
    d = 2 * np.pi * np.arange(N) / N
    s = np.tensordot(np.sin(d) * sign, I, axes=(0, 0))
    c = np.tensordot(np.cos(d), I, axes=(0, 0))
    psi = np.arctan2(s, c)
    amp = np.hypot(s, c) / N
    return psi, amp

def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def spatial_unwrap(psi, roi, largest_only=True):
    """absolute phase up to one global constant per connected component"""
    m = roi.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(roi)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, 4]))
        keep = (lab == big) if largest_only else (lab > 0)
    z = psi.copy(); z[~keep] = 0.0
    un = unwrap_phase(z)
    return un, keep

def scene(sid):
    raw = {}
    for sub, N, ratio in SUBS:
        I = load_seq(sub, sid)
        raw[sub] = (I, N, ratio)
    mx = np.max([raw[s][0].max(0) for s, _, _ in SUBS], axis=0)
    roi = cv2.morphologyEx((mx >= 25).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    rec = dict(sid=sid, roi=float(roi.mean()))
    if roi.mean() < 0.02:
        return None

    # ---- sign convention: pick the one that makes the 1->3 residual more concentrated
    best = None
    for sign in (+1, -1):
        psi1, amp1 = nstep(raw["1_12"][0], sign)
        psi3, _ = nstep(raw["3_16"][0], sign)
        u1, _ = spatial_unwrap(psi1, roi)
        e = wrap(3.0 * u1 - psi3)[roi]
        conc = float(np.abs(np.angle(np.mean(np.exp(1j * e)))))   # 0 = perfectly concentrated
        if best is None or conc < best[0]:
            best = (conc, sign)
    sign = best[1]
    rec["sign"] = int(sign)
    rec["concentration"] = float(best[0])

    # ---- P1: does psi_1 already carry a wrap line?
    psi1, amp1 = nstep(raw["1_12"][0], sign)
    gx = np.abs(np.diff(psi1, axis=1)); gy = np.abs(np.diff(psi1, axis=0))
    wrap1 = int((gx > np.pi).sum() + (gy > np.pi).sum())
    rec["psi1_wrap_px"] = wrap1

    # ---- cascade
    # NOTE: the order k MUST be computed from the UNWRAPPED difference a*Phi_low - psi_high.
    # Wrapping it first destroys the order information and forces k == 0 (a bug caught in
    # the pilot).  The offset is estimated from the wrapped residual only (it is a constant).
    phi = {}
    psi = {}
    offs = {}
    e_each = {}
    u1, _ = spatial_unwrap(psi1, roi)
    phi["1_12"], psi["1_12"] = u1, psi1
    for (sub_l, N_l, r_l), (sub_h, N_h, r_h) in zip(SUBS[:-1], SUBS[1:]):
        if sub_h not in psi:
            psi[sub_h], _ = nstep(raw[sub_h][0], sign)
        ratio = r_h / r_l
        d_raw = ratio * phi[sub_l] - psi[sub_h]
        off = float(np.angle(np.mean(np.exp(1j * wrap(d_raw)[roi]))))
        offs[f"{sub_l}->{sub_h}"] = off
        k = np.round((d_raw - off) / (2 * np.pi))
        phi[sub_h] = 2 * np.pi * k + psi[sub_h] + off
        e_each[sub_h] = float(np.percentile(np.abs(wrap(d_raw - off)[roi]), 99))

    phi64 = phi["64_15"]
    e_last = wrap(8.0 * phi["8_6"] - psi["64_15"])
    rec["offsets"] = offs
    rec["e99_per_step"] = e_each
    rec["e_last_p50"] = float(np.median(np.abs(e_last[roi])))
    rec["e_last_p99"] = float(np.percentile(np.abs(e_last[roi]), 99))
    rec["e_last_max"] = float(np.abs(e_last[roi]).max())
    rec["e_last_frac_gt_halfpi"] = float((np.abs(e_last[roi]) > np.pi / 2).mean())

    # ---- R2 spatial tearing
    dx = np.abs(np.diff(phi64, axis=1))[roi[:, :-1] & roi[:, 1:]]
    dy = np.abs(np.diff(phi64, axis=0))[roi[:-1, :] & roi[1:, :]]
    rec["tear_px"] = int((dx > np.pi).sum() + (dy > np.pi).sum())
    rec["grad_p99"] = float(np.percentile(np.concatenate([dx, dy]), 99)) if dx.size else 0.0

    # ---- R3 independent cross-check on the largest component
    u64, keep = spatial_unwrap(psi["64_15"], roi)
    if keep.sum() > 1000:
        d3 = wrap(phi64[keep] - u64[keep])
        rec["r3_p99"] = float(np.percentile(np.abs(d3), 99))
        rec["r3_frac_nonzero"] = float((np.abs(d3) > 0.5).mean())
    else:
        rec["r3_p99"] = np.nan; rec["r3_frac_nonzero"] = np.nan

    # ---- case-selection features
    lap1 = np.abs(cv2.Laplacian(phi["1_12"].astype(np.float32), cv2.CV_32F))
    lap64 = np.abs(cv2.Laplacian(phi64.astype(np.float32), cv2.CV_32F))
    g64 = np.abs(cv2.Sobel(psi["64_15"].astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
    rec["flat"] = float(np.median(lap1[roi]))
    rec["complex"] = float(np.percentile(lap64[roi], 95))
    rec["steep"] = float((g64[roi] > 2.0).mean())
    rec["dark"] = float(np.median(amp1[roi]))
    rec["_e_map"] = e_last
    rec["_roi"] = roi
    rec["_phi64"] = phi64
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    ids = [f"{int(round(x)):05d}" for x in np.linspace(0, 3599, a.n)]
    out = []
    for j, sid in enumerate(ids):
        try:
            r = scene(sid)
        except Exception as ex:
            print(f"  [{j+1}/{len(ids)}] {sid} ERROR {type(ex).__name__}: {ex}")
            continue
        if r is None:
            print(f"  [{j+1}/{len(ids)}] {sid} skipped (tiny ROI)"); continue
        print(f"  [{j+1}/{len(ids)}] {sid} roi={r['roi']*100:5.1f}% e99={r['e_last_p99']:.4f}rad "
              f"tear={r['tear_px']:5d} r3p99={r['r3_p99']:.4f} sign={r['sign']:+d}")
        out.append(r)
    # strip heavy arrays before dumping the table
    light = [{k: v for k, v in r.items() if not k.startswith('_')} for r in out]
    json.dump(light, open(os.path.join(RES, "w2_d1_pilot.json"), "w"), indent=1)

    # ---------------- markdown table
    lines = ["| # | scene | ROI% | e99 (rad) | e_max (rad) | >pi/2 % | tear px | grad P99 (rad/px) | R3 P99 (rad) | R3 order-err % |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(light, 1):
        lines.append(f"| {i} | {r['sid']} | {r['roi']*100:.1f} | {r['e_last_p99']:.4f} | {r['e_last_max']:.3f} | "
                     f"{r['e_last_frac_gt_halfpi']*100:.2f} | {r['tear_px']} | {r['grad_p99']:.3f} | "
                     f"{r['r3_p99']:.4f} | {r['r3_frac_nonzero']*100:.2f} |")
    md = os.path.join(ROOT, "doc", "W2_D1_50场景报告.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# W2 / D1–D2  50 场景 Pilot 报告（级联解包裹）\n\n")
        f.write("**前置结论**：SynthFringe **没有 GT 相位文件**（176,400 BMP + 8 标定 txt）。\n")
        f.write("因此残差按 TPU 物理定义给三重判据：**R1** 定阶残差 `e=wrap(ratio·Φ_low−ψ_high)`；"
                "**R2** 空间撕裂计数 `|ΔΦ_64|>π`；**R3** 仅在最大连通域上、**不用任何低频**地空间解包 ψ_64 做独立互检。\n\n")
        f.write("\n".join(lines) + "\n")
    print("\nwrote", md)
    print(f"scenes={len(light)}  e99 median={np.median([r['e_last_p99'] for r in light]):.4f} rad  "
          f"tear total={sum(r['tear_px'] for r in light)}")

if __name__ == "__main__":
    main()
