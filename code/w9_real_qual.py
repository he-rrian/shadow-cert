"""w9_real_qual.py -- descriptive real-scene point-error table (6 GT-valid scenes).

Scenes: scene_01, scene_04, scene_05, Train, 实测2 (shice2), 实验4 (exp4).
This is descriptive only: no G7 gate, no acceptance set.  It runs the S5 model on
single-frame 256^2 windows, merges psi/sigma, aligns to the audited GT, and reports
median/P95/max/RMSE on GT-valid pixels.

Run on the GPU server (torch):
  python code/w9_real_qual.py --real-root data/real11 --ckpt-root runs \
      --seeds 0,1,2,3,4 --out probes/results/w9_real_qual.json
"""
import os, sys, json, math, argparse
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from w9_gt_audit import (unwrap_quality, scene_4step, valid_from_mod,
                         signed_carrier_fx, align, align_plane)
from w9_roi import estimate_roi
from w5_prep_frontend import frontend

PHI_TOL = math.pi / 10.0
WIN = 256
STRIDE = 128

# scene name, dir, input file, kind, gt file
SCENES = [
    ("scene_01", "scene_01", "single_shot.png", "phase", "gt_phase.npy"),
    ("scene_04", "scene_04", "single_shot.png", "phase", "gt_phase.npy"),
    ("scene_05", "scene_05", "single_shot.png", "phase", "gt_phase.npy"),
    ("Train", "Train", "input00.png", "train", "gt15_rad.npy"),
    ("实测2", "shice2", "fpp_shice_2_1.bmp", "4step", ""),
    ("实验4", "exp4", "cropped_fpp_shice_3_1.jpg", "4step", ""),
]


def im(p):
    a = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(p)
    if a.ndim == 3:
        a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2GRAY)
    return a.astype(np.float32)


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def positions(n):
    if n <= WIN:
        return [0]
    ps = list(range(0, n - WIN + 1, STRIDE))
    if ps[-1] != n - WIN:
        ps.append(n - WIN)
    return sorted(set(ps))


def load_scene(real_root, name, sub, fin, kind, gtfile):
    d = os.path.join(real_root, sub)
    I = im(os.path.join(d, fin))
    if kind == "phase":
        gt = np.load(os.path.join(d, gtfile)).astype(np.float32)
        valid = np.isfinite(gt)
        bad = None
        bp = os.path.join(d, "badmask_single.png")
        if os.path.exists(bp):
            bad = cv2.imdecode(np.fromfile(bp, np.uint8), cv2.IMREAD_GRAYSCALE) > 127
        src = "gt_phase.npy (total)"
    elif kind == "train":
        gt15 = np.load(os.path.join(d, gtfile)).astype(np.float32)
        mask = np.load(os.path.join(d, "gt15_mask.npy")).astype(bool)
        H, W = gt15.shape
        yy, xx = np.mgrid[0:H, 0:W]
        # audited deviation->total convention: phi_total = -gt15_rad + a + bx*x + by*y
        gt = -gt15 + (-196.3548593435633 + 0.6302519826223105 * xx + 2.621553102482904e-05 * yy)
        valid = mask & np.isfinite(gt)
        bad = None
        src = "gt15_rad.npy (deviation+carrier plane)"
    elif kind == "4step":
        fs = sorted(f for f in os.listdir(d) if f.lower().endswith((".bmp", ".jpg", ".png"))
                    and (f.startswith("fpp_shice_2_") or f.startswith("cropped_fpp_shice_3_")))
        frames = [im(os.path.join(d, f)) for f in fs[:4]]
        phi_w, B = scene_4step(frames)
        valid = valid_from_mod(B, 0.10)
        gt = unwrap_quality(phi_w, B, valid)
        bad = None
        src = "4-step quality-guided unwrap"
    else:
        raise ValueError(kind)

    # canonicalise to +x (same convention as training)
    fx = signed_carrier_fx(I, valid if valid.any() else np.ones(I.shape, bool))
    if np.isfinite(fx) and fx < 0:
        I = I[:, ::-1].copy(); gt = gt[:, ::-1].copy(); valid = valid[:, ::-1].copy()
        if bad is not None:
            bad = bad[:, ::-1].copy()
    roi, fb, empty = estimate_roi(I)
    if empty or roi.sum() < 100:
        roi = np.ones(I.shape, bool)
    return dict(name=name, I=I, gt=gt, valid=valid, bad=bad, roi=roi, src=src)


def build_windows(scene):
    I, roi, gt, valid = scene["I"], scene["roi"], scene["gt"], scene["valid"]
    H, W = I.shape
    wins = []
    for y in positions(H):
        for x in positions(W):
            r = roi[y:y + WIN, x:x + WIN]
            if r.sum() < 0.02 * WIN * WIN:
                continue
            if valid[y:y + WIN, x:x + WIN].sum() < 200:
                continue
            Iw = I[y:y + WIN, x:x + WIN]
            try:
                fe = frontend(Iw, r)
            except Exception:
                continue
            wins.append(dict(y=y, x=x, I=Iw.astype(np.uint8),
                             h=fe["h"].astype(np.float32), M=fe["M"].astype(np.float32),
                             roi=r.astype(bool), dead=fe["dead"].astype(bool),
                             carrier=np.asarray(fe["carrier"], np.float32)))
    return wins


def merge_seed(model, wins, device, H, W):
    """Merge overlapping windows after per-window 2*pi/offset alignment.

    Each window's frontend h (and therefore psi) has its own global piston/branch;
    averaging raw window psi tears the phase.  Align each new window to the current
    canvas on the overlap (2*pi*k + constant), then merge.
    """
    import torch
    psi_sum = np.zeros((H, W), np.float64); sig_sum = np.zeros((H, W), np.float64)
    cnt = np.zeros((H, W), np.int32); dead_any = np.zeros((H, W), bool)
    shifts = []
    bs = 8
    with torch.no_grad():
        for i in range(0, len(wins), bs):
            w = wins[i:i + bs]
            b = dict(
                I=torch.from_numpy(np.stack([q["I"] for q in w])).to(device),
                h=torch.from_numpy(np.stack([q["h"] for q in w])).to(device),
                M=torch.from_numpy(np.stack([q["M"] for q in w])).to(device),
                roi=torch.from_numpy(np.stack([q["roi"] for q in w])).to(device),
                carrier=torch.from_numpy(np.stack([q["carrier"] for q in w])).to(device),
            )
            from w5_model import make_input, split_output
            dpsi, log_sigma, _, _, _ = split_output(model(make_input(b)))
            psi = (b["h"].float().unsqueeze(1) + b["M"].float().unsqueeze(1) * dpsi).squeeze(1)
            sig = torch.exp(0.5 * log_sigma).squeeze(1)
            psi = psi.cpu().numpy(); sig = sig.cpu().numpy()
            for j, q in enumerate(w):
                y, x = q["y"], q["x"]
                sl = (slice(y, y + WIN), slice(x, x + WIN))
                ps = psi[j].astype(np.float64)
                ov = cnt[sl] > 0
                if ov.sum() >= 100:
                    canvas = psi_sum[sl][ov] / cnt[sl][ov]
                    wv = ps[ov]
                    best = None
                    for k in range(-20, 21):
                        d = canvas - (wv + 2 * np.pi * k)
                        c = float(np.median(d))
                        r = np.abs(d - c)
                        med = float(np.median(r))
                        if best is None or med < best[0]:
                            best = (med, k, c)
                    if best is not None:
                        ps = ps + 2 * np.pi * best[1] + best[2]
                        shifts.append((q["y"], q["x"], int(best[1]), float(best[2])))
                psi_sum[sl] += ps
                sig_sum[sl] += sig[j]
                cnt[sl] += 1
                dead_any[sl] |= q["dead"]
    m = cnt > 0
    psi = np.full((H, W), np.nan, np.float64); sig = np.full((H, W), np.nan, np.float64)
    psi[m] = psi_sum[m] / cnt[m]; sig[m] = sig_sum[m] / cnt[m]
    return psi, sig, dead_any, m, shifts


def metrics_pair(gt, psi, mask):
    """Two descriptive alignments:
    offset = sign + 2*pi*k + constant piston (raw global error);
    plane  = sign + fitted linear carrier/tilt (local shape error).
    """
    if mask.sum() < 100:
        return None
    yy, xx = np.mgrid[0:gt.shape[0], 0:gt.shape[1]]
    off = align(gt, psi, mask)
    pl = align_plane(gt, psi, mask)
    out = {}
    for name, a in (("offset", off), ("plane", pl)):
        if "k" in a:
            al = a["sign"] * psi + 2 * np.pi * a["k"] + a["offset"]
        else:
            al = a["sign"] * psi + a["a"] + a["bx"] * xx + a["by"] * yy
        e = np.abs(gt - al)[mask]
        ew = np.abs(wrap(gt - al))[mask]
        out[name] = dict(n=int(mask.sum()), median=float(np.median(e)),
                         p95=float(np.percentile(e, 95)), max=float(np.max(e)),
                         rmse=float(np.sqrt(np.mean(e ** 2))),
                         median_wrap=float(np.median(ew)),
                         p95_wrap=float(np.percentile(ew, 95)),
                         frac_tol=float(np.mean(e <= PHI_TOL)),
                         frac_tol_wrap=float(np.mean(ew <= PHI_TOL)),
                         align=a)
    return out


def save_fig(path, I, gt, psi, e, mask):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e2 = np.where(mask, e, np.nan)
    vmax = max(float(np.nanpercentile(e2, 95)) * 1.5, 0.5)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(I, cmap="gray"); ax[0].set_title("input")
    ax[1].imshow(gt, cmap="jet"); ax[1].set_title("GT (rad)")
    ax[2].imshow(psi, cmap="jet"); ax[2].set_title("Ours (offset-aligned)")
    im4 = ax[3].imshow(e2, cmap="magma", vmin=0, vmax=vmax); ax[3].set_title("|err| rad")
    for a in ax:
        a.axis("off")
    fig.colorbar(im4, ax=ax[3], fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-root", default=os.path.join(ROOT, "data", "real11"))
    ap.add_argument("--ckpt-root", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default=os.path.join(ROOT, "probes", "results", "w9_real_qual.json"))
    a = ap.parse_args()
    import torch
    from w5_model import W5UNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(x) for x in a.seeds.split(",")]
    os.makedirs(os.path.join(ROOT, "probes", "results", "w9_real_qual"), exist_ok=True)
    rows = []
    for name, sub, fin, kind, gtfile in SCENES:
        print("====", name, flush=True)
        sc = load_scene(a.real_root, name, sub, fin, kind, gtfile)
        wins = build_windows(sc)
        H, W = sc["I"].shape
        print(f"  windows={len(wins)} shape={H}x{W} roi={sc['roi'].mean():.3f}", flush=True)
        seed_rows = []
        seed4 = None
        for s in seeds:
            ck = os.path.join(a.ckpt_root, f"w5_p0_seed{s}", "best_stage3.pth")
            if not os.path.exists(ck):
                print("  missing", ck); continue
            ckpt = torch.load(ck, map_location="cpu", weights_only=False)
            model = W5UNet(base=128).to(device); model.load_state_dict(ckpt["model"]); model.eval()
            psi, sig, dead, cov, shifts = merge_seed(model, wins, device, H, W)
            mask = sc["valid"] & sc["roi"] & cov
            mp = metrics_pair(sc["gt"], psi, mask)
            if mp is None:
                continue
            met = dict(seed=s, eval_px=mp["offset"]["n"],
                       offset=mp["offset"], plane=mp["plane"])
            if sc["bad"] is not None:
                cm = metrics_pair(sc["gt"], psi, mask & (~sc["bad"]))
                met["clean_plane_p95"] = cm["plane"]["p95"] if cm else None
            else:
                met["clean_plane_p95"] = met["plane"]["p95"]
            seed_rows.append(met)
            print(f"  seed{s}: n={met['eval_px']} med={met['offset']['median']:.4f} "
                  f"P95={met['offset']['p95']:.4f} P95wrap={met['offset']['p95_wrap']:.4f} "
                  f"fracTol={met['offset']['frac_tol']:.3f} fracTolWrap={met['offset']['frac_tol_wrap']:.3f}", flush=True)
            if s == 4:
                seed4 = (psi, met)
            del model
        if not seed_rows:
            rows.append(dict(scene=name, gt=sc["src"], n_windows=len(wins), ok=False))
            continue
        pp = [m["offset"]["p95"] for m in seed_rows]
        pw = [m["offset"]["p95_wrap"] for m in seed_rows]
        ft = [m["offset"]["frac_tol"] for m in seed_rows]
        fw = [m["offset"]["frac_tol_wrap"] for m in seed_rows]
        cleans = [m["clean_plane_p95"] for m in seed_rows if m.get("clean_plane_p95") is not None]
        row = dict(scene=name, gt=sc["src"], n_windows=len(wins),
                   eval_px=seed_rows[0]["eval_px"],
                   seed4=next((m for m in seed_rows if m["seed"] == 4), seed_rows[0]),
                   p95_offset_mean=float(np.mean(pp)),
                   p95_offset_std=float(np.std(pp, ddof=1)) if len(pp) > 1 else 0.0,
                   p95_wrap_mean=float(np.mean(pw)),
                   frac_tol_mean=float(np.mean(ft)),
                   frac_tol_wrap_mean=float(np.mean(fw)),
                   seeds=seed_rows)
        rows.append(row)
        if seed4 is not None:
            psi, met = seed4
            al = met["offset"]["align"]
            aligned = al["sign"] * psi + 2 * np.pi * al["k"] + al["offset"]
            e = np.abs(sc["gt"] - aligned)
            mask = sc["valid"] & sc["roi"] & np.isfinite(psi)
            save_fig(os.path.join(ROOT, "probes", "results", "w9_real_qual", f"{sub}.png"),
                     sc["I"], sc["gt"], psi, e, mask)
    json.dump(dict(date="2026-11-21", note="descriptive only; no G7 gate", scenes=rows),
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
