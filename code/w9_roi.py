"""w9_roi.py -- W9 item 6: ROI given vs single-frame estimated ROI (CPU only).

The certificate is conditional on the measurement domain Omega_roi.  A single frame
cannot uniquely separate "dark occluded surface" from "no surface" (the plan's
Discussion sentence).  This script quantifies the cost of that ambiguity on the
synthetic val/test splits:

  estimate : single-frame ROI from the modulation envelope + contour continuity
             (threshold 0.35 * P95 of the analytic envelope; interior closed contour;
              fallback = largest connected component when no interior contour exists)
  eval     : P/U labels, C1 ORR failure, and C3 coverage under the estimated ROI,
             compared with the given-ROI labels on the same full float16 cache.

This is an ablation, not the official C3 table.  The official C3 numbers remain the
float32 `w8_p_only/` run; this script uses `data/preds_full/` (float16) so that the
same pixels are used for both the given-ROI and estimated-ROI arms.

Run:
  python code/w9_roi.py --stage estimate
  python code/w9_roi.py --stage eval --seeds 0,1,2,3,4
"""
import os, sys, json, math, argparse
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from w5_dataset import label_from_meta
from w5_prep_frontend import carrier_from_image, local_p90
from w3_gen import torsion_vinf

PHI_TOL = math.pi / 10.0
DELTA_G = 0.13
GAMMA = 0.75
BETA = 0.05
ALPHAS = (0.05, 0.10, 0.20)
COLLAR = 2
ROI_THR = 0.35
EST_DIR = os.path.join(ROOT, "data", "w9_roi_est")
FULL = os.path.join(ROOT, "data", "preds_full")


# ------------------------------------------------------------------ ROI estimate
def envelope_abs(I):
    """Analytic-signal envelope at the dominant carrier (no ROI prior)."""
    fy, fx, F, U, V = carrier_from_image(I, np.ones_like(I, bool))
    f0 = max(np.hypot(fx, fy), 1e-6); w = 0.5 * f0
    bm = np.exp(-((U - fx) ** 2 + (V - fy) ** 2) / (2 * (w / 2) ** 2))
    bm *= (np.hypot(U - fx, V - fy) < 2 * w)
    Z = np.fft.ifft2(np.fft.ifftshift(F * bm))
    return np.abs(Z)


def estimate_roi(I, thr=ROI_THR):
    """Single-frame ROI: modulation threshold + interior-contour continuity.

    Returns (mask, fallback, empty).  `fallback=True` means no closed contour that
    avoids the image border was found; the largest connected component is returned
    instead -- this is exactly the failure mode the ablation is meant to expose.
    """
    m = envelope_abs(I)
    ref = float(np.percentile(m, 95))
    if not np.isfinite(ref) or ref <= 0:
        return np.zeros(I.shape, bool), True, True
    raw = ((m >= thr * ref) & (I > 2) & (I < 250)).astype(np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    h, w = raw.shape
    ff = raw.copy(); mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 1)
    filled = raw | (1 - ff)
    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None; best_area = 0.0
    for c in cnts:
        a = cv2.contourArea(c); x, y, ww, hh = cv2.boundingRect(c)
        if x <= 0 or y <= 0 or x + ww >= w - 1 or y + hh >= h - 1:
            continue
        if a > best_area:
            best_area = a; best = c
    if best is None:
        n, lab, st, _ = cv2.connectedComponentsWithStats(filled, 8)
        if n <= 1:
            return np.zeros((h, w), bool), True, True
        j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        return (lab == j), True, False
    out = np.zeros((h, w), np.uint8)
    cv2.drawContours(out, [best], -1, 1, -1)
    ff = out.copy(); mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 1)
    out = out | (1 - ff)
    return out.astype(bool), False, False


# ------------------------------------------------------------------ io
def load_flip(split):
    """Frontend carrier-sign canonicalisation: 1 = source frame was mirrored to +x."""
    d = os.path.join(ROOT, "data", "frontend", split)
    man = json.load(open(os.path.join(d, "frontend_manifest.json")))
    out = []
    for sh in man["shards"]:
        z = np.load(os.path.join(d, sh["path"]))
        out.append(z["flip"].astype(bool))
    return np.concatenate(out)


def iter_source(split):
    man = json.load(open(os.path.join(ROOT, "data", split, "manifest_000.json")))
    flips = load_flip(split)
    k = 0
    for sh in man["shards"]:
        z = np.load(os.path.join(ROOT, "data", split, sh["path"]))
        # bind once per shard: repeated z["I"] lookups decompress the whole member
        I = z["I"]; phi = z["phi"]; roi = z["roi"].astype(bool)
        dead = z["dead"].astype(bool); orr = z["orr"]; kmap = z["kmap"]; meta = z["meta"]
        for i in range(len(I)):
            ii = I[i]; pp = phi[i]; rr = roi[i]; dd = dead[i]; oo = orr[i]; kk = kmap[i]
            if flips[k]:
                ii = ii[:, ::-1].copy(); pp = pp[:, ::-1].copy(); rr = rr[:, ::-1].copy()
                dd = dd[:, ::-1].copy(); oo = oo[:, ::-1].copy(); kk = kk[:, ::-1].copy()
            yield k, dict(I=ii, phi=pp, roi=rr, dead=dd, orr=oo, kmap=kk,
                          meta=json.loads(meta[i]))
            k += 1


def stage_estimate(splits):
    os.makedirs(EST_DIR, exist_ok=True)
    out = {}
    for split in splits:
        masks = []; stats = []
        for k, s in iter_source(split):
            est, fb, empty = estimate_roi(s["I"].astype(np.float32))
            masks.append(est)
            inter = float((est & s["roi"]).sum()); union = float((est | s["roi"]).sum())
            miss = float((s["roi"] & ~est).sum() / max(s["roi"].sum(), 1))
            leak = float((est & ~s["roi"]).sum() / max(est.sum(), 1))
            ret = float((s["dead"] & est).sum() / max(s["dead"].sum(), 1))
            stats.append(dict(idx=k, iou=inter / max(union, 1), miss=miss, leak=leak,
                              dead_ret=ret, n_dead=int(s["dead"].sum()),
                              n_dead_ret=int((s["dead"] & est).sum()),
                              fallback=bool(fb), empty=bool(empty)))
            if (k + 1) % 250 == 0:
                print(f"  {split}: {k+1} scenes", flush=True)
        np.savez_compressed(os.path.join(EST_DIR, f"{split}.npz"),
                            roi_est=np.stack(masks, 0))
        out[split] = stats
        print(f"{split}: {len(stats)} scenes -> {EST_DIR}/{split}.npz", flush=True)
    good = [r for r in out["val"] if not r["empty"]]
    with_dead = [r for r in out["val"] if r["n_dead"] > 0]
    print("val ROI summary: mean IoU %.3f, median %.3f, fallback %.1f%%, empty %.1f%%, "
          "dead retention mean %.3f, dead-loss scenes %.1f%%"
          % (np.mean([r["iou"] for r in good]), np.median([r["iou"] for r in good]),
             100 * np.mean([r["fallback"] for r in out["val"]]),
             100 * np.mean([r["empty"] for r in out["val"]]),
             np.mean([r["dead_ret"] for r in with_dead]),
             100 * np.mean([r["dead_ret"] < 0.999 for r in with_dead])))
    json.dump(out, open(os.path.join(ROOT, "probes", "results", "w9_roi_stats.json"), "w"), indent=1)


# ------------------------------------------------------------------ stats helpers
def q_high(a, q):
    return float(np.quantile(np.asarray(a), q, method="higher"))


def tau_for(scores, alpha):
    n = len(scores)
    if n == 0:
        return float("nan")
    k = int(math.ceil((n + 1) * (1 - alpha))) - 1
    k = min(max(k, 0), n - 1)
    return float(np.sort(scores)[k])


def c3_scenes(cal, test, alpha):
    if not cal or not test:
        return dict(n_cal=0, n_test=0, tau=float("nan"), scene_success=None,
                    mean_coverage=None)
    tau = tau_for([q_high(s["score"], 1 - BETA) for s in cal], alpha)
    cov = [float(np.mean(s["e"] <= tau * s["s"])) for s in test]
    return dict(n_cal=len(cal), n_test=len(test), tau=float(tau),
                scene_success=float(np.mean([c >= 1 - BETA for c in cov])),
                mean_coverage=float(np.mean(cov)))


def cp95(k, n, delta=0.05):
    from scipy.stats import beta
    if n == 0 or k >= n:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, n - k))


def _label_from_k(kappa, vinf, branch):
    if branch == "degraded":
        return 2
    if vinf is None or not np.isfinite(kappa) or not np.isfinite(vinf):
        return 1
    return 0 if (GAMMA * float(kappa) * float(vinf) + DELTA_G) <= PHI_TOL else 1


def _collar_kappa(phi, dead_eff, est):
    """Fixed-kernel P75 on the delta_g collar inside the estimated ROI (same estimator as W4)."""
    lap = cv2.Laplacian(phi.astype(np.float32), cv2.CV_32F)
    k5 = np.ones((2 * COLLAR + 1, 2 * COLLAR + 1), np.uint8)
    dom = (cv2.dilate(dead_eff.astype(np.uint8), k5) > 0) & est
    collar = dom & (~dead_eff)
    v = np.abs(lap[collar])
    if v.size < 20:
        return float("nan"), int(v.size)
    return float(np.percentile(v, 75)), int(v.size)


def _border_invalid(dead_eff, est):
    if not dead_eff.any():
        return False
    k3 = np.ones((3, 3), np.uint8)
    grown = cv2.dilate(dead_eff.astype(np.uint8), k3) > 0
    img_border = dead_eff[0, :].any() or dead_eff[-1, :].any() or dead_eff[:, 0].any() or dead_eff[:, -1].any()
    return bool(img_border or (grown & ~est).any())


# ------------------------------------------------------------------ eval
def stage_eval(splits, seeds):
    stats = json.load(open(os.path.join(ROOT, "probes", "results", "w9_roi_stats.json")))
    est = {sp: np.load(os.path.join(EST_DIR, f"{sp}.npz"))["roi_est"] for sp in splits}
    # source arrays (shared across seeds)
    src = {}
    for split in splits:
        rows = []
        for k, s in iter_source(split):
            rows.append(s)
        src[split] = rows
    # per-scene certificate under given/estimated ROI
    cert = {}
    for split in splits:
        rows = []
        for k, s in enumerate(src[split]):
            e = est[split][k]
            dead = s["dead"]; roi = s["roi"]
            d_eff = dead & e
            k_fix = float(s["meta"]["kappa_hat_ps"][1]) if s["meta"].get("kappa_hat_ps") else float("nan")
            vinf_gt = s["meta"].get("vinf"); vinf_gt = None if vinf_gt is None else float(vinf_gt)
            y_gt = _label_from_k(k_fix, vinf_gt, s["meta"].get("branch", ""))
            # estimated-ROI certificate: vinf on retained dead, kappa on retained collar
            bad = _border_invalid(d_eff, e)
            vinf_est = None
            if d_eff.any() and not bad:
                vinf_est = torsion_vinf(d_eff, e)
            k_est, n_col = _collar_kappa(s["phi"], d_eff, e)
            if not np.isfinite(k_est):
                k_est = k_fix
            y_est = _label_from_k(k_est, vinf_est, s["meta"].get("branch", ""))
            # C1 ORR failure on the evaluated dead set
            def _orr_fail(d):
                if not d.any():
                    return False, float("nan")
                mx = float(np.max(np.abs(s["orr"][d].astype(np.float32) - s["phi"][d].astype(np.float32))))
                return bool(mx > PHI_TOL), mx
            orr_gt, mx_gt = _orr_fail(dead)
            orr_est, mx_est = _orr_fail(d_eff)
            rows.append(dict(idx=k, branch=s["meta"].get("branch", ""),
                             y_gt=int(y_gt), y_est=int(y_est),
                             k_fix=k_fix, k_est=float(k_est), n_collar=n_col,
                             vinf_gt=vinf_gt, vinf_est=None if vinf_est is None else float(vinf_est),
                             border_invalid=bool(bad),
                             n_dead=int(dead.sum()), n_dead_eff=int(d_eff.sum()),
                             dead_ret=float(d_eff.sum() / max(dead.sum(), 1)),
                             orr_fail_gt=bool(orr_gt), orr_max_gt=float(mx_gt),
                             orr_fail_est=bool(orr_est), orr_max_est=float(mx_est)))
        cert[split] = rows
    # summary of ROI / label changes
    flips = {}
    for split in splits:
        r = cert[split]
        valid = [x for x in r if x["y_gt"] in (0, 1) and x["n_dead"] > 0]
        flips[split] = dict(
            n=len(valid),
            changed=sum(int(x["y_gt"] != x["y_est"]) for x in valid),
            frac=sum(int(x["y_gt"] != x["y_est"]) for x in valid) / max(len(valid), 1),
            nP_gt=sum(int(x["y_gt"] == 0) for x in valid),
            nP_est=sum(int(x["y_est"] == 0) for x in valid),
            p_lost=sum(int(x["y_gt"] == 0 and x["y_est"] != 0) for x in valid),
            p_gained=sum(int(x["y_gt"] != 0 and x["y_est"] == 0) for x in valid),
            border_invalid=sum(int(x["border_invalid"]) for x in r),
            dead_loss_scenes=sum(int(x["dead_ret"] < 0.999) for x in r),
            empty_roi=sum(int(stats[split][x["idx"]]["empty"]) for x in r),
        )
    # C1 (ORR) per label set
    c1 = {}
    for split in splits:
        for name, ykey, fkey in (("gt", "y_gt", "orr_fail_gt"), ("est", "y_est", "orr_fail_est")):
            P = [x for x in cert[split] if x[ykey] == 0 and x["n_dead"] > 0]
            k = sum(int(x[fkey]) for x in P); n = len(P)
            c1[f"{split}_{name}"] = dict(nP=n, failures=k,
                                         fail_rate=float(k / n) if n else None,
                                         cp95=cp95(k, n))
    # C3 per seed from full float16 cache
    seed_rows = []
    for seed in seeds:
        fp = {sp: np.load(os.path.join(FULL, f"seed{seed}_{sp}.npz")) for sp in splits}
        # build the four scene lists (gt/est x val/test) once per seed
        lists = {}
        for split in ("val", "test"):
            z = fp[split]
            psi_all = z["psi"]; sig_all = z["sigma"]; idx_all = z["idx"]
            scenes = {"gt": [], "est": []}
            for i in range(len(psi_all)):
                k = int(idx_all[i])
                c = cert[split][k]
                dead = src[split][k]["dead"]
                phi = src[split][k]["phi"].astype(np.float32)
                psi = psi_all[i].astype(np.float32)
                sig = np.maximum(sig_all[i].astype(np.float32), 1e-6)
                e_img = np.abs(psi - phi)
                for name, mask, ykey in (("gt", dead, "y_gt"), ("est", dead & est[split][k], "y_est")):
                    if not mask.any():
                        continue
                    ee = e_img[mask].astype(np.float64); ss = sig[mask].astype(np.float64)
                    scenes[name].append(dict(y=int(c[ykey]), e=ee, s=ss, score=ee / ss))
            lists[split] = scenes
        rec = dict(seed=seed, grid=[])
        for al in ALPHAS:
            grid = dict(alpha=al)
            for arm in ("gt", "est"):
                cal = [s for s in lists["val"][arm] if s["y"] == 0]
                test = [s for s in lists["test"][arm] if s["y"] == 0]
                grid[arm] = c3_scenes(cal, test, al)
            rec["grid"].append(grid)
        seed_rows.append(rec)
        del fp, lists
        print(f"  seed{seed} done", flush=True)
    # summary
    summary = {}
    for i, al in enumerate(ALPHAS):
        def mv(arm, key):
            v = [r["grid"][i][arm][key] for r in seed_rows if r["grid"][i][arm][key] is not None]
            return dict(mean=float(np.mean(v)) if v else None,
                        std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                        values=[float(x) for x in v])
        summary[f"a{int(al*100):02d}"] = dict(
            gt_success=mv("gt", "scene_success"), est_success=mv("est", "scene_success"),
            gt_cov=mv("gt", "mean_coverage"), est_cov=mv("est", "mean_coverage"),
            gt_nP=mv("gt", "n_test"), est_nP=mv("est", "n_test"))
    out = dict(roi_thr=ROI_THR, gamma=GAMMA, delta_g=DELTA_G, phi_tol=PHI_TOL,
               flips=flips, c1=c1, seed_rows=seed_rows, summary=summary,
               note="Estimated-ROI arm uses full float16 predictions; ablation only, not the official C3 table.")
    json.dump(out, open(os.path.join(ROOT, "probes", "results", "w9_roi.json"), "w"), indent=1)
    print(json.dumps(dict(flips=flips, c1=c1, summary=summary), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("estimate", "eval", "both"), default="both")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args()
    splits = [x.strip() for x in a.splits.split(",")]
    if a.stage in ("estimate", "both"):
        stage_estimate(splits)
    if a.stage in ("eval", "both"):
        stage_eval(splits, [int(x) for x in a.seeds.split(",")])


if __name__ == "__main__":
    main()
