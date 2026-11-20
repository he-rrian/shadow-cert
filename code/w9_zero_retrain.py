"""w9_zero_retrain.py -- W9 zero-retrain ablations, CPU only (no GPU, no retraining).

Six pre-registered ablations of the S6 certificate / interval / risk stack:

  1. mondrian   : class-conditional (P-only) tau vs global tau on P+U
  2. rcps       : C1+C3 without the risk layer (accept all P) vs P-only RCPS
  3. pixel      : pixel-pooled tau vs scene-level tau (the 0.715 collapse)
  4. rho_gate   : no certificate gate (P+U pooled through C3/C4) vs certificate-first P
  5. kappa      : fixed-kernel SG proxy (kappa_hat_ps[1]) vs masked local-quadratic (kmap)
  6. roi        : given ROI vs single-frame estimated ROI (implemented in w9_roi.py)

Items 1-4 and 5 are implemented here; item 6 is in `w9_roi.py` because it needs the
full float16 prediction cache and a Poisson solve per scene.

Conventions
-----------
* Compact cache `data/preds_compact/seed{s}_{split}.npz` is the W8 dead-only cache.
  Its `e`/`sigma` are float16-derived (the official C3 table remains the float32
  `w8_p_only/` run; this script never overwrites it).
* Labels are recomputed from `y75` (the official gamma=0.75 P/U labels).
* `phi_tol = pi/10`, `delta_g = 0.13`, `beta = 0.05` (interval target 95%).
* All numbers are seed-wise; summaries report mean +/- std over seeds.

Run:
  python code/w9_zero_retrain.py --items 1,2,3,4
  python code/w9_zero_retrain.py --items 5
  python code/w9_zero_retrain.py --items 1,2,3,4,5 --out probes/results/w9_zero_retrain.json
"""
import os, sys, json, math, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from w8_conformal import q_high, tau_for
from w8_rcps import curves as rcps_curves

PHI_TOL = math.pi / 10.0
DELTA_G = 0.13
GAMMA = 0.75
BETA = 0.05
ALPHAS = (0.05, 0.10, 0.20)
COLLAR = 2
COMPACT = os.path.join(ROOT, "data", "preds_compact")


# ------------------------------------------------------------------ data
def load_compact(seed, split):
    z = np.load(os.path.join(COMPACT, f"seed{seed}_{split}.npz"))
    n = z["n_dead"].astype(int); off = np.r_[0, np.cumsum(n)]
    e = z["e"].astype(np.float64); s = np.maximum(z["sigma"].astype(np.float64), 1e-6)
    y = z["y75"].astype(int); sid = z["sid"].astype(int); idx = z["idx"].astype(int)
    br = z["branch_id"].astype(int)
    scenes = []
    for i in range(len(n)):
        ee = e[off[i]:off[i + 1]]; ss = s[off[i]:off[i + 1]]
        scenes.append(dict(idx=int(idx[i]), sid=int(sid[i]), branch=int(br[i]), y=int(y[i]),
                           e=ee, s=ss, score=ee / ss))
    return scenes


def load_source_meta(split):
    man = json.load(open(os.path.join(ROOT, "data", split, "manifest_000.json")))
    metas = []
    for sh in man["shards"]:
        z = np.load(os.path.join(ROOT, "data", split, sh["path"]))
        metas.extend([json.loads(m) for m in z["meta"]])
    return metas


# ------------------------------------------------------------------ statistics
def scene_scores(scenes):
    return [q_high(s["score"], 1 - BETA) for s in scenes]


def c3_scenes(cal, test, alpha):
    """Two-level conformal: scene quantile score -> tau -> scene success."""
    if not cal or not test:
        return dict(n_cal=0, n_test=0, tau=float("nan"), scene_success=None,
                    mean_coverage=None, median_width=None)
    tau = tau_for(scene_scores(cal), alpha)
    cov = [float(np.mean(s["e"] <= tau * s["s"])) for s in test]
    width = [2.0 * tau * float(np.median(s["s"])) for s in test]
    return dict(n_cal=len(cal), n_test=len(test), tau=float(tau),
                scene_success=float(np.mean([c >= 1 - BETA for c in cov])),
                mean_coverage=float(np.mean(cov)),
                median_width=float(np.median(width)))


def c3_pixel(cal, test, alpha):
    """Pixel-pooled split conformal: one tau over all dead pixels in cal."""
    if not cal or not test:
        return dict(n_cal_px=0, n_test=0, tau=float("nan"), scene_success=None,
                    mean_coverage=None)
    pool = np.concatenate([s["score"] for s in cal])
    tau = tau_for(pool, alpha)
    cov = [float(np.mean(s["e"] <= tau * s["s"])) for s in test]
    return dict(n_cal_px=int(len(pool)), n_test=len(test), tau=float(tau),
                scene_success=float(np.mean([c >= 1 - BETA for c in cov])),
                mean_coverage=float(np.mean(cov)))


def accept_stats(test, lam=None, alpha=0.05):
    """Accepted-pixel risk for a given sigma threshold (None = accept all)."""
    tot_acc = tot_fail = 0.0
    per_scene = []
    by_class = {0: [0.0, 0.0], 1: [0.0, 0.0], 2: [0.0, 0.0]}  # accepted, failures
    for s in test:
        acc = np.ones_like(s["e"], bool) if lam is None else (s["s"] <= lam)
        fail = acc & (s["e"] > PHI_TOL)
        a = float(acc.sum()); f = float(fail.sum())
        tot_acc += a; tot_fail += f
        per_scene.append(dict(n_acc=int(a), n_fail=int(f), acc_frac=float(acc.mean()),
                              risk=float(f / a) if a > 0 else None))
        if s["y"] in by_class:
            by_class[s["y"]][0] += a; by_class[s["y"]][1] += f
    scene_any = [r for r in per_scene if r["n_acc"] > 0 and r["risk"] is not None]
    return dict(lam=None if lam is None else float(lam),
                acc_frac=float(np.mean([r["acc_frac"] for r in per_scene])) if per_scene else None,
                pooled_risk=float(tot_fail / tot_acc) if tot_acc > 0 else None,
                scene_any_risk=float(np.mean([r["risk"] for r in scene_any])) if scene_any else None,
                scene_any_fail=float(np.mean([r["n_fail"] > 0 for r in scene_any])) if scene_any else None,
                n_acc=int(tot_acc), n_fail=int(tot_fail),
                by_class={str(k): dict(acc=int(v[0]), fail=int(v[1]),
                                       risk=float(v[1] / v[0]) if v[0] > 0 else None)
                          for k, v in by_class.items()})


def mean_std(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return dict(mean=None, std=None, values=[])
    return dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                values=[float(x) for x in v])


# ------------------------------------------------------------------ item 1
def item_mondrian(seeds):
    """P-only tau vs global (P+U) tau; evaluate on P and on all."""
    rows = []
    for seed in seeds:
        val = load_compact(seed, "val"); test = load_compact(seed, "test")
        Pval = [s for s in val if s["y"] == 0]; Ptest = [s for s in test if s["y"] == 0]
        rec = dict(seed=seed, grid=[])
        for al in ALPHAS:
            p = c3_scenes(Pval, Ptest, al)
            gP = c3_scenes(val, Ptest, al)
            gA = c3_scenes(val, test, al)
            rec["grid"].append(dict(alpha=al, p_only=p, global_evalP=gP, global_evalAll=gA))
        rows.append(rec)
    summary = {}
    for i, al in enumerate(ALPHAS):
        summary[f"a{int(al*100):02d}"] = dict(
            p_only=mean_std([r["grid"][i]["p_only"]["scene_success"] for r in rows]),
            global_evalP=mean_std([r["grid"][i]["global_evalP"]["scene_success"] for r in rows]),
            global_evalAll=mean_std([r["grid"][i]["global_evalAll"]["scene_success"] for r in rows]),
            tau_p=mean_std([r["grid"][i]["p_only"]["tau"] for r in rows]),
            tau_g=mean_std([r["grid"][i]["global_evalP"]["tau"] for r in rows]),
            width_p=mean_std([r["grid"][i]["p_only"]["median_width"] for r in rows]),
            width_g=mean_std([r["grid"][i]["global_evalP"]["median_width"] for r in rows]),
            nP=mean_std([r["grid"][i]["p_only"]["n_test"] for r in rows]),
        )
    return dict(item="mondrian", seeds=list(seeds), rows=rows, summary=summary)


# ------------------------------------------------------------------ item 2
def item_rcps(seeds):
    """No risk layer (accept all P) vs P-only RCPS; negative control, not evidence C4 works."""
    rows = []
    for seed in seeds:
        val = load_compact(seed, "val"); test = load_compact(seed, "test")
        Pval = [s for s in val if s["y"] == 0]; Ptest = [s for s in test if s["y"] == 0]
        rec = dict(seed=seed, grid=[])
        nP_dead = float(sum(len(s["e"]) for s in Ptest))
        n_all_dead = float(sum(len(s["e"]) for s in test))
        for al in ALPHAS:
            no = accept_stats(Ptest, lam=None, alpha=al)
            r = rcps_curves(Pval, Ptest, al) if Pval else None
            with_rcps = None
            if r is not None:
                with_rcps = dict(lam=r["chosen_test"]["lam"], cal_lam=r["chosen"]["lam"],
                                 cal_acc=r["chosen"]["acc_frac"], cal_risk=r["chosen"]["r_pix"],
                                 cal_u_run=r["chosen"]["u_run"],
                                 test=accept_stats(Ptest, lam=r["chosen_test"]["lam"], alpha=al))
            rec["grid"].append(dict(alpha=al, no_rcps=no,
                                    no_rcps_overall_acc=float(nP_dead / max(n_all_dead, 1e-9)),
                                    with_rcps=with_rcps))
        rows.append(rec)
    summary = {}
    for i, al in enumerate(ALPHAS):
        summary[f"a{int(al*100):02d}"] = dict(
            no_rcps_acc=mean_std([r["grid"][i]["no_rcps_overall_acc"] for r in rows]),
            no_rcps_risk=mean_std([r["grid"][i]["no_rcps"]["pooled_risk"] for r in rows]),
            no_rcps_any=mean_std([r["grid"][i]["no_rcps"]["scene_any_fail"] for r in rows]),
            rcps_acc=mean_std([r["grid"][i]["with_rcps"]["test"]["acc_frac"] if r["grid"][i]["with_rcps"] else None for r in rows]),
            rcps_risk=mean_std([r["grid"][i]["with_rcps"]["test"]["pooled_risk"] if r["grid"][i]["with_rcps"] else None for r in rows]),
            rcps_any=mean_std([r["grid"][i]["with_rcps"]["test"]["scene_any_fail"] if r["grid"][i]["with_rcps"] else None for r in rows]),
            rcps_u_run=mean_std([r["grid"][i]["with_rcps"]["cal_u_run"] if r["grid"][i]["with_rcps"] else None for r in rows]),
        )
    return dict(item="rcps", seeds=list(seeds), rows=rows, summary=summary)


# ------------------------------------------------------------------ item 3
def item_pixel(seeds):
    """Pixel-pooled vs scene-level conformal, all-class and P-only."""
    rows = []
    for seed in seeds:
        val = load_compact(seed, "val"); test = load_compact(seed, "test")
        Pval = [s for s in val if s["y"] == 0]; Ptest = [s for s in test if s["y"] == 0]
        rec = dict(seed=seed, grid=[])
        for al in ALPHAS:
            rec["grid"].append(dict(alpha=al,
                                    all_scene=c3_scenes(val, test, al),
                                    all_pixel=c3_pixel(val, test, al),
                                    p_scene=c3_scenes(Pval, Ptest, al),
                                    p_pixel=c3_pixel(Pval, Ptest, al)))
        rows.append(rec)
    summary = {}
    for i, al in enumerate(ALPHAS):
        summary[f"a{int(al*100):02d}"] = dict(
            all_scene=mean_std([r["grid"][i]["all_scene"]["scene_success"] for r in rows]),
            all_pixel=mean_std([r["grid"][i]["all_pixel"]["scene_success"] for r in rows]),
            p_scene=mean_std([r["grid"][i]["p_scene"]["scene_success"] for r in rows]),
            p_pixel=mean_std([r["grid"][i]["p_pixel"]["scene_success"] for r in rows]),
        )
    return dict(item="pixel", seeds=list(seeds), rows=rows, summary=summary)


# ------------------------------------------------------------------ item 4
def item_rho_gate(seeds):
    """No certificate gate: P+U pooled through C3 and C4."""
    rows = []
    for seed in seeds:
        val = load_compact(seed, "val"); test = load_compact(seed, "test")
        Pval = [s for s in val if s["y"] == 0]; Ptest = [s for s in test if s["y"] == 0]
        rec = dict(seed=seed, grid=[])
        for al in ALPHAS:
            no_gate_c3 = c3_scenes(val, test, al)
            p_c3 = c3_scenes(Pval, Ptest, al)
            no_gate_rcps = rcps_curves(val, test, al)
            p_rcps = rcps_curves(Pval, Ptest, al)
            no_gate_test = accept_stats(test, lam=no_gate_rcps["chosen_test"]["lam"], alpha=al)
            p_test = accept_stats(Ptest, lam=p_rcps["chosen_test"]["lam"], alpha=al)
            rec["grid"].append(dict(alpha=al,
                                    c3_all=no_gate_c3, c3_P=p_c3,
                                    c4_all=dict(cal=no_gate_rcps["chosen"],
                                                test=no_gate_test),
                                    c4_P=dict(cal=p_rcps["chosen"], test=p_test)))
        rows.append(rec)
    summary = {}
    for i, al in enumerate(ALPHAS):
        summary[f"a{int(al*100):02d}"] = dict(
            c3_all=mean_std([r["grid"][i]["c3_all"]["scene_success"] for r in rows]),
            c3_P=mean_std([r["grid"][i]["c3_P"]["scene_success"] for r in rows]),
            c4_all_acc=mean_std([r["grid"][i]["c4_all"]["test"]["acc_frac"] for r in rows]),
            c4_all_risk=mean_std([r["grid"][i]["c4_all"]["test"]["pooled_risk"] for r in rows]),
            c4_P_acc=mean_std([r["grid"][i]["c4_P"]["test"]["acc_frac"] for r in rows]),
            c4_P_risk=mean_std([r["grid"][i]["c4_P"]["test"]["pooled_risk"] for r in rows]),
            c4_all_U_acc=mean_std([r["grid"][i]["c4_all"]["test"]["by_class"]["1"]["acc"] for r in rows]),
            c4_all_U_risk=mean_std([r["grid"][i]["c4_all"]["test"]["by_class"]["1"]["risk"] for r in rows]),
            c4_all_P_risk=mean_std([r["grid"][i]["c4_all"]["test"]["by_class"]["0"]["risk"] for r in rows]),
        )
    return dict(item="rho_gate", seeds=list(seeds), rows=rows, summary=summary)


# ------------------------------------------------------------------ item 5
def _collar(dead, roi):
    import cv2
    k = np.ones((2 * COLLAR + 1, 2 * COLLAR + 1), np.uint8)
    dom = (cv2.dilate(dead.astype(np.uint8), k) > 0) & roi
    return dom & (~dead)


def _mlq_p75(kmap, collar):
    v = np.abs(kmap[collar & np.isfinite(kmap) & (kmap != 0)].astype(np.float64))
    if v.size < 20:
        return float("nan"), int(v.size)
    return float(np.percentile(v, 75)), int(v.size)


def _cp95(k, n, delta=0.05):
    from scipy.stats import beta
    if n == 0 or k >= n:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, n - k))


def load_kappa_table(split):
    """Per-scene kappa (fixed-kernel P75 vs MLQ P75) and ORR failure, source order."""
    man = json.load(open(os.path.join(ROOT, "data", split, "manifest_000.json")))
    out = []
    for sh in man["shards"]:
        z = np.load(os.path.join(ROOT, "data", split, sh["path"]))
        phi = z["phi"].astype(np.float32); dead = z["dead"].astype(bool)
        roi = z["roi"].astype(bool); orr = z["orr"].astype(np.float32)
        kmap = z["kmap"].astype(np.float32); meta = z["meta"]
        for i in range(len(phi)):
            m = json.loads(meta[i])
            d = dead[i]; r = roi[i]
            collar = _collar(d, r)
            k_mlq, n_col = _mlq_p75(kmap[i], collar)
            k_fix = float(m["kappa_hat_ps"][1]) if m.get("kappa_hat_ps") else float("nan")
            vinf = m.get("vinf"); branch = m.get("branch", "")
            # C1: ORR max error on GT dead
            if d.any():
                max_err = float(np.max(np.abs(orr[i][d].astype(np.float32) - phi[i][d])))
                orr_fail = bool(max_err > PHI_TOL)
            else:
                max_err = float("nan"); orr_fail = False
            out.append(dict(idx=len(out), branch=branch, vinf=None if vinf is None else float(vinf),
                            k_fix=k_fix, k_mlq=k_mlq, n_collar=n_col,
                            orr_fail=orr_fail, max_err=max_err,
                            roi_n=int(r.sum()), dead_n=int(d.sum())))
    return out


def _label(kappa, m):
    if m["branch"] == "degraded":
        return 2
    if m["vinf"] is None or not np.isfinite(kappa):
        return 1
    bound = GAMMA * float(kappa) * float(m["vinf"]) + DELTA_G
    return 0 if bound <= PHI_TOL else 1


def item_kappa(seeds):
    """Fixed-kernel SG proxy (kappa_hat_ps[1]) vs masked local quadratic (kmap P75)."""
    tabs = {sp: load_kappa_table(sp) for sp in ("val", "test")}
    # label flips among certificate-eligible scenes (vinf is not None); degraded
    # scenes have no dead zone and are absent from the compact cache.
    flips = {}
    for sp, tab in tabs.items():
        n = 0; ch = 0; nP_fix = nP_mlq = 0; fallback = 0
        for m in tab:
            if m["vinf"] is None or m["branch"] == "degraded":
                continue
            yf = _label(m["k_fix"], m); ym = _label(m["k_mlq"], m)
            n += 1; ch += int(yf != ym); nP_fix += int(yf == 0); nP_mlq += int(ym == 0)
            fallback += int(not np.isfinite(m["k_mlq"]))
        flips[sp] = dict(n=n, changed=ch, frac=ch / max(n, 1),
                         nP_fixed=nP_fix, nP_mlq=nP_mlq, fallback=fallback)
    # C1 per label set
    c1 = {}
    for sp, tab in tabs.items():
        for name, key in (("fixed", "k_fix"), ("mlq", "k_mlq")):
            P = [m for m in tab if _label(m[key], m) == 0]
            k = sum(int(m["orr_fail"]) for m in P); n = len(P)
            c1[f"{sp}_{name}"] = dict(nP=n, failures=k,
                                      fail_rate=float(k / n) if n else None,
                                      cp95=_cp95(k, n))
    # C3 per seed under each label set
    rows = []
    for seed in seeds:
        v = load_compact(seed, "val"); t = load_compact(seed, "test")
        vmeta = tabs["val"]; tmeta = tabs["test"]
        for s in v:
            m = vmeta[s["idx"]]
            s["y_fix"] = _label(m["k_fix"], m)
            s["y_mlq"] = _label(m["k_mlq"], m)
        for s in t:
            m = tmeta[s["idx"]]
            s["y_fix"] = _label(m["k_fix"], m)
            s["y_mlq"] = _label(m["k_mlq"], m)
        rec = dict(seed=seed, grid=[])
        for al in ALPHAS:
            rec["grid"].append(dict(alpha=al,
                                    fixed=c3_scenes([s for s in v if s["y_fix"] == 0],
                                                    [s for s in t if s["y_fix"] == 0], al),
                                    mlq=c3_scenes([s for s in v if s["y_mlq"] == 0],
                                                  [s for s in t if s["y_mlq"] == 0], al)))
        rows.append(rec)
    summary = {}
    for i, al in enumerate(ALPHAS):
        summary[f"a{int(al*100):02d}"] = dict(
            fixed=mean_std([r["grid"][i]["fixed"]["scene_success"] for r in rows]),
            mlq=mean_std([r["grid"][i]["mlq"]["scene_success"] for r in rows]),
            fixed_nP=mean_std([r["grid"][i]["fixed"]["n_test"] for r in rows]),
            mlq_nP=mean_std([r["grid"][i]["mlq"]["n_test"] for r in rows]),
        )
    return dict(item="kappa", seeds=list(seeds), flips=flips, c1=c1,
                rows=rows, summary=summary,
                note="kappa_hat_ps[1] is a fixed-kernel Laplacian (SG proxy), NOT a real SG7 estimator.")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="1,2,3,4,5")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default=os.path.join(ROOT, "probes", "results", "w9_zero_retrain.json"))
    a = ap.parse_args()
    items = [x.strip() for x in a.items.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",")]
    out = dict(phi_tol=PHI_TOL, delta_g=DELTA_G, gamma=GAMMA, beta=BETA,
               alphas=list(ALPHAS), items={})
    if "1" in items:
        print("item 1 mondrian ...", flush=True)
        out["items"]["mondrian"] = item_mondrian(seeds)
    if "2" in items:
        print("item 2 rcps ...", flush=True)
        out["items"]["rcps"] = item_rcps(seeds)
    if "3" in items:
        print("item 3 pixel ...", flush=True)
        out["items"]["pixel"] = item_pixel(seeds)
    if "4" in items:
        print("item 4 rho_gate ...", flush=True)
        out["items"]["rho_gate"] = item_rho_gate(seeds)
    if "5" in items:
        print("item 5 kappa ...", flush=True)
        out["items"]["kappa"] = item_kappa(seeds)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
