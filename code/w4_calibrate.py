"""W4 -- baseline table + joint (gamma, delta_g) calibration on val/test.

Protocol fixed by the project lead on 2026-11-17 (do not change silently):

  A-all (diagnostic of kappa_hat, NOT the deployment metric)
      numerator   : per-pixel MAX |error| of the deployable estimator (Harmonic on the
                    observed boundary) over the dead zone  -- NEVER the mean;
      denominator : ALL samples that carry `vinf` (750 per val/test split);
      violation   : max_err > gamma * kappa_hat * ||v||_inf + delta_g .
      This is the "is kappa_hat globally too small?" curve.  It is reported as a diagnostic
      because it mixes in samples that would never be certified.

  A-cert (MAIN, certificate soundness on the issued set)
      denominator : n_cert (the certified set); same numerator as A-all.
      gamma*_A    : smallest gamma whose 95% CP upper bound is <= eps (with a margin for
                    the 5% main setting; see w4_decide.py).

  B (appendix, the promise to the user)
      certified   : gamma * kappa_hat * ||v||_inf + delta_g <= phi_tol ;
      failure     : max_err > phi_tol ;
      denominator : n_cert -- n_cert MUST always be reported, otherwise a large gamma
                    shrinks the set and the rate looks like 0.

  Every rate carries a one-sided 95% Clopper-Pearson upper bound (n = samples, not pixels).
  C_5pt_max = max|ORR - phi| / (max|Delta_5pt phi| * ||v||) is reported on the CERTIFIED set
  (a theorem-level soundness check), and on all vinf samples only for contrast.

The fill implementations are imported from w4_baseline.py, which stays frozen as the reference;
`--regression` reproduces its 64-sample smoke numbers of 2026-11-17.

Usage
  python w4_calibrate.py eval --split val --delta-g 0.13
  python w4_calibrate.py eval --split val --delta-g 0.05 --methods harmonic --vinf-only
  python w4_calibrate.py eval --split test --delta-g 0.13
  python w4_calibrate.py report
  python w4_calibrate.py regression
"""
import os, sys, json, time, argparse, importlib.util
import numpy as np
import cv2
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "probes", "results")

PHI_TOL = float(np.pi / 10.0)          # 0.314159 rad, frozen by 题目2_定义与实施前置
DELTA_G = 0.13                          # probe-A P95, frozen in the S3 labels
COLLAR = 2
GAMMA_GRID = np.round(np.arange(0.0, 6.0001, 0.05), 4)
N_DENOM = 750                           # samples with vinf per val/test split
N_MIN_CERT = 100                        # B-calibration must not hide behind a tiny cert set
ALPHA_CP = 0.05
METHODS = ["Identity", "Nearest", "Harmonic", "Biharm", "Poly2", "ORR"]

# w4_baseline.py smoke run, 2026-11-17 (64 val samples, delta_g=0.13), printed values.
REF_SMOKE = {
    "Identity": (20.7211, 325.569),
    "Nearest":  (1.4723, 19.942),
    "Harmonic": (0.1027, 1.163),
    "Biharm":   (0.1187, 1.535),
    "Poly2":    (0.9208, 13.910),
    "ORR":      (0.0980, 1.000),
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


W4 = load_module("w4_ref", os.path.join(HERE, "w4_baseline.py"))


# ----------------------------------------------------------------------------- stats helpers
def cp_upper(k, n, alpha=ALPHA_CP):
    """one-sided 95% Clopper-Pearson upper bound for k/n."""
    if n <= 0:
        return float("nan")
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, k + 1.0, n - k))


def cp_interval(k, n, alpha=ALPHA_CP):
    """two-sided 95% Clopper-Pearson interval."""
    if n <= 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lo, hi)


# ----------------------------------------------------------------------------- evaluation
def shard_files(split):
    man = json.load(open(os.path.join(DATA, split, "manifest_000.json")))
    return [os.path.join(DATA, split, s["path"]) for s in man["shards"]]


def eval_one(phi, orr, dead, roi, m, delta_g, methods, seed):
    """compute per-method pixel errors and certificate covariates for one sample."""
    rng = np.random.default_rng(seed)
    src = phi + rng.normal(0.0, delta_g / 2.0, phi.shape) * (~dead)
    fills = {}
    if methods == "all":
        fills["Identity"] = W4.identity_fill(dead, src)
        fills["Nearest"] = W4.nearest_fill(dead, src)
        fills["Harmonic"] = W4.harmonic_fill(dead, src)
        try:
            fills["Biharm"] = W4.biharmonic_fill(dead, src)
        except Exception:
            pass
        fills["Poly2"] = W4.poly_fill(dead, src)
    else:
        fills["Harmonic"] = W4.harmonic_fill(dead, src)
    fills["ORR"] = orr
    err = {}
    for k, arr in fills.items():
        e = np.abs(np.asarray(arr, np.float64)[dead] - phi[dead])
        err[k] = dict(mean=float(e.mean()), p95=float(np.percentile(e, 95)),
                      max=float(e.max()), frac2pi=float((e > 2.0 * np.pi).mean()))
    rel = {k: (err[k]["mean"] / max(err["ORR"]["mean"], 1e-9)) for k in err}
    lap = cv2.Laplacian(phi, cv2.CV_64F)
    k_max = float(np.abs(lap[dead]).max())
    dom = (cv2.dilate(dead.astype(np.uint8), np.ones((2 * COLLAR + 1,) * 2, np.uint8)) > 0) & roi
    k_max_dom = float(np.abs(lap[dom]).max()) if dom.any() else float("nan")
    vinf = m.get("vinf")
    c5 = (err["ORR"]["max"] / (k_max * vinf)) if (vinf and k_max > 0) else None
    c5_dom = (err["ORR"]["max"] / (k_max_dom * vinf)) if (vinf and k_max_dom > 0) else None
    return dict(id=m["id"], branch=m["branch"], cls=m["cls"], area=int(dead.sum()),
                npx=int(dead.sum()), vinf=None if vinf is None else float(vinf),
                k_or=float(m["kappa_oracle"]), k_hat=float(m["kappa_hat"]),
                k_hat_p90=float(m["kappa_hat_ps"][2]) if len(m.get("kappa_hat_ps", [])) > 2 else None,
                has_vinf=vinf is not None, k_max=k_max, k_max_dom=k_max_dom,
                c5=None if c5 is None else float(c5),
                c5_dom=None if c5_dom is None else float(c5_dom),
                delta_g=float(delta_g), seed=int(seed), err=err, rel=rel)


def eval_split(split, delta_g, methods="all", vinf_only=False, out=None, n_max=0,
               cls_filter=None):
    rows, t0 = [], time.time()
    for si, path in enumerate(shard_files(split)):
        d = np.load(path)
        n = len(d["meta"])
        for i in range(n):
            m = json.loads(d["meta"][i])
            dead = np.asarray(d["dead"][i], bool)
            if dead.sum() < 30:
                continue
            if vinf_only and m.get("vinf") is None:
                continue
            if cls_filter and m.get("cls") != cls_filter:
                continue
            phi = np.asarray(d["phi"][i], np.float64)
            orr = np.asarray(d["orr"][i], np.float64)
            roi = np.asarray(d["roi"][i], bool)
            r = eval_one(phi, orr, dead, roi, m, delta_g, methods, seed=i)
            r["split"] = split
            r["shard"] = si
            r["idx"] = i
            rows.append(r)
            if len(rows) % 100 == 0:
                print(f"  [{split} dg={delta_g:.2f}] {len(rows)} rows  "
                      f"{time.time()-t0:.0f}s", flush=True)
            if n_max and len(rows) >= n_max:
                break
        del d
        if n_max and len(rows) >= n_max:
            break
    out = out or os.path.join(RES, f"w4_rows_{split}_dg{delta_g:.2f}.json")
    json.dump(rows, open(out, "w"), indent=0, default=float)
    print(f"saved {out}  ({len(rows)} rows, {time.time()-t0:.0f}s)", flush=True)
    return rows


# ----------------------------------------------------------------------------- calibration
def _arrays(rows, kappa_key="k_hat"):
    v = np.array([r["vinf"] for r in rows], float)
    kh = np.array([r[kappa_key] if r[kappa_key] is not None else np.nan for r in rows], float)
    emax = np.array([r["err"]["Harmonic"]["max"] for r in rows], float)
    ok = np.isfinite(v) & np.isfinite(kh) & np.isfinite(emax)
    return v[ok], kh[ok], emax[ok]


def curve_A(rows, delta_g, kappa_key="k_hat"):
    v, kh, emax = _arrays(rows, kappa_key)
    out = []
    for g in GAMMA_GRID:
        bound = g * kh * v + delta_g
        k = int(np.sum(emax > bound)); n = len(emax)
        out.append(dict(gamma=float(g), k=k, n=n, rate=k / n if n else None,
                        cp=cp_upper(k, n)))
    return out


def gamma_star_A(curve, eps):
    for rec in curve:
        if rec["rate"] is not None and rec["rate"] <= eps:
            return rec
    return None


def curve_B(rows, delta_g):
    v, kh, emax = _arrays(rows, "k_hat")
    out = []
    for g in GAMMA_GRID:
        bound = g * kh * v + delta_g
        cert = bound <= PHI_TOL
        n_cert = int(cert.sum())
        k = int(np.sum(cert & (emax > PHI_TOL)))
        out.append(dict(gamma=float(g), n_cert=n_cert, k=k,
                        rate=(k / n_cert if n_cert else None),
                        cp=(cp_upper(k, n_cert) if n_cert else None)))
    return out


def gamma_star_B(curve, eps, n_min=N_MIN_CERT):
    for rec in curve:
        if rec["n_cert"] >= n_min and rec["rate"] is not None and rec["rate"] <= eps:
            return rec
    return None


def cert_stats(rows, gamma, delta_g, kappa_key="k_hat"):
    v, kh, _ = _arrays(rows, kappa_key)
    bound = gamma * kh * v + delta_g
    k = int(np.sum(bound <= PHI_TOL)); n = len(bound)
    lo, hi = cp_interval(k, n)
    return dict(gamma=float(gamma), n_cert=k, n=n, rate=k / n if n else None,
                lo=lo, hi=hi)


def viol_A_at(rows, gamma, delta_g):
    v, kh, emax = _arrays(rows, "k_hat")
    k = int(np.sum(emax > gamma * kh * v + delta_g)); n = len(emax)
    return dict(gamma=float(gamma), k=k, n=n, rate=k / n if n else None, cp=cp_upper(k, n))


def viol_B_at(rows, gamma, delta_g):
    v, kh, emax = _arrays(rows, "k_hat")
    bound = gamma * kh * v + delta_g
    cert = bound <= PHI_TOL
    n_cert = int(cert.sum()); k = int(np.sum(cert & (emax > PHI_TOL)))
    return dict(gamma=float(gamma), n_cert=n_cert, k=k,
                rate=(k / n_cert if n_cert else None),
                cp=(cp_upper(k, n_cert) if n_cert else None))


def c5_stats(rows, gamma, delta_g):
    vals = []
    for r in rows:
        if not r["has_vinf"] or r.get("c5") is None:
            continue
        bound = gamma * r["k_hat"] * r["vinf"] + delta_g
        if bound <= PHI_TOL:
            vals.append(r["c5"])
    vals = np.array(vals, float)
    if len(vals) == 0:
        return dict(n=0, median=None, p95=None, max=None, frac_gt1=None, cp=None)
    k = int(np.sum(vals > 1.0)); n = len(vals)
    return dict(n=n, median=float(np.median(vals)), p95=float(np.percentile(vals, 95)),
                max=float(vals.max()), frac_gt1=k / n, cp=cp_upper(k, n))


# ----------------------------------------------------------------------------- tables
def _med(rows, meth, stat):
    vals = [r["err"][meth][stat] for r in rows if meth in r["err"]]
    return float(np.median(vals)) if vals else None


def _rel_med(rows, meth):
    vals = [r["rel"][meth] for r in rows if meth in r["rel"]]
    return float(np.median(vals)) if vals else None


def _pix2pi(rows, meth):
    num = sum(r["err"][meth]["frac2pi"] * r["npx"] for r in rows if meth in r["err"])
    den = sum(r["npx"] for r in rows if meth in r["err"])
    return float(num / den) if den else None


def baseline_table(rows):
    groups = [("interior (vinf, n=%d)" % sum(1 for r in rows if r["has_vinf"]),
               [r for r in rows if r["has_vinf"]]),
              ("border-touching (n=%d)" % sum(1 for r in rows if not r["has_vinf"]),
               [r for r in rows if not r["has_vinf"]]),
              ("all holes (n=%d)" % len(rows), rows)]
    out = {}
    for gname, sel in groups:
        rec = {}
        for meth in METHODS:
            rr = [r for r in sel if meth in r["err"]]
            if not rr:
                continue
            rec[meth] = dict(n=len(rr), mean_med=_med(rr, meth, "mean"),
                             p95_med=_med(rr, meth, "p95"), max_med=_med(rr, meth, "max"),
                             rel_med=_rel_med(rr, meth), pix2pi=_pix2pi(rr, meth),
                             samp2pi=float(np.mean(
                                 [r["err"][meth]["max"] > 2.0 * np.pi for r in rr])))
        out[gname] = rec
    return out


def fmt(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def pct(x, nd=2):
    return "—" if x is None else f"{100.0*x:.{nd}f}%"


def print_baseline(bt):
    for gname, rec in bt.items():
        print(f"\n  [{gname}]")
        print(f"  {'method':10s}{'mean(med)':>11s}{'P95(med)':>10s}{'max(med)':>10s}"
              f"{'Rel-ORR':>9s}{'px>2pi':>8s}{'samp>2pi':>9s}")
        for meth in METHODS:
            if meth not in rec:
                continue
            d = rec[meth]
            print(f"  {meth:10s}{fmt(d['mean_med']):>11s}{fmt(d['p95_med']):>10s}"
                  f"{fmt(d['max_med']):>10s}{fmt(d['rel_med'], 3):>9s}"
                  f"{pct(d['pix2pi']):>8s}{pct(d['samp2pi']):>9s}")


def print_calA(cal):
    print(f"\n  [A-all diagnostic: denominator = {N_DENOM} vinf samples, numerator = max err > bound]")
    print(f"  {'eps':>5s}{'gamma*':>8s}{'k':>5s}{'n':>5s}{'rate':>8s}{'CP95 up':>9s}"
          f"{'cert@gamma*':>12s}{'cert rate':>10s}")
    for eps, rec in cal.items():
        eps = float(eps)
        if rec is None:
            print(f"  {eps:5.2f}{'—':>8s}")
            continue
        c = rec["cert"]
        print(f"  {eps:5.2f}{rec['gamma']:8.2f}{rec['k']:5d}{rec['n']:5d}"
              f"{pct(rec['rate']):>8s}{pct(rec['cp']):>9s}{c['n_cert']:12d}{pct(c['rate']):>10s}")


def print_calB(cal):
    print(f"\n  [B: certified = bound <= phi_tol; failure = max err > phi_tol]")
    print(f"  {'eps':>5s}{'gamma*':>8s}{'n_cert':>8s}{'k':>5s}{'rate':>8s}{'CP95 up':>9s}")
    for eps, rec in cal.items():
        eps = float(eps)
        if rec is None:
            print(f"  {eps:5.2f}{'—':>8s}   (no gamma with n_cert >= {N_MIN_CERT})")
            continue
        print(f"  {eps:5.2f}{rec['gamma']:8.2f}{rec['n_cert']:8d}{rec['k']:5d}"
              f"{pct(rec['rate']):>8s}{pct(rec['cp']):>9s}")


def print_cert(name, rec):
    print(f"  {name:28s} n_cert={rec['n_cert']:4d}/{rec['n']:<4d} "
          f"rate={pct(rec['rate']):>7s}  95%CI=[{pct(rec['lo'])}, {pct(rec['hi'])}]")


def print_c5(name, rec):
    print(f"  {name:28s} n={rec['n']:4d} med={fmt(rec['median'],3)} P95={fmt(rec['p95'],3)} "
          f"max={fmt(rec['max'],3)} frac>1={pct(rec['frac_gt1'])} CP95={pct(rec['cp'])}")


def build_report(val, test, val_dg05=None, val_dg02=None):
    eps_list = [0.05, 0.10, 0.20]
    calA, calB = {}, {}
    for eps in eps_list:
        recA = gamma_star_A(curve_A(val, DELTA_G), eps)
        if recA is not None:
            recA = dict(recA); recA["cert"] = cert_stats(val, recA["gamma"], DELTA_G)
        calA[eps] = recA
        calB[eps] = gamma_star_B(curve_B(val, DELTA_G), eps)

    # curves (selected gamma values) for the paper figure/table
    curveA = curve_A(val, DELTA_G)
    curveB = curve_B(val, DELTA_G)
    selected = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    curveA_sel = [r for r in curveA if round(r["gamma"], 2) in selected]
    curveB_sel = [r for r in curveB if round(r["gamma"], 2) in selected]

    out = dict(
        protocol=dict(phi_tol=PHI_TOL, delta_g=DELTA_G, n_denom=N_DENOM,
                      n_min_cert=N_MIN_CERT, alpha_cp=ALPHA_CP,
                      gamma_grid=[float(GAMMA_GRID[0]), float(GAMMA_GRID[-1]),
                                  float(GAMMA_GRID[1] - GAMMA_GRID[0])]),
        counts=dict(val=len(val), val_vinf=sum(r["has_vinf"] for r in val),
                    test=len(test), test_vinf=sum(r["has_vinf"] for r in test)),
        baseline=baseline_table(val),
        calA={str(e): (None if r is None else r) for e, r in calA.items()},
        calB={str(e): (None if r is None else r) for e, r in calB.items()},
        curveA=curveA_sel, curveB=curveB_sel,
        cert_val=dict(gamma1=cert_stats(val, 1.0, DELTA_G),
                      gammaA10=cert_stats(val, calA[0.10]["gamma"], DELTA_G)),
        cert_test=dict(gamma1=cert_stats(test, 1.0, DELTA_G),
                       gammaA10=cert_stats(test, calA[0.10]["gamma"], DELTA_G)),
        viol_val_A=viol_A_at(val, calA[0.10]["gamma"], DELTA_G),
        viol_test_A=viol_A_at(test, calA[0.10]["gamma"], DELTA_G),
        viol_val_B=viol_B_at(val, calA[0.10]["gamma"], DELTA_G),
        viol_test_B=viol_B_at(test, calA[0.10]["gamma"], DELTA_G),
        test_calA={str(e): (None if calA[e] is None else
                            dict(gamma=calA[e]["gamma"],
                                 cert=cert_stats(test, calA[e]["gamma"], DELTA_G),
                                 viol=viol_A_at(test, calA[e]["gamma"], DELTA_G)))
                   for e in eps_list},
        c5_val_1=c5_stats(val, 1.0, DELTA_G),
        c5_val_A=c5_stats(val, calA[0.10]["gamma"], DELTA_G),
        c5_test_1=c5_stats(test, 1.0, DELTA_G),
        c5_test_A=c5_stats(test, calA[0.10]["gamma"], DELTA_G),
        sens_kappa={},
    )
    p75rec = gamma_star_A(curve_A(val, DELTA_G, "k_hat"), 0.10)
    p90rec = gamma_star_A(curve_A(val, DELTA_G, "k_hat_p90"), 0.10)
    out["sens_kappa"]["p75"] = (None if p75rec is None else
                                dict(gamma=p75rec["gamma"],
                                     cert=cert_stats(val, p75rec["gamma"], DELTA_G)))
    out["sens_kappa"]["p90"] = (None if p90rec is None else
                                dict(gamma=p90rec["gamma"],
                                     cert=cert_stats(val, p90rec["gamma"], DELTA_G,
                                                     kappa_key="k_hat_p90")))
    # delta_g sensitivity: re-evaluate gamma*_A at 0.05 / 0.02
    sens = []
    for tag, rows in (("0.13", val), ("0.05", val_dg05), ("0.02", val_dg02)):
        if rows is None:
            continue
        d = float(tag)
        rec = gamma_star_A(curve_A(rows, d), 0.10)
        if rec is None:
            sens.append(dict(delta_g=d, gamma=None))
            continue
        sens.append(dict(delta_g=d, gamma=rec["gamma"], k=rec["k"], n=rec["n"],
                         rate=rec["rate"], cp=rec["cp"],
                         cert1=cert_stats(rows, 1.0, d),
                         cert_star=cert_stats(rows, rec["gamma"], d)))
    out["sens_delta_g"] = sens
    return out


def print_report(rep):
    print("=" * 78)
    print("W4 baseline + calibration report")
    print(f"phi_tol={PHI_TOL:.6f}  delta_g={DELTA_G}  N_denom={N_DENOM}  "
          f"n_min_cert={N_MIN_CERT}")
    print(f"counts: {rep['counts']}")
    print_baseline(rep["baseline"])
    print("\n" + "=" * 78)
    print_calA(rep["calA"])
    print_calB(rep["calB"])
    print("\n  [certification rate: both gamma=1 and gamma*_A(eps=0.10)]")
    print_cert("val  gamma=1", rep["cert_val"]["gamma1"])
    print_cert("val  gamma*_A(0.10)", rep["cert_val"]["gammaA10"])
    print_cert("test gamma=1", rep["cert_test"]["gamma1"])
    print_cert("test gamma*_A(0.10)", rep["cert_test"]["gammaA10"])
    print("\n  [violation A: max err > bound; denominator = all vinf samples]")
    print(f"  val  gamma*=%.2f: k=%d/%d rate=%s CP95=%s" %
          (rep["viol_val_A"]["gamma"], rep["viol_val_A"]["k"], rep["viol_val_A"]["n"],
           pct(rep["viol_val_A"]["rate"]), pct(rep["viol_val_A"]["cp"])))
    print(f"  test gamma*=%.2f: k=%d/%d rate=%s CP95=%s" %
          (rep["viol_test_A"]["gamma"], rep["viol_test_A"]["k"], rep["viol_test_A"]["n"],
           pct(rep["viol_test_A"]["rate"]), pct(rep["viol_test_A"]["cp"])))
    print("\n  [violation B: certified set, max err > phi_tol]")
    for nm, rec in (("val", rep["viol_val_B"]), ("test", rep["viol_test_B"])):
        print(f"  {nm} gamma*_A: n_cert={rec['n_cert']} k={rec['k']} "
              f"rate={pct(rec['rate'])} CP95={pct(rec['cp'])}")
    print("\n  [C_5pt_max on certified set]")
    print_c5("val  gamma=1", rep["c5_val_1"])
    print_c5("val  gamma*_A(0.10)", rep["c5_val_A"])
    print_c5("test gamma=1", rep["c5_test_1"])
    print_c5("test gamma*_A(0.10)", rep["c5_test_A"])
    print("\n  [sensitivity: delta_g]")
    for rec in rep["sens_delta_g"]:
        print(f"  delta_g={rec['delta_g']:.2f}  gamma*_A(0.10)={fmt(rec.get('gamma'),2)}  "
              f"cert@1={pct(rec['cert1']['rate']) if rec.get('cert1') else '—'}  "
              f"cert@*={pct(rec['cert_star']['rate']) if rec.get('cert_star') else '—'}")
    print("\n  [sensitivity: kappa_hat P75 vs P90, gamma*_A(0.10)]")
    for kk in ("p75", "p90"):
        rec = rep["sens_kappa"][kk]
        if rec is None:
            print(f"  {kk}: —")
            continue
        print(f"  {kk}: gamma*={rec['gamma']:.2f}  cert@*="
              f"{pct(rec['cert']['rate'])} ({rec['cert']['n_cert']}/{rec['cert']['n']})")


# ----------------------------------------------------------------------------- regression
def regression():
    path = shard_files("val")[0]
    d = np.load(path)
    rows = []
    for i in range(64):
        m = json.loads(d["meta"][i])
        dead = np.asarray(d["dead"][i], bool)
        if dead.sum() < 30 or m.get("vinf") is None:
            continue
        phi = np.asarray(d["phi"][i], np.float64)
        orr = np.asarray(d["orr"][i], np.float64)
        roi = np.asarray(d["roi"][i], bool)
        rows.append(eval_one(phi, orr, dead, roi, m, DELTA_G, "all", seed=i))
    print(f"regression samples: {len(rows)} (expected 64)")
    ok = True
    print(f"{'method':10s}{'P95(med)':>11s}{'ref':>10s}{'diff':>10s}"
          f"{'Rel(med)':>10s}{'ref':>9s}{'diff':>9s}")
    for meth in METHODS:
        p95 = np.median([r["err"][meth]["p95"] for r in rows])
        rel = np.median([r["rel"][meth] for r in rows])
        rp, rr = REF_SMOKE[meth]
        dp, dr = abs(p95 - rp), abs(rel - rr)
        good = dp <= 5e-4 and dr <= 5e-3
        ok &= good
        print(f"{meth:10s}{p95:11.4f}{rp:10.4f}{dp:10.1e}"
              f"{rel:10.3f}{rr:9.3f}{dr:9.1e}  {'OK' if good else 'FAIL'}")
    order_ok = (np.median([r["err"]["ORR"]["p95"] for r in rows]) <
                np.median([r["err"]["Harmonic"]["p95"] for r in rows]) <
                np.median([r["err"]["Biharm"]["p95"] for r in rows]))
    rel_ok = np.median([r["rel"]["Harmonic"] for r in rows]) < 1.2
    print(f"\nORR < Harmonic < Biharm (P95 median): {order_ok}   "
          f"Harmonic Rel-ORR < 1.2: {rel_ok}")
    print("REGRESSION", "PASS" if (ok and order_ok and rel_ok) else "FAIL")
    return ok and order_ok and rel_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "report", "regression"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--delta-g", type=float, default=DELTA_G)
    ap.add_argument("--methods", choices=["all", "harmonic"], default="all")
    ap.add_argument("--vinf-only", action="store_true")
    ap.add_argument("--n-max", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cls", default=None)
    a = ap.parse_args()
    if a.mode == "eval":
        eval_split(a.split, a.delta_g, a.methods, a.vinf_only, out=a.out, n_max=a.n_max,
                   cls_filter=a.cls)
        return
    if a.mode == "regression":
        sys.exit(0 if regression() else 1)
    # report
    paths = {("val", 0.13): "w4_rows_val_dg0.13.json",
             ("test", 0.13): "w4_rows_test_dg0.13.json",
             ("val", 0.05): "w4_rows_val_dg0.05.json",
             ("val", 0.02): "w4_rows_val_dg0.02.json"}
    def load(key):
        p = os.path.join(RES, paths[key])
        if not os.path.exists(p):
            print("missing", p); return None
        return json.load(open(p))
    val = load(("val", 0.13)); test = load(("test", 0.13))
    if val is None or test is None:
        raise SystemExit("need val/test dg=0.13 rows")
    rep = build_report(val, test, load(("val", 0.05)), load(("val", 0.02)))
    print_report(rep)
    out = os.path.join(RES, "w4_report.json")
    json.dump(rep, open(out, "w"), indent=1, default=float)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
