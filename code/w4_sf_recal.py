"""w4_sf_recal.py -- zero-retrain, target-domain gamma recalibration on SynthFringe.

Question: the MATLAB-calibrated certificate does not transfer to SynthFringe
(gamma=1: 0/240 certifiable).  If we keep the SAME model-free certificate and only
re-calibrate gamma on SF-val (no network, no retraining), does a usable SF
certificate exist?  Report SF-val -> SF-test (test is one-shot, not used to tune).

Protocol is the same as W4:
  cert      : gamma*kappa_hat*vinf + delta_g <= phi_tol
  A-cert    : certified and per-pixel MAX error > gamma*kappa_hat*vinf + delta_g
  B         : certified and per-pixel MAX error > phi_tol
  rate      : one-sided 95% Clopper-Pearson upper bound
  gamma*    : smallest gamma with n_cert >= N_MIN and CP95(A-cert) <= eps

Run: python w4_sf_recal.py
Writes probes/results/w4_sf_recal.json / .md
"""
import os, json, time, importlib.util
import numpy as np, cv2
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "probes", "results")
PHI_TOL = float(np.pi / 10)
DELTA_G = 0.13
N_MIN = 50
EPS = 0.05


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


W4 = load("w4_ref", os.path.join(HERE, "w4_baseline.py"))


def cp_upper(k, n, alpha=0.05):
    if n <= 0:
        return float("nan")
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, k + 1.0, n - k))


def eval_split(split):
    d = os.path.join(DATA, f"sf_{split}")
    man = json.load(open(os.path.join(d, "manifest.json")))
    rows = []
    t0 = time.time()
    for s in man["shards"]:
        z = np.load(os.path.join(d, s["path"]))
        for i in range(len(z["meta"])):
            m = json.loads(z["meta"][i])
            vinf = m.get("vinf")
            dead = np.asarray(z["dead"][i], bool)
            if vinf is None or dead.sum() < 30:
                continue
            phi = np.asarray(z["phi"][i], np.float64)
            orr = np.asarray(z["orr"][i], np.float64)
            rng = np.random.default_rng(i)
            src = phi + rng.normal(0.0, DELTA_G / 2.0, phi.shape) * (~dead)
            har = W4.harmonic_fill(dead, src)
            e = np.abs(har[dead] - phi[dead])
            lap = np.abs(cv2.Laplacian(phi, cv2.CV_64F)[dead]).max()
            eo = np.abs(orr[dead] - phi[dead]).max()
            rows.append(dict(split=split, idx=i, id=m["id"], branch=m["branch"],
                             npx=int(dead.sum()), vinf=float(vinf),
                             k_hat=float(m["kappa_hat"]),
                             err_max=float(e.max()), err_p95=float(np.percentile(e, 95)),
                             err_mean=float(e.mean()),
                             orr_max=float(eo), k_max=float(lap),
                             c5=float(eo / (lap * vinf)) if lap > 0 else None))
        print(f"  {split}: {len(rows)} rows  {time.time()-t0:.0f}s", flush=True)
    return rows


def curve(rows):
    out = []
    for g in np.round(np.arange(0.0, 1.6001, 0.05), 4):
        bounds = [g * r["k_hat"] * r["vinf"] + DELTA_G for r in rows]
        cert = [b <= PHI_TOL for b in bounds]
        n_cert = int(sum(cert))
        kA = sum(1 for r, c, b in zip(rows, cert, bounds)
                 if c and r["err_max"] > b)
        kB = sum(1 for r, c in zip(rows, cert) if c and r["err_max"] > PHI_TOL)
        out.append(dict(gamma=float(g), n=len(rows), n_cert=n_cert,
                        cert_rate=n_cert / len(rows),
                        A=kA, A_rate=kA / max(n_cert, 1), A_cp=cp_upper(kA, n_cert),
                        B=kB, B_rate=kB / max(n_cert, 1), B_cp=cp_upper(kB, n_cert)))
    return out


def pick(curve_, eps=EPS, n_min=N_MIN):
    for rec in curve_:
        if rec["n_cert"] >= n_min and rec["A_cp"] <= eps:
            return rec
    return None


def main():
    val = eval_split("val")
    test = eval_split("test")
    cv = curve(val); ct = curve(test)
    g = pick(cv)
    # also report the best gamma at gamma=1 and a few fixed points
    fixed = {str(x): next(r for r in cv if abs(r["gamma"] - x) < 1e-9)
             for x in (0.5, 0.75, 1.0, 1.3)}
    out = dict(protocol=dict(phi_tol=PHI_TOL, delta_g=DELTA_G, eps=EPS, n_min=N_MIN),
               n_val=len(val), n_test=len(test),
               gamma_star=g, val_curve=cv, test_at_gamma_star=None,
               fixed=fixed, by_eps={})
    if g is not None:
        gt = next(r for r in ct if abs(r["gamma"] - g["gamma"]) < 1e-9)
        out["test_at_gamma_star"] = gt
    for eps in (0.05, 0.10, 0.20):
        rec = pick(cv, eps, N_MIN)
        ent = {"val": rec, "test": None}
        if rec is not None:
            ent["test"] = next(r for r in ct if abs(r["gamma"] - rec["gamma"]) < 1e-9)
        out["by_eps"][str(eps)] = ent
    json.dump(out, open(os.path.join(RES, "w4_sf_recal.json"), "w"), indent=1, default=float)

    L = ["# SynthFringe 目标域重标定（零重训）", "",
         f"phi_tol={PHI_TOL:.6f}  delta_g={DELTA_G}  eps={EPS}  n_min={N_MIN}",
         f"SF-val {len(val)} / SF-test {len(test)} 个含 vinf 样本", ""]
    L.append("| gamma | val n_cert | val cert% | val A-cert (CP95) | val B (CP95) |")
    L.append("|---|---|---|---|---|")
    for r in cv:
        if round(r["gamma"], 2) in (0.5, 0.75, 1.0, 1.3) or (g and abs(r["gamma"] - g["gamma"]) < 1e-9):
            L.append(f"| {r['gamma']:.2f} | {r['n_cert']} | {100*r['cert_rate']:.1f}% | "
                     f"{r['A']}/{r['n_cert']} = {100*r['A_rate']:.1f}% (CP {100*r['A_cp']:.1f}%) | "
                     f"{r['B']}/{r['n_cert']} = {100*r['B_rate']:.1f}% (CP {100*r['B_cp']:.1f}%) |")
    if g:
        gt = out["test_at_gamma_star"]
        L += ["", f"**gamma*_SF = {g['gamma']:.2f}**（val：认证 {g['n_cert']}/{g['n']} = "
              f"{100*g['cert_rate']:.1f}%，A-cert {100*g['A_rate']:.1f}%（CP {100*g['A_cp']:.1f}%））",
              "", "| SF-test（一次性） | n_cert | cert% | A-cert (CP95) | B (CP95) |",
              "|---|---|---|---|---|",
              f"| gamma*_SF={gt['gamma']:.2f} | {gt['n_cert']}/{gt['n']} | {100*gt['cert_rate']:.1f}% | "
              f"{gt['A']} ({100*gt['A_rate']:.1f}%, CP {100*gt['A_cp']:.1f}%) | "
              f"{gt['B']} ({100*gt['B_rate']:.1f}%, CP {100*gt['B_cp']:.1f}%) |"]
    else:
        L += ["", f"**没有 gamma 满足 n_cert>={N_MIN} 且 CP95(A-cert)<={EPS}**；"
              "SF 上连重标定后的 5% 证书也不存在，只能报误差或更宽的风险档。"]
    L += ["", "## 不同风险档的 SF 可认证性（SF-val 校准，SF-test 一次性）", "",
          "| eps | gamma*_SF | val n_cert | val cert% | val A-cert (CP95) | test n_cert | test A-cert (CP95) |",
          "|---|---|---|---|---|---|---|"]
    for eps, ent in out["by_eps"].items():
        v, t = ent["val"], ent["test"]
        if v is None:
            L.append(f"| {eps} | — | — | — | 无解（n_cert>={N_MIN} 且 CP95<={eps}） | — | — |")
        else:
            tcell = (f"{t['n_cert']}/{t['n']} ({100*t['A_rate']:.1f}%, CP {100*t['A_cp']:.1f}%)"
                     if t else "—")
            L.append(f"| {eps} | {v['gamma']:.2f} | {v['n_cert']}/{v['n']} | {100*v['cert_rate']:.1f}% | "
                     f"{v['A']}/{v['n_cert']} ({100*v['A_rate']:.1f}%, CP {100*v['A_cp']:.1f}%) | {tcell} |")
    text = "\n".join(L)
    open(os.path.join(RES, "w4_sf_recal.md"), "w", encoding="utf-8").write(text)
    print(text)
    print("\nsaved", os.path.join(RES, "w4_sf_recal.json"))


if __name__ == "__main__":
    main()
