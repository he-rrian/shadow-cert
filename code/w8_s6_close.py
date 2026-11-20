"""w8_s6_close.py -- S6 carry-over: kappa P75/P90, block coverage, P-only C4.

Local, no torch. Uses compact dead-only scenes + source meta for kappa.
Writes probes/results/w8_s6_close.json and w8_s6_main_table.md.
"""
import os, json, math
import numpy as np
from scipy.stats import beta, binom
from scipy.optimize import brentq

ROOT = r"F:\paper\2\P2"
PHI = math.pi / 10
DG = 0.13
GAMMA = 0.75
BLOCK = 200


def load_split(split, seed=4):
    # compact arrays in source order (idx = dataset index)
    z = np.load(os.path.join(ROOT, "data", "preds_compact", f"seed{seed}_{split}.npz"))
    n = z["n_dead"].astype(int); off = np.r_[0, np.cumsum(n)]
    e = z["e"].astype(np.float64); s = np.maximum(z["sigma"].astype(np.float64), 1e-6)
    sid = z["sid"].astype(int); idx = z["idx"].astype(int)
    y75 = z["y75"].astype(int); br = z["branch_id"].astype(int)
    # source meta
    man = json.load(open(os.path.join(ROOT, "data", split, "manifest_000.json")))
    metas = []
    for sh in man["shards"]:
        q = np.load(os.path.join(ROOT, "data", split, sh["path"]))
        metas.extend([json.loads(m) for m in q["meta"]])
    scenes = []
    for k in range(len(n)):
        i = int(idx[k]); m = metas[i]
        v = m.get("vinf")
        kh = m.get("kappa_hat_ps")
        scenes.append(dict(sid=int(sid[k]), idx=i, branch=int(br[k]), y75=int(y75[k]),
                           vinf=None if v is None else float(v),
                           kh=None if not kh else [float(x) for x in kh],
                           e=e[off[k]:off[k+1]], s=s[off[k]:off[k+1]],
                           score=e[off[k]:off[k+1]] / s[off[k]:off[k+1]]))
    return scenes


def label(m, kh_idx):
    if m.get("branch") == 6:  # not used; branch id handled outside
        pass
    if m["vinf"] is None or m["kh"] is None:
        return 1
    return 0 if (GAMMA * m["kh"][kh_idx] * m["vinf"] + DG) <= PHI else 1


def tau_for(vals, alpha):
    n = len(vals)
    if n == 0:
        return float("nan")
    k = int(math.ceil((n + 1) * (1 - alpha))) - 1
    k = min(max(k, 0), n - 1)
    return float(np.sort(vals)[k])


def q_high(a, q):
    return float(np.quantile(np.asarray(a), q, method="higher"))


def c3_for(cal, test, kh_idx, alpha):
    calP = [m for m in cal if label(m, kh_idx) == 0]
    testP = [m for m in test if label(m, kh_idx) == 0]
    scores = [q_high(m["score"], 0.95) for m in calP]
    t = tau_for(scores, alpha)
    cov = [float(np.mean(m["e"] <= t * m["s"])) for m in testP]
    return dict(n_cal=len(calP), n_test=len(testP), tau=t,
                scene_success=float(np.mean([c >= 0.95 for c in cov])) if cov else None,
                mean_coverage=float(np.mean(cov)) if cov else None)


def flip_stats(cal, test, k1=1, k2=2):
    out = {}
    for name, ss in (("val", cal), ("test", test)):
        a = [label(m, k1) for m in ss if m["vinf"] is not None]
        b = [label(m, k2) for m in ss if m["vinf"] is not None]
        ch = sum(1 for x, y in zip(a, b) if x != y)
        out[name] = dict(n=len(a), changed=ch, frac=ch / max(len(a), 1),
                        nP_p75=sum(1 for x in a if x == 0), nP_p90=sum(1 for x in b if x == 0))
    return out


def block_cov(cal, test, alpha):
    calP = [m for m in cal if label(m, 1) == 0]
    t = tau_for([q_high(m["score"], 0.95) for m in calP], alpha)
    rows = {}
    for m in test:
        if label(m, 1) != 0:
            continue
        b = m["sid"] // BLOCK
        cov = float(np.mean(m["e"] <= t * m["s"]))
        rows.setdefault(b, []).append(cov)
    return {int(b): dict(n=len(v), mean_coverage=float(np.mean(v)),
                         success=float(np.mean([x >= 0.95 for x in v])))
            for b, v in sorted(rows.items())}


def cp_upper(k, n, delta=0.05):
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, n - k))


def bentkus_ucb(x, delta=0.05):
    """One-sided Bentkus UCB for mean of iid [0,1] variables.

    P(mean <= t) <= e * P(Bin(n, mu) <= ceil(n t)).
    Invert for mu at equality.
    """
    x = np.asarray(x, dtype=float); n = len(x)
    if n == 0:
        return 1.0
    t = float(x.mean())
    k = int(math.ceil(n * t))
    if k >= n:
        return 1.0
    def g(mu):
        return math.e * binom.cdf(k, n, mu) - delta
    # g(0)=e-delta>0, g(1)=-delta<0
    try:
        return float(brentq(g, 0.0, 1.0))
    except Exception:
        return 1.0


def c4_alpha(scenes, alpha, reps=20, seed=20261120, n_sel=120):
    """P-only split: select lambda on n_sel, certify on rest via Bentkus + CP."""
    P = [m for m in scenes if m["y75"] == 0]
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(reps):
        idx = rng.permutation(len(P)); sel = [P[i] for i in idx[:n_sel]]; hol = [P[i] for i in idx[n_sel:]]
        grid = np.unique(np.quantile(np.concatenate([m["s"] for m in sel]), np.linspace(0.1, 1.0, 30)))
        best = None
        for L in grid:
            a = np.array([np.mean(m["s"] <= L) for m in sel])
            f = np.array([np.mean((m["s"] <= L) & (m["e"] > PHI)) for m in sel])
            if f.sum() / max(a.sum(), 1e-9) <= alpha:
                best = L
        if best is None:
            rows.append(dict(lam=0.0, acc=0.0, risk=0.0, bentkus=1.0, cp=1.0, n_acc=0))
            continue
        a = np.array([np.mean(m["s"] <= best) for m in hol])
        f = np.array([np.mean((m["s"] <= best) & (m["e"] > PHI)) for m in hol])
        W = f - alpha * a + alpha
        Z = W / (1 + alpha)
        anyfail = (f > 0).astype(float)
        rows.append(dict(lam=float(best), acc=float(a.mean()),
                         risk=float(f.sum() / max(a.sum(), 1e-9)),
                         bentkus=bentkus_ucb(Z), cp=cp_upper(int(anyfail.sum()), len(anyfail)),
                         n_acc=int(a.sum() > 0)))
    return rows


def main():
    val = load_split("val"); test = load_split("test")
    out = {}
    # C3 kappa sensitivity
    out["c3"] = {}
    for name, ki in (("P50", 0), ("P75", 1), ("P90", 2)):
        out["c3"][name] = {a: c3_for(val, test, ki, a) for a in (0.05, 0.10, 0.20)}
    out["flip"] = flip_stats(val, test)
    out["flip_p50"] = flip_stats(val, test, 1, 0)
    out["blocks"] = block_cov(val, test, 0.05)
    # C4 P-only
    out["c4"] = {}
    for sd in range(5):
        _, test_sd = load_split("val", sd), load_split("test", sd)
        out["c4"][str(sd)] = {a: c4_alpha(test_sd, a, seed=20261120 + sd) for a in (0.05, 0.10)}
    json.dump(out, open(os.path.join(ROOT, "probes", "results", "w8_s6_close.json"), "w"), indent=1)

    # markdown table
    lines = []
    lines.append("# S6 carry-over: kappa sensitivity, block coverage, P-only C4\n")
    lines.append("Local compact cache + source meta; no GPU. C3 P75 official float32 numbers are in `w8_p_only/`.\n")
    lines.append("## 1. C3 P-only under kappa P75 vs P90\n")
    lines.append("| kappa | alpha | n_cal P | n_test P | scene success | mean coverage |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in ("P50", "P75", "P90"):
        for a in (0.05, 0.10, 0.20):
            r = out["c3"][name][a]
            lines.append(f"| {name} | {a:.2f} | {r['n_cal']} | {r['n_test']} | {r['scene_success']:.4f} | {r['mean_coverage']:.4f} |")
    lines.append("\n## 2. P/U flip P75 -> P90\n")
    lines.append("| split | n with vinf | changed | frac | nP P75 | nP P90 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for k, r in out["flip"].items():
        lines.append(f"| {k} | {r['n']} | {r['changed']} | {r['frac']:.3f} | {r['nP_p75']} | {r['nP_p90']} |")
    lines.append("\n## 3. Block coverage (P-only, P75, alpha=.05)\n")
    lines.append("| test block | n P | mean coverage | scene success |")
    lines.append("|---:|---:|---:|---:|")
    for b, r in out["blocks"].items():
        lines.append(f"| {b} | {r['n']} | {r['mean_coverage']:.4f} | {r['success']:.3f} |")
    lines.append("\n## 4. C4 P-only (120 select / 73 certify, 20 reps)\n")
    lines.append("| alpha | accept | pooled risk | Bentkus UCB mean | Bentkus pass rate | CP scene-any UCB mean | CP pass rate |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for a in (0.05, 0.10):
        rr = [x for sd in range(5) for x in out["c4"][str(sd)][a]]
        acc = np.mean([x["acc"] for x in rr]); risk = np.mean([x["risk"] for x in rr])
        bu = np.mean([x["bentkus"] for x in rr]); bp = np.mean([x["bentkus"] <= a for x in rr])
        cu = np.mean([x["cp"] for x in rr]); cp = np.mean([x["cp"] <= a for x in rr])
        lines.append(f"| {a:.2f} | {acc:.3f} | {risk:.4f} | {bu:.4f} | {bp:.3f} | {cu:.4f} | {cp:.3f} |")
    open(os.path.join(ROOT, "probes", "results", "w8_s6_main_table.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()




