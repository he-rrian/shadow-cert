"""w8_class_gated.py -- certificate-first acceptance (P only) from compact cache.

U/R scenes are rejected before RCPS; RCPS is calibrated/evaluated only on P scenes.
Reports:
  - P-only acceptance fraction (among P dead pixels)
  - overall accepted fraction (P accepted / all dead pixels, since U/R rejected)
  - test failure rate among accepted P pixels
"""
import os, json, math
import numpy as np

ROOT = r"F:\paper\2\P2"
PHI_TOL = math.pi / 10
CONF_DELTA = 0.05   # confidence parameter delta for the risk bound (NOT delta_g=0.13 in the certificate)


def load(seed, split):
    z = np.load(os.path.join(ROOT, "data", "preds_compact", f"seed{seed}_{split}.npz"))
    e = z["e"].astype(np.float64); s = np.maximum(z["sigma"].astype(np.float64), 1e-6)
    n = z["n_dead"].astype(int); off = np.r_[0, np.cumsum(n)]
    y = z["y75"].astype(int)
    scenes = []
    for i in range(len(n)):
        scenes.append(dict(y=int(y[i]), e=e[off[i]:off[i+1]], s=s[off[i]:off[i+1]]))
    return scenes


def curves(cal, alpha, n_grid=200):
    sig = np.concatenate([q["s"] for q in cal])
    lam = np.unique(np.quantile(sig, np.linspace(0, 1, n_grid)))
    n = len(cal); rows = []
    for L in lam:
        a = np.array([np.mean(q["s"] <= L) for q in cal])
        e = np.array([np.mean((q["s"] <= L) & (q["e"] > PHI_TOL)) for q in cal])
        W = e - alpha * a + alpha
        var = float(W.var(ddof=1)) if len(W) > 1 else 0.0
        L2 = math.log(2 / CONF_DELTA)
        eps = math.sqrt(2 * var * L2 / n) + 7 * (1 + alpha) * L2 / (3 * max(n - 1, 1))
        u = float(W.mean() + eps - alpha)
        rows.append(dict(lam=float(L), acc=float(a.mean()), risk=float(e.sum() / max(a.sum(), 1e-9)), u=u))
    U = np.array([r["u"] for r in rows]); Ur = np.minimum.accumulate(U[::-1])[::-1]
    for r, u in zip(rows, Ur): r["u_run"] = float(u)
    valid = [i for i, r in enumerate(rows) if r["u_run"] <= 0]
    return rows[max(valid)] if valid else dict(lam=0.0, acc=0.0, risk=0.0, u_run=0.0)


def eval_test(test, lam):
    e = np.concatenate([q["e"] for q in test]); s = np.concatenate([q["s"] for q in test])
    acc = s <= lam
    return dict(acc_frac=float(acc.mean()), risk=float(np.mean(e[acc] > PHI_TOL)) if acc.any() else 0.0,
                n_accepted=int(acc.sum()), n_dead=int(len(e)))


def main():
    out = []
    for seed in range(5):
        val = load(seed, "val"); test = load(seed, "test")
        Pval = [q for q in val if q["y"] == 0]; Ptest = [q for q in test if q["y"] == 0]
        r = curves(Pval, 0.05)
        t = eval_test(Ptest, r["lam"])
        # overall acceptance relative to all dead pixels (U/R rejected)
        all_dead = sum(len(q["e"]) for q in test)
        overall = t["n_accepted"] / max(all_dead, 1)
        rec = dict(seed=seed, nP_val=len(Pval), nP_test=len(Ptest),
                   lam=r["lam"], P_acc=t["acc_frac"], P_risk=t["risk"],
                   overall_acc=overall, nP_accepted=t["n_accepted"])
        out.append(rec)
        print(rec)
    json.dump(out, open(os.path.join(ROOT, "probes", "results", "w8_class_gated.json"), "w"), indent=1)
    print("mean P_acc", np.mean([r["P_acc"] for r in out]), "P_risk", np.mean([r["P_risk"] for r in out]),
          "overall_acc", np.mean([r["overall_acc"] for r in out]))


if __name__ == "__main__":
    main()
