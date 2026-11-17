"""w4_decide.py -- decision tables on top of the W4 per-sample rows.

Reads probes/results/w4_rows_{split}_dg*.json (produced by w4_calibrate.py eval) and
answers the three questions the W4 data raised:

  1. which gamma to use for the Tier-1 certificate (global vs certified-set denominator);
  2. how the certification rate depends on the tolerance phi_tol (pi/20, pi/10, pi/5);
  3. whether Biharm (not ORR) is the strongest classical baseline, per branch.

All rates use per-pixel MAX error (never the mean) and report a one-sided 95%
Clopper-Pearson upper bound.

Run: python w4_decide.py
Writes probes/results/w4_decision.json and prints markdown tables.
"""
import json, os, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "probes", "results")

spec = importlib.util.spec_from_file_location("w4c", os.path.join(HERE, "w4_calibrate.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

TOLS = {"pi/20": np.pi / 20, "pi/10": np.pi / 10, "pi/5": np.pi / 5}
GAMMAS = [0.40, 0.45, 0.55, 0.60, 0.65, 0.75, 0.85, 0.90, 1.00, 1.20]
METHODS = C.METHODS


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def vinf(rows):
    return [r for r in rows if r["has_vinf"]]


def cert_mask(rows, gamma, delta, tol):
    return [b <= tol for b in (gamma * r["k_hat"] * r["vinf"] + delta for r in rows)]


def metrics(rows, gamma, delta, tol):
    """cert rate, A on certified, B on certified, A over all vinf samples."""
    rows = vinf(rows)
    bounds = [gamma * r["k_hat"] * r["vinf"] + delta for r in rows]
    cert = [b <= tol for b in bounds]
    n = len(rows); n_cert = int(sum(cert))
    kA = sum(1 for r, c, b in zip(rows, cert, bounds)
             if c and r["err"]["Harmonic"]["max"] > b)
    kB = sum(1 for r, c in zip(rows, cert)
             if c and r["err"]["Harmonic"]["max"] > tol)
    kAll = sum(1 for r, b in zip(rows, bounds)
               if r["err"]["Harmonic"]["max"] > b)
    return dict(gamma=gamma, phi_tol=tol, n=n, n_cert=n_cert,
                cert_rate=n_cert / n,
                A_cert=kA, A_cert_rate=kA / max(n_cert, 1), A_cert_cp=C.cp_upper(kA, n_cert),
                B=kB, B_rate=kB / max(n_cert, 1), B_cp=C.cp_upper(kB, n_cert),
                A_all=kAll, A_all_rate=kAll / n, A_all_cp=C.cp_upper(kAll, n))


def cp_safe(rows, delta, eps, tol, kind, n_min=100):
    """smallest gamma whose 95% upper bound <= eps; kind in {'A_all','A_cert','B'}"""
    for g in np.round(np.arange(0.0, 6.0001, 0.05), 4):
        m = metrics(rows, g, delta, tol)
        if kind == "A_all":
            # denominator is the full vinf split, so n_cert is irrelevant here
            if m["A_all_cp"] <= eps:
                return m
            continue
        if m["n_cert"] < n_min:
            continue
        if kind == "A_cert" and m["A_cert_cp"] <= eps:
            return m
        if kind == "B" and m["B_cp"] <= eps:
            return m
    return None


def branch_table(rows):
    rows = vinf(rows)
    out = {}
    for br in ("sliver", "arc", "blob"):
        rr = [r for r in rows if r["branch"] == br]
        rec = {"n": len(rr)}
        for m in METHODS:
            rec[m] = dict(p95=float(np.median([r["err"][m]["p95"] for r in rr])),
                          rel=float(np.median([r["rel"][m] for r in rr])))
        out[br] = rec
    return out


def main():
    val = load("w4_rows_val_dg0.13.json")
    test = load("w4_rows_test_dg0.13.json")
    train = load("w4_rows_train_dg0.13.json")
    out = {"branch": branch_table(val), "matrix": [], "cp_safe": {}, "train": None,
           "test_at_val_gamma": {}, "counts": {}}
    out["counts"] = dict(val=len(val), val_vinf=len(vinf(val)),
                         test=len(test), test_vinf=len(vinf(test)),
                         train=len(train) if train else 0)

    for tname, tol in TOLS.items():
        for g in GAMMAS:
            m = metrics(val, g, C.DELTA_G, tol)
            m["tol_name"] = tname
            out["matrix"].append(m)

    for eps in (0.05, 0.10, 0.20):
        out["cp_safe"][str(eps)] = {
            "A_all_pi10": cp_safe(val, C.DELTA_G, eps, np.pi / 10, "A_all"),
            "A_cert_pi10": cp_safe(val, C.DELTA_G, eps, np.pi / 10, "A_cert"),
            "B_pi10": cp_safe(val, C.DELTA_G, eps, np.pi / 10, "B"),
            "A_cert_pi5": cp_safe(val, C.DELTA_G, eps, np.pi / 5, "A_cert"),
            "B_pi5": cp_safe(val, C.DELTA_G, eps, np.pi / 5, "B"),
        }

    # test at the val-selected gammas (no re-tuning)
    for name, g in (("gamma1", 1.0), ("gamma0.65", 0.65), ("gamma0.55", 0.55)):
        out["test_at_val_gamma"][name] = metrics(test, g, C.DELTA_G, np.pi / 10)

    if train:
        out["train"] = dict(
            gamma1=metrics(train, 1.0, C.DELTA_G, np.pi / 10),
            gamma075=metrics(train, 0.75, C.DELTA_G, np.pi / 10),
            gamma060=metrics(train, 0.60, C.DELTA_G, np.pi / 10),
        )

    # ---- markdown output
    L = []
    L.append("# W4 decision tables (auto-generated by code/w4_decide.py)\n")
    L.append(f"counts: {out['counts']}\n")

    L.append("\n## 1. Per-branch baselines (val, delta_g=0.13, median over samples)\n")
    L.append("| branch | n | metric | Identity | Nearest | Harmonic | Biharm | Poly2 | ORR |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for br, rec in out["branch"].items():
        for stat, key in (("P95 err", "p95"), ("Rel-ORR", "rel")):
            vals = " | ".join(f"{rec[m][key]:.4f}" if key == "p95" else f"{rec[m][key]:.3f}"
                              for m in METHODS)
            L.append(f"| {br} | {rec['n']} | {stat} | {vals} |")

    L.append("\n## 2. gamma x tolerance (val)\n")
    L.append("| tol | gamma | n_cert | cert% | A_cert k/% (CP95) | B k/% (CP95) | A_all k/% (CP95) |")
    L.append("|---|---|---|---|---|---|---|")
    for m in out["matrix"]:
        L.append(f"| {m['tol_name']} | {m['gamma']:.2f} | {m['n_cert']} | "
                 f"{100*m['cert_rate']:.1f}% | {m['A_cert']}/{m['A_cert_rate']*100:.1f}% "
                 f"({m['A_cert_cp']*100:.1f}%) | {m['B']}/{m['B_rate']*100:.1f}% "
                 f"({m['B_cp']*100:.1f}%) | {m['A_all']}/{m['A_all_rate']*100:.1f}% "
                 f"({m['A_all_cp']*100:.1f}%) |")

    L.append("\n## 3. CP-safe gamma (95% upper bound <= eps)\n")
    L.append("| eps | A_all pi/10 | A_cert pi/10 | B pi/10 | A_cert pi/5 | B pi/5 |")
    L.append("|---|---|---|---|---|---|")
    for eps, rec in out["cp_safe"].items():
        cells = []
        for k in ("A_all_pi10", "A_cert_pi10", "B_pi10", "A_cert_pi5", "B_pi5"):
            r = rec[k]
            if r is None:
                cells.append("—")
            else:
                rate = r["A_all_rate"] if k == "A_all_pi10" else (
                    r["A_cert_rate"] if "A_cert" in k else r["B_rate"])
                cells.append(f"g={r['gamma']:.2f}, {r['n_cert']}/{r['n']} "
                             f"({100*rate:.1f}%, CP {100*(r['A_all_cp'] if k=='A_all_pi10' else (r['A_cert_cp'] if 'A_cert' in k else r['B_cp'])):.1f}%)")
        L.append(f"| {eps} | " + " | ".join(cells) + " |")

    L.append("\n## 4. Test at val-selected gamma (pi/10)\n")
    L.append("| gamma | n_cert | cert% | A_cert (CP95) | B (CP95) | A_all (CP95) |")
    L.append("|---|---|---|---|---|---|")
    for name, m in out["test_at_val_gamma"].items():
        L.append(f"| {m['gamma']:.2f} | {m['n_cert']} | {100*m['cert_rate']:.1f}% | "
                 f"{m['A_cert']} ({100*m['A_cert_cp']:.1f}%) | {m['B']} "
                 f"({100*m['B_cp']:.1f}%) | {m['A_all']} ({100*m['A_all_cp']:.1f}%) |")

    if out["train"]:
        L.append("\n## 5. Train P-class check (delta_g=0.13, pi/10)\n")
        L.append("> train 只评估了 S3 的 P 类 1364 个（占 train 6000 个 vinf 样本的 22.7%）；"
                 "表中 cert% 是**条件认证率**（=100%），不是全体认证率；本节只用于 "
                 "G4 的 n_cert≥1000 与条件违例率。\n")
        L.append("| gamma | n_cert | cert% | A_cert (CP95) | B (CP95) | A_all (CP95) |")
        L.append("|---|---|---|---|---|---|")
        for name, m in out["train"].items():
            L.append(f"| {m['gamma']:.2f} | {m['n_cert']} | {100*m['cert_rate']:.1f}% | "
                     f"{m['A_cert']} ({100*m['A_cert_cp']:.1f}%) | {m['B']} "
                     f"({100*m['B_cp']:.1f}%) | {m['A_all']} ({100*m['A_all_cp']:.1f}%) |")

    text = "\n".join(L)
    open(os.path.join(RES, "w4_decision.md"), "w", encoding="utf-8").write(text)
    json.dump(out, open(os.path.join(RES, "w4_decision.json"), "w"), indent=1, default=float)
    print(text)
    print("\nsaved", os.path.join(RES, "w4_decision.md"))


if __name__ == "__main__":
    main()
