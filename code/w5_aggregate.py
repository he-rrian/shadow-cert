"""w5_aggregate.py -- 5-seed mean +/- std for the P0 fair-tier G5."""
import os, json, glob
import numpy as np

ROOT = "probes/results/w5_5seeds"
VINF = ("sliver", "arc", "blob")


def med(rows, key):
    return float(np.median([r[key] for r in rows])) if rows else float("nan")


def main():
    seeds = []
    for s in range(5):
        d = os.path.join(ROOT, f"seed{s}")
        gv = os.path.join(d, "gate_val.json")
        rv = os.path.join(d, "rows_val.json")
        if not os.path.exists(gv) or not os.path.exists(rv):
            continue
        gate = json.load(open(gv)); rows = json.load(open(rv))
        rec = dict(seed=s, n=len(rows), ours=gate["ours"], bih=gate["bih"], orr=gate["orr"],
                   ratio_bih=gate["overall_ratio_bih"], cover=gate["cover"],
                   p95_h=gate["p95_h"], har_sf=gate["har_sf"], bih_sf=gate["bih_sf"],
                   ratio_fair=gate["ours"] / max(gate["bih_sf"], 1e-9),
                   residual_h=gate["residual_frac_h"])
        for b in VINF:
            rr = [r for r in rows if r["branch"] == b]
            rec[f"{b}_ours"] = med(rr, "p95")
            rec[f"{b}_bih_sf"] = med(rr, "bih_sf")
            rec[f"{b}_fair"] = rec[f"{b}_ours"] / max(rec[f"{b}_bih_sf"], 1e-9)
            rec[f"{b}_orr"] = med(rr, "orr")
            rec[f"{b}_ratio_orr"] = rec[f"{b}_ours"] / max(rec[f"{b}_orr"], 1e-9)
        gs = os.path.join(d, "gate_sf_test.json")
        if os.path.exists(gs):
            g = json.load(open(gs)); rec["sf_ours"] = g["ours"]
            rec["sf_drop"] = (g["ours"] - gate["ours"]) / max(gate["ours"], 1e-9)
        seeds.append(rec)
    if not seeds:
        print("no seed results"); return
    keys = [k for k in seeds[0] if k not in ("seed",)]
    print(f"n_seeds={len(seeds)}")
    print(f"{'metric':22s} {'mean':>9s} {'std':>9s}  per-seed")
    for k in keys:
        vals = [r[k] for r in seeds if k in r and np.isfinite(r[k])]
        if not vals:
            continue
        print(f"{k:22s} {np.mean(vals):9.4f} {np.std(vals, ddof=1) if len(vals) > 1 else 0:9.4f}  "
              + " ".join(f"{v:.4f}" for v in vals))
    json.dump(dict(seeds=seeds, n=len(seeds)), open(os.path.join(ROOT, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
