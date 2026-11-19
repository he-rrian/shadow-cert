"""w8_compact_eval.py -- 5-seed W8 from dead-only compact caches (no GPU)."""
import os, sys, json, argparse, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from w8_conformal import eval_cal, pixel_baseline
from w8_rcps import curves


def load(seed, split, gamma, d):
    z = np.load(os.path.join(d, f"seed{seed}_{split}.npz"))
    e = z["e"].astype(np.float64); s = np.maximum(z["sigma"].astype(np.float64), 1e-6)
    n = z["n_dead"].astype(int); off = np.r_[0, np.cumsum(n)]
    y = z["y75"] if gamma == 0.75 else z["y20"]
    scenes = []
    for i in range(len(n)):
        if n[i] == 0:
            continue
        ee = e[off[i]:off[i + 1]]; ss = s[off[i]:off[i + 1]]
        scenes.append(dict(sid=int(z["sid"][i]), y=int(y[i]), branch=int(z["branch_id"][i]),
                           e=ee, s=ss, score=ee / ss))
    return scenes


def vals(fn, rows):
    v = [fn(r) for r in rows]
    return dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                values=[float(x) for x in v])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", default="data/preds_compact")
    ap.add_argument("--out", default="probes/results/w8_5seeds_compact.json")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args()
    rows = []
    for s in [int(x) for x in a.seeds.split(",")]:
        val = load(s, "val", 0.75, a.d); test = load(s, "test", 0.75, a.d)
        sfv = load(s, "sf_val", 0.20, a.d); sft = load(s, "sf_test", 0.20, a.d)
        conf = [eval_cal(val, test, al, 0.05) for al in (0.05, 0.10, 0.20)]
        for al, c in zip((0.05, 0.10, 0.20), conf):
            c["pixel_baseline"] = pixel_baseline(val, test, al, 0.05)
        rc = [curves(val, test, al) for al in (0.05, 0.10, 0.20)]
        sf = eval_cal(sfv, sft, 0.05, 0.05)
        sf["pixel_baseline"] = pixel_baseline(sfv, sft, 0.05, 0.05)
        rows.append(dict(seed=s, conf=conf, rcps=rc, sf=sf))
        print(f"seed{s}: conf {conf[0]['global_']['scene_success']:.3f}/"
              f"{conf[1]['global_']['scene_success']:.3f}/{conf[2]['global_']['scene_success']:.3f}; "
              f"rcps acc {rc[0]['chosen_test']['acc_frac']:.3f} R {rc[0]['chosen_test']['r_pix']:.4f}; "
              f"sf {sf['global_']['scene_success']:.3f}", flush=True)
    summ = dict(
        conf05=vals(lambda r: r["conf"][0]["global_"]["scene_success"], rows),
        conf10=vals(lambda r: r["conf"][1]["global_"]["scene_success"], rows),
        conf20=vals(lambda r: r["conf"][2]["global_"]["scene_success"], rows),
        pixel05=vals(lambda r: r["conf"][0]["pixel_baseline"]["scene_success"], rows),
        rcps05_acc=vals(lambda r: r["rcps"][0]["chosen_test"]["acc_frac"], rows),
        rcps05_risk=vals(lambda r: r["rcps"][0]["chosen_test"]["r_pix"], rows),
        rcps10_acc=vals(lambda r: r["rcps"][1]["chosen_test"]["acc_frac"], rows),
        rcps10_risk=vals(lambda r: r["rcps"][1]["chosen_test"]["r_pix"], rows),
        rcps20_acc=vals(lambda r: r["rcps"][2]["chosen_test"]["acc_frac"], rows),
        rcps20_risk=vals(lambda r: r["rcps"][2]["chosen_test"]["r_pix"], rows),
        sf=vals(lambda r: r["sf"]["global_"]["scene_success"], rows),
        sfP=vals(lambda r: r["sf"]["classes"]["P"]["scene_success"], rows),
        sfU=vals(lambda r: r["sf"]["classes"]["U"]["scene_success"], rows),
    )
    json.dump(dict(seeds=rows, summary=summ), open(os.path.join(ROOT, a.out), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
