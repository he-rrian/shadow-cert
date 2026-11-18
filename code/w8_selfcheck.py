"""w8_selfcheck.py -- same-distribution self-check for the two-level conformal.

Split the TEST split 50/50 (scenes), 20 random repeats:
  calibrate tau on one half, evaluate scene success on the other half.
Under strict exchangeability the algorithm should hit ~1-alpha.
"""
import os, sys, json, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from w8_conformal import collect, eval_cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--alphas", default="0.05,0.10,0.20")
    ap.add_argument("--betas", default="0.05")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scenes = collect(a.ckpt, a.split, dev)
    rng = np.random.default_rng(20261119)
    out = dict(ckpt=a.ckpt, split=a.split, n=len(scenes), reps=a.reps, rows=[])
    for al in [float(x) for x in a.alphas.split(",")]:
        for be in [float(x) for x in a.betas.split(",")]:
            vals = []
            for r in range(a.reps):
                idx = rng.permutation(len(scenes))
                h = len(idx) // 2
                ca = [scenes[i] for i in idx[:h]]
                te = [scenes[i] for i in idx[h:]]
                res = eval_cal(ca, te, al, be)
                vals.append(res["global_"]["scene_success"])
            out["rows"].append(dict(alpha=al, beta=be, mean=float(np.mean(vals)),
                                    std=float(np.std(vals, ddof=1)), values=[float(v) for v in vals]))
            print(f"alpha={al:.2f} beta={be:.2f} self-check success={np.mean(vals):.4f} "
                  f"std={np.std(vals, ddof=1):.4f} target={1-al:.2f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
