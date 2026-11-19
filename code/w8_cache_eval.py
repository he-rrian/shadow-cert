"""w8_cache_eval.py -- GPU-free W8 from exported prediction caches.

Reads data/preds/seed{s}_{split}.npz (psi,sigma,dead,sid,y,branch_id)
and GT phi/dead from data/{split}.  Recomputes labels at the requested gamma
(SF uses gamma=0.20, MATLAB gamma=0.75).
"""
import os, sys, json, argparse, math
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from w5_dataset import label_from_meta
from w8_conformal import eval_cal, pixel_baseline, q_high, tau_for
from w8_rcps import curves

ROOT = os.path.dirname(HERE)
PHI_TOL = math.pi / 10.0


def source_phi_meta(split):
    d = os.path.join(ROOT, "data", split)
    manp = os.path.join(d, "manifest.json" if split.startswith("sf_") else "manifest_000.json")
    man = json.load(open(manp))
    phis, metas = [], []
    for s in man["shards"]:
        z = np.load(os.path.join(d, s["path"]))
        phis.append(z["phi"].astype(np.float32))
        metas.extend(z["meta"])
    return np.concatenate(phis), metas


def load_scenes(pred, split, gamma):
    z = np.load(pred)
    psi_all = z["psi"]; sig_all = z["sigma"]; dead_all = z["dead"]
    sid_all = z["sid"]; br_all = z["branch_id"]
    phi, metas = source_phi_meta(split)
    n = len(z["psi"])
    assert len(phi) == n, (len(phi), n)
    scenes = []
    for i in range(n):
        m = json.loads(metas[i])
        y, _ = label_from_meta(m, gamma)
        dead = dead_all[i].astype(bool)
        if not dead.any():
            continue
        e = np.abs(psi_all[i].astype(np.float32) - phi[i]).astype(np.float64)[dead]
        s = np.maximum(sig_all[i].astype(np.float32).astype(np.float64)[dead], 1e-6)
        scenes.append(dict(sid=int(sid_all[i]), y=int(y), branch=int(br_all[i]),
                           e=e, s=s, score=e / s))
    return scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="data/preds")
    ap.add_argument("--out", default="probes/results/w8_5seeds.json")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    rows = []
    for s in seeds:
        val = load_scenes(os.path.join(a.pred, f"seed{s}_val.npz"), "val", 0.75)
        test = load_scenes(os.path.join(a.pred, f"seed{s}_test.npz"), "test", 0.75)
        sfv = load_scenes(os.path.join(a.pred, f"seed{s}_sf_val.npz"), "sf_val", 0.20)
        sft = load_scenes(os.path.join(a.pred, f"seed{s}_sf_test.npz"), "sf_test", 0.20)
        rec = dict(seed=s, n_val=len(val), n_test=len(test), conformal=[], rcps=[])
        for al in (0.05, 0.10, 0.20):
            r = eval_cal(val, test, al, 0.05)
            r["pixel_baseline"] = pixel_baseline(val, test, al, 0.05)
            rec["conformal"].append(r)
            rec["rcps"].append(curves(val, test, al))
        # SF target recalibrated
        sf = eval_cal(sfv, sft, 0.05, 0.05)
        sf["pixel_baseline"] = pixel_baseline(sfv, sft, 0.05, 0.05)
        rec["sf_target"] = sf
        rows.append(rec)
        print(f"seed{s}: conf .05/.10/.20 = "
              f"{rec['conformal'][0]['global_']['scene_success']:.3f}/"
              f"{rec['conformal'][1]['global_']['scene_success']:.3f}/"
              f"{rec['conformal'][2]['global_']['scene_success']:.3f}; "
              f"RCPS .05 acc={rec['rcps'][0]['chosen_test']['acc_frac']:.3f} "
              f"R={rec['rcps'][0]['chosen_test']['r_pix']:.4f}; "
              f"SF={sf['global_']['scene_success']:.3f}", flush=True)
    # mean/std summary
    summary = {}
    for key in ("conf", "rcps", "sf"):
        pass
    def mvals(fn):
        v = [fn(r) for r in rows]
        return dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v)>1 else 0.0, values=[float(x) for x in v])
    summary["conf"] = {
        "a05": mvals(lambda r: r["conformal"][0]["global_"]["scene_success"]),
        "a10": mvals(lambda r: r["conformal"][1]["global_"]["scene_success"]),
        "a20": mvals(lambda r: r["conformal"][2]["global_"]["scene_success"]),
        "pixel05": mvals(lambda r: r["conformal"][0]["pixel_baseline"]["scene_success"]),
    }
    summary["rcps"] = {}
    for i, al in enumerate((0.05, 0.10, 0.20)):
        summary["rcps"][f"a{int(al*100):02d}"] = dict(
            acc=mvals(lambda r, i=i: r["rcps"][i]["chosen_test"]["acc_frac"]),
            risk=mvals(lambda r, i=i: r["rcps"][i]["chosen_test"]["r_pix"]),
            cal_acc=mvals(lambda r, i=i: r["rcps"][i]["chosen"]["acc_frac"]),
            cal_risk=mvals(lambda r, i=i: r["rcps"][i]["chosen"]["r_pix"]),
        )
    summary["sf_target"] = mvals(lambda r: r["sf_target"]["global_"]["scene_success"])
    summary["sf_p"] = mvals(lambda r: r["sf_target"]["classes"]["P"]["scene_success"])
    summary["sf_u"] = mvals(lambda r: r["sf_target"]["classes"]["U"]["scene_success"])
    json.dump(dict(seeds=rows, summary=summary), open(os.path.join(ROOT, a.out), "w"), indent=1)
    print("summary:", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

