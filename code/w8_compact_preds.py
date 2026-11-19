"""w8_compact_preds.py -- compact dead-only prediction cache for W8/W9.

Input : data/preds/seed{s}_{split}.npz (full psi/sigma)
Output: data/preds_compact/seed{s}_{split}.npz with only GT-dead pixels:
        e, sigma, sid, y75, y20, branch_id, idx, n_dead
This is enough for W8 conformal/RCPS and W9 phase metrics; full psi stays on
the server for optional qualitative export.
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from w5_dataset import label_from_meta


def flips(split):
    d = os.path.join(ROOT, "data", "frontend", split)
    man = json.load(open(os.path.join(d, "frontend_manifest.json")))
    out = []
    for s in man["shards"]:
        z = np.load(os.path.join(d, s["path"])); out.append(z["flip"].astype(bool))
    return np.concatenate(out)


def source(split):
    d = os.path.join(ROOT, "data", split)
    manp = os.path.join(d, "manifest.json" if split.startswith("sf_") else "manifest_000.json")
    man = json.load(open(manp))
    phis, metas = [], []
    for s in man["shards"]:
        z = np.load(os.path.join(d, s["path"]))
        phis.append(z["phi"].astype(np.float32)); metas.extend(z["meta"])
    return np.concatenate(phis), metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="data/preds")
    ap.add_argument("--out", default="data/preds_compact")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--splits", default="val,test,sf_val,sf_test")
    a = ap.parse_args()
    for s in [int(x) for x in a.seeds.split(",")]:
        for split in a.splits.split(","):
            z = np.load(os.path.join(a.pred, f"seed{s}_{split}.npz"))
            psi_all = z["psi"]; sig_all = z["sigma"]; dead_all = z["dead"]
            sid_all = z["sid"]; br_all = z["branch_id"]; idx_all = z["idx"]
            phi, metas = source(split)
            flip = flips(split)
            E, S, SID, Y75, Y20, BR, IDX, ND = [], [], [], [], [], [], [], []
            for i in range(len(z["psi"])):
                m = json.loads(metas[i])
                dead = dead_all[i].astype(bool)
                if not dead.any():
                    continue
                pi = phi[i][:, ::-1] if flip[i] else phi[i]
                e = np.abs(psi_all[i].astype(np.float32) - pi).astype(np.float32)[dead]
                sig = np.maximum(sig_all[i].astype(np.float32), 1e-6)[dead]
                y75, _ = label_from_meta(m, 0.75)
                y20, _ = label_from_meta(m, 0.20)
                E.append(e); S.append(sig); SID.append(int(z["sid"][i]))
                Y75.append(int(y75)); Y20.append(int(y20)); BR.append(int(br_all[i]))
                IDX.append(int(idx_all[i])); ND.append(int(dead.sum()))
            os.makedirs(a.out, exist_ok=True)
            out = os.path.join(a.out, f"seed{s}_{split}.npz")
            np.savez_compressed(out, e=np.concatenate(E), sigma=np.concatenate(S),
                                n_dead=np.asarray(ND, np.int32),
                                sid=np.asarray(SID, np.int32), y75=np.asarray(Y75, np.int8),
                                y20=np.asarray(Y20, np.int8), branch_id=np.asarray(BR, np.int8),
                                idx=np.asarray(IDX, np.int32))
            print(f"seed{s}/{split}: {len(ND)} scenes, {sum(ND)} dead px -> {out}", flush=True)


if __name__ == "__main__":
    main()


