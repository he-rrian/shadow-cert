"""w5_eval.py -- S5 G5 evaluation + figures.

G5 gates (doc 濠电偛鐡ㄩ崵搴ㄥ磹閺嶎偅鏆滅痪鎵檮4闂備礁鎲＄敮鐐测枍? 闂?.3):
  G5-a overall : median P95(Ours) / median P95(Biharm) <= 0.95
  G5-a shape   : sliver Ours/ORR <= 1.05; arc/blob Ours/Biharm <= 1.05
  G5-b SF      : zero-shot error drop <= 20% (SF-test vs MATLAB val)
  G5-c residual: ||Ours-ORR|| < ||Harmonic-ORR|| on >= 80% of samples
  G5-d         : params / training time / memory / 5 seeds record

Run:
  python code/w5_eval.py --ckpt runs/w5_seed0/best.pth --splits val,test,sf_test --out probes/results/w5_g5_seed0
"""
import os, sys, json, argparse, time, importlib.util
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from w5_dataset import W5Dataset                                  # noqa: E402
from w5_model import W5UNet, split_output, make_input             # noqa: E402

spec = importlib.util.spec_from_file_location("w4", os.path.join(HERE, "w4_baseline.py"))
W4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(W4)
DELTA_G = 0.13


def collate(batch):
    keys = ("I", "h", "M", "roi", "dead", "phi", "orr", "clean", "carrier", "y", "rho",
            "sid", "idx", "branch_id", "phi_obs", "dead_pred")
    return {k: torch.from_numpy(np.stack([b[k] for b in batch])) for k in keys}

BRANCH_NAMES = ["sliver", "arc", "blob", "edge", "cut", "degraded"]


def baseline_one(phi, orr, dead, roi, key):
    rng = np.random.default_rng(int(key) + 20261117)
    src = phi + rng.normal(0.0, DELTA_G / 2.0, phi.shape) * (~dead)
    har = W4.harmonic_fill(dead, src)
    try:
        bih = W4.biharmonic_fill(dead, src)
    except Exception:
        bih = har
    e = lambda a: np.abs(np.asarray(a, np.float64)[dead] - phi[dead])
    return dict(key=int(key), har=float(np.percentile(e(har), 95)),
                bih=float(np.percentile(e(bih), 95)),
                orr=float(np.percentile(e(orr), 95)),
                l2_har=float(np.sqrt(np.mean((har[dead] - orr[dead]) ** 2))),
                l2_bih=float(np.sqrt(np.mean((bih[dead] - orr[dead]) ** 2))))


def fair_one(po, dp, roi, phi, dead):
    """single-frame fair baselines: observed boundary + learned dead mask, no GT."""
    if not dp.any():
        return dict(har_sf=np.nan, bih_sf=np.nan)
    src = np.where(dp, 0.0, po)
    har = W4.harmonic_fill(dp, src)
    try:
        bih = W4.biharmonic_fill(dp, po)
    except Exception:
        bih = har
    e = lambda a: np.abs(np.asarray(a, np.float64)[dead] - phi[dead])
    return dict(har_sf=float(np.percentile(e(har), 95)),
                bih_sf=float(np.percentile(e(bih), 95)))


@torch.no_grad()
def eval_split(ckpt, split, device="cuda", max_samples=0, baseline_cache=None, qual_dir=None):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = W5UNet(base=128).to(device); model.load_state_dict(ck["model"]); model.eval()
    ds = W5Dataset(split, gamma=0.75, augment=False)
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate,
                    pin_memory=True, persistent_workers=True)
    base = {}
    if baseline_cache and os.path.exists(baseline_cache):
        base = {int(r["key"]): r for r in json.load(open(baseline_cache))}
    rows = []
    qual = []
    t0 = time.time()
    for bi, batch in enumerate(ld):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(make_input(batch))
        dpsi, log_sigma, _, _, logits = split_output(out)
        psi = (batch["h"].float().unsqueeze(1) + batch["M"].float().unsqueeze(1) * dpsi).squeeze(1).cpu().numpy()
        phi = batch["phi"].cpu().numpy(); orr = batch["orr"].cpu().numpy()
        dead = batch["dead"].cpu().numpy().astype(bool); roi = batch["roi"].cpu().numpy().astype(bool)
        I = batch["I"].cpu().numpy(); h = batch["h"].cpu().numpy(); M = batch["M"].cpu().numpy()
        po = batch["phi_obs"].cpu().numpy(); dp = batch["dead_pred"].cpu().numpy().astype(bool)
        y = batch["y"].cpu().numpy()
        for j in range(len(psi)):
            ssid = int(batch["sid"][j].item())
            idx = int(batch["idx"][j].item())
            br = BRANCH_NAMES[int(batch["branch_id"][j].item())]
            if br not in ("sliver", "arc", "blob"):
                continue                    # G5 only scores the 750 vinf samples
            if not dead[j].any():
                continue
            e = np.abs(psi[j][dead[j]] - phi[j][dead[j]])
            p95 = float(np.percentile(e, 95)); emax = float(e.max()); emean = float(e.mean())
            cov = dead[j] & (M[j] >= 0.25)
            p95_cov = float(np.percentile(np.abs(psi[j][cov] - phi[j][cov]), 95)) if cov.any() else float("nan")
            cover = float(cov.sum() / max(dead[j].sum(), 1))
            if idx not in base:
                base[idx] = baseline_one(phi[j], orr[j], dead[j], roi[j], idx)
            b = base[idx]
            l2_ours = float(np.sqrt(np.mean((psi[j][dead[j]] - orr[j][dead[j]]) ** 2)))
            l2_h = float(np.sqrt(np.mean((h[j][dead[j]] - orr[j][dead[j]]) ** 2)))
            p95_h = float(np.percentile(np.abs(h[j][dead[j]] - phi[j][dead[j]]), 95))
            fair = fair_one(po[j], dp[j], roi[j], phi[j], dead[j])
            rows.append(dict(sid=ssid, branch=br, y=int(y[j]),
                             p95=p95, p95_cov=p95_cov, cover=cover,
                             max=emax, mean=emean, bih=b["bih"],
                             orr=b["orr"], har=b["har"], l2_ours=l2_ours,
                             l2_har=b["l2_har"], l2_bih=b["l2_bih"],
                             l2_h=l2_h, p95_h=p95_h, har_sf=fair["har_sf"], bih_sf=fair["bih_sf"]))
            if qual_dir and len(qual) < 6:
                qual.append(dict(I=I[j], phi=phi[j], psi=psi[j], h=h[j], M=M[j],
                                 dead=dead[j], roi=roi[j]))
        if max_samples and len(rows) >= max_samples:
            break
        if (bi + 1) % 20 == 0:
            print(f"  {split} batch {bi+1} rows {len(rows)} {time.time()-t0:.0f}s", flush=True)
    if baseline_cache and not os.path.exists(baseline_cache):
        json.dump(list(base.values()), open(baseline_cache, "w"), indent=1)
    if qual_dir:
        os.makedirs(qual_dir, exist_ok=True)
        np.savez(os.path.join(qual_dir, f"qual_{split}.npz"), **{k: np.stack([q[k] for q in qual])
                                                                  for k in qual[0]})
    print(f"{split}: {len(rows)} rows in {time.time()-t0:.0f}s", flush=True)
    return rows


def ds_branch(split, sid):
    """branch from the dataset cache is not stored; read from source meta quickly."""
    # cached by caller via a helper dict
    return BRANCH_BY_SID.get(int(sid), "")


def batch_sid(batch, j):
    return batch["sid"][j].item() if "sid" in batch else -1


def build_branch_map(split):
    _, man = __import__("w5_dataset").load_manifest(split)
    out = {}
    for s in man["shards"]:
        z = np.load(os.path.join(ROOT, "data", split, s["path"]))
        for m in z["meta"]:
            mm = json.loads(m)
            ss = str(mm["id"]).split("_")[0]
            if ss.startswith("SF"):
                ss = ss[2:]
            out[int(ss)] = mm["branch"]
    return out


BRANCH_BY_SID = {}


def gate(rows, name):
    import statistics
    by = {}
    for b in ("sliver", "arc", "blob"):
        rr = [r for r in rows if r["branch"] == b]
        by[b] = dict(n=len(rr),
                     ours=float(np.median([r["p95"] for r in rr])) if rr else None,
                     ours_cov=float(np.nanmedian([r["p95_cov"] for r in rr])) if rr else None,
                     cover=float(np.mean([r["cover"] for r in rr])) if rr else None,
                     bih=float(np.median([r["bih"] for r in rr])) if rr else None,
                     orr=float(np.median([r["orr"] for r in rr])) if rr else None)
        if rr:
            by[b]["ratio_orr"] = by[b]["ours"] / max(by[b]["orr"], 1e-9)
            by[b]["ratio_bih"] = by[b]["ours"] / max(by[b]["bih"], 1e-9)
    allr = rows
    out = dict(name=name, n=len(allr),
               ours=float(np.median([r["p95"] for r in allr])),
               ours_cov=float(np.nanmedian([r["p95_cov"] for r in allr])),
               cover=float(np.mean([r["cover"] for r in allr])),
               bih=float(np.median([r["bih"] for r in allr])),
               orr=float(np.median([r["orr"] for r in allr])),
               har=float(np.median([r["har"] for r in allr])),
               overall_ratio_bih=float(np.median([r["p95"] for r in allr]) /
                                       max(np.median([r["bih"] for r in allr]), 1e-9)),
               residual_frac=float(np.mean([r["l2_ours"] < r["l2_har"] for r in allr])),
               residual_frac_h=float(np.mean([r["l2_ours"] < r["l2_h"] for r in allr])),
               p95_h=float(np.median([r["p95_h"] for r in allr])),
               har_sf=float(np.median([r["har_sf"] for r in allr])),
               bih_sf=float(np.median([r["bih_sf"] for r in allr])),
               shapes=by)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", default="val,test,sf_test")
    ap.add_argument("--out", default="probes/results/w5_g5")
    ap.add_argument("--max-samples", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for split in a.splits.split(","):
        global BRANCH_BY_SID
        BRANCH_BY_SID = build_branch_map(split)
        cache = os.path.join(a.out, f"baseline_{split}.json")
        rows = eval_split(a.ckpt, split, device, a.max_samples, cache,
                          os.path.join(a.out, "qual"))
        json.dump(rows, open(os.path.join(a.out, f"rows_{split}.json"), "w"), indent=1)
        results[split] = gate(rows, split)
        json.dump(results[split], open(os.path.join(a.out, f"gate_{split}.json"), "w"), indent=1)
        print(json.dumps(results[split], indent=1), flush=True)
    json.dump(results, open(os.path.join(a.out, "g5_all.json"), "w"), indent=1)


if __name__ == "__main__":
    main()





