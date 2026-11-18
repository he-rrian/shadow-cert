"""w8_precheck.py -- 2-minute W8 readiness check on a trained S5 checkpoint.

Checks:
  1. sigma head: Spearman(|psi-phi|, sigma) on GT-dead pixels (per-sample median + pooled)
     and median sigma by shape.
  2. cls head: P-class recall and confusion matrix on val.
  3. checkpoint metadata: epoch/stage, to make sure best_stage3 comes from stage 3.
"""
import os, sys, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from w5_dataset import W5Dataset, BRANCH_ID
from w5_model import W5UNet, split_output, make_input

NAMES = {v: k for k, v in BRANCH_ID.items()}


def collate(batch):
    keys = ("I", "h", "M", "roi", "dead", "phi", "orr", "clean", "carrier", "y", "rho",
            "sid", "idx", "branch_id", "phi_obs", "dead_pred")
    return {k: torch.from_numpy(np.stack([b[k] for b in batch])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/w5_p0_seed0/last.pth")
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    print("checkpoint meta:", {k: ck.get(k) for k in ("epoch", "stage", "val_phase", "params_m")})
    m = W5UNet(base=128).to(device); m.load_state_dict(ck["model"]); m.eval()
    ds = W5Dataset(a.split, gamma=0.75, augment=False)
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate)
    per_rho = []; per_rho_b = {b: [] for b in NAMES.values()}; scene_e = []; scene_s = []
    pooled_e = []; pooled_s = []; sig_by = {b: [] for b in NAMES.values()}
    y_true = []; y_pred = []; n_dead = 0
    with torch.no_grad():
        for batch in ld:
            b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            dpsi, log_sigma, _, _, logits = split_output(m(make_input(b)))
            psi = (b["h"].float().unsqueeze(1) + b["M"].float().unsqueeze(1) * dpsi).squeeze(1)
            sigma = torch.exp(0.5 * log_sigma).squeeze(1)   # log_sigma is log VARIANCE; report std
            e = (psi - b["phi"].float()).abs()
            dead = b["dead"].bool()
            y_true.append(b["y"].cpu().numpy())
            y_pred.append(logits.mean(dim=(2, 3)).argmax(1).cpu().numpy())
            for j in range(len(psi)):
                d = dead[j]
                if not d.any():
                    continue
                ee = e[j][d].cpu().numpy(); ss = sigma[j][d].cpu().numpy()
                n_dead += len(ee)
                pooled_e.append(ee); pooled_s.append(ss)
                if len(ee) > 8 and np.std(ee) > 1e-8 and np.std(ss) > 1e-8:
                    per_rho.append(float(spearmanr(ee, ss).correlation))
                bid = int(batch["branch_id"][j].item())
                sig_by[NAMES.get(bid, "?")].append(float(np.median(ss)))
                scene_e.append(float(np.median(ee))); scene_s.append(float(np.median(ss)))
    yt = np.concatenate(y_true); yp = np.concatenate(y_pred)
    out = {"ckpt": a.ckpt, "n": int(len(yt)), "n_dead_px": int(n_dead),
           "sigma_spearman_per_sample_median": float(np.nanmedian(per_rho)),
           "sigma_spearman_per_sample_mean": float(np.nanmean(per_rho)),
           "sigma_spearman_pooled": float(spearmanr(np.concatenate(pooled_e), np.concatenate(pooled_s)).correlation),
           "sigma_median_by_shape": {b: float(np.median(v)) for b, v in sig_by.items() if v},
           "sigma_spearman_by_shape": {b: float(np.nanmedian(v)) for b, v in per_rho_b.items() if v},
           "scene_spearman_error_vs_sigma": float(spearmanr(scene_e, scene_s).correlation) if len(scene_e) > 5 else None,
           "p_recall": float((yp[yt == 0] == 0).mean()) if (yt == 0).any() else None,
           "u_recall": float((yp[yt == 1] == 1).mean()) if (yt == 1).any() else None,
           "r_recall": float((yp[yt == 2] == 2).mean()) if (yt == 2).any() else None,
           "acc": float((yt == yp).mean()),
           "confusion": [[int(((yt == i) & (yp == j)).sum()) for j in range(3)] for i in range(3)]}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.splitext(a.ckpt)[0] + "_w8_precheck.json", "w"), indent=1)


if __name__ == "__main__":
    main()


