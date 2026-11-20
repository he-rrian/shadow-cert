"""w8_export_preds.py -- export frozen S5 predictions for GPU-free W8/W9.

For each seed and split (val, test, sf_test), save:
  psi   float16 (N,256,256)
  sigma float16 (N,256,256)
  dead  bool
  sid, y, branch_id, idx
GT phi/dead are already in data/{split}; no GPU is needed afterwards.
"""
import os, sys, argparse, time
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from w5_dataset import W5Dataset
from w5_model import W5UNet, split_output, make_input


def collate(batch):
    keys = ("I", "h", "M", "roi", "dead", "phi", "carrier", "sid", "idx", "branch_id", "y")
    return {k: torch.from_numpy(np.stack([b[k] for b in batch])) for k in keys}


@torch.no_grad()
def export(ckpt, split, out, device, no_phasor=False):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = W5UNet(base=128, cin=3 if no_phasor else 7).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    ds = W5Dataset(split, gamma=0.75, augment=False, light=True)
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate,
                    pin_memory=True, persistent_workers=True)
    P = []; S = []; D = []; SID = []; Y = []; BR = []; IDX = []
    t0 = time.time()
    for b in ld:
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        dpsi, log_sigma, _, _, _ = split_output(m(make_input(b, use_phasor=not no_phasor)))
        psi = (b["h"].float().unsqueeze(1) + b["M"].float().unsqueeze(1) * dpsi).squeeze(1)
        sig = torch.exp(0.5 * log_sigma).squeeze(1)
        P.append(psi.half().cpu().numpy()); S.append(sig.half().cpu().numpy())
        D.append(b["dead"].cpu().numpy().astype(bool))
        SID.append(b["sid"].cpu().numpy()); Y.append(b["y"].cpu().numpy())
        BR.append(b["branch_id"].cpu().numpy()); IDX.append(b["idx"].cpu().numpy())
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, psi=np.concatenate(P), sigma=np.concatenate(S),
                        dead=np.concatenate(D), sid=np.concatenate(SID),
                        y=np.concatenate(Y), branch_id=np.concatenate(BR),
                        idx=np.concatenate(IDX))
    print(f"{split}: {len(P)} batches -> {out} ({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--out", default="data/preds")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--splits", default="val,test,sf_test")
    ap.add_argument("--no-phasor", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for s in [int(x) for x in a.seeds.split(",")]:
        ck = os.path.join(a.root, f"w5_p0_seed{s}", "best_stage3.pth")
        if not os.path.exists(ck):
            # single-run W10 layout: runs/<variant>_seed{s}/best_stage3.pth
            ck = os.path.join(a.root, "best_stage3.pth")
        for sp in a.splits.split(","):
            export(ck, sp, os.path.join(a.out, f"seed{s}_{sp}.npz"), device,
                   no_phasor=a.no_phasor)


if __name__ == "__main__":
    main()

