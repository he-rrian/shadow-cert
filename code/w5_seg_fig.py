"""w5_seg_fig.py -- quick visual check of the dead-segmentation head."""
import os, sys, argparse
import numpy as np, torch, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w5_seg import SegNet, SegDS, collate
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="runs/w5_seg/best.pth")
ap.add_argument("--out", default="probes/results/w5_figs/seg_qual.png")
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
m = SegNet().cuda(); m.load_state_dict(ck["model"]); m.eval()
rows = []
for split in ("val", "sf_val"):
    ds = SegDS(split)
    # first sliver sample
    idx = next(i for i in range(len(ds)) if ds[i]["branch"] == 0)
    b = ds[idx]
    x = torch.from_numpy(b["x"][None]).cuda()
    with torch.no_grad():
        prob = torch.sigmoid(m(x)).cpu().numpy()[0, 0]
    pred = prob > 0.5
    pred = cv2.dilate(pred.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    rows.append((split, b["x"][0], b["y"][0] > 0.5, prob, pred))

fig, axes = plt.subplots(len(rows), 4, figsize=(9, 2.6 * len(rows)))
for i, (name, I, gt, prob, pred) in enumerate(rows):
    for j, (im, cm, title) in enumerate([
            (I, "gray", f"{name}: I"),
            (gt, "gray", "GT dead"),
            (prob, "viridis", "dead_prob"),
            (pred, "gray", "pred (1 px dilate)")]):
        ax = axes[i, j]; ax.imshow(im, cmap=cm); ax.axis("off"); ax.set_title(title, fontsize=9)
fig.tight_layout(); fig.savefig(a.out, dpi=170); print("saved", a.out)
