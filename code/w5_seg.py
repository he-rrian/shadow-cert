"""w5_seg.py -- lightweight dead-zone segmentation head (S5 P0 fix).

Input : I/255, roi, m_hilb (analytic envelope / local P90, same as frontend C)
Target: GT dead
Output: dead_prob; then compact dead_pred is used to rebuild h_input.

Run:
  python code/w5_seg.py --train --epochs 8 --out runs/w5_seg
  python code/w5_seg.py --eval --ckpt runs/w5_seg/best.pth --split val,sf_val
"""
import os, sys, json, argparse, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from w5_dataset import W5Dataset, BRANCH_ID                     # noqa: E402
import w5_prep_frontend as PF                                  # noqa: E402


def hilbert_mrel(I, roi):
    """same analytic envelope + 55x55 local P90 as frontend C; returned in [0, 2.5]."""
    fy, fx, F, U, V = PF.carrier_from_image(I.astype(np.float64), roi)
    f0 = max(np.hypot(fx, fy), 1e-6); w = 0.5 * f0
    bm = np.exp(-((U - fx) ** 2 + (V - fy) ** 2) / (2 * (w / 2) ** 2))
    bm *= (np.hypot(U - fx, V - fy) < 2 * w)
    Z = np.fft.ifft2(np.fft.ifftshift(F * bm))
    m = np.abs(Z).astype(np.float32)
    mr = PF.local_p90(m, win=55)
    return np.clip(m / (mr + 1e-6), 0, 2.5).astype(np.float32)


class SegDS(Dataset):
    def __init__(self, split, augment=False):
        self.ds = W5Dataset(split, gamma=0.75, augment=False, light=True)
        self.augment = augment
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        b = self.ds[i]
        m = hilbert_mrel(b["I"], b["roi"])
        I = b["I"].astype(np.float32) / 255.0
        roi = b["roi"].astype(np.float32)
        dead = b["dead"].astype(np.float32)
        if self.augment and np.random.rand() < 0.5:
            I = I[::-1].copy(); roi = roi[::-1].copy(); dead = dead[::-1].copy(); m = m[::-1].copy()
        return dict(x=np.stack([I, roi, m], 0), y=dead[None], branch=np.int64(b["branch_id"]))


def collate(b):
    return dict(x=torch.from_numpy(np.stack([q["x"] for q in b])),
                y=torch.from_numpy(np.stack([q["y"] for q in b])),
                branch=torch.from_numpy(np.stack([q["branch"] for q in b])))


def gn(c, g=8):
    g = min(g, c)
    while c % g and g > 1:
        g -= 1
    return nn.GroupNorm(g, c)


class DS(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.dw = nn.Conv2d(a, a, 3, padding=1, groups=a, bias=False)
        self.pw = nn.Conv2d(a, b, 1, bias=False)
        self.n = gn(b); self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.n(self.pw(self.dw(x))))


class SegNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.e1 = DS(3, base); self.e2 = DS(base, base * 2); self.e3 = DS(base * 2, base * 4)
        self.p = nn.MaxPool2d(2)
        self.u2 = DS(base * 4 + base * 2, base * 2)
        self.u1 = DS(base * 2 + base, base)
        self.h = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.p(e1)); e3 = self.e3(self.p(e2))
        u2 = F.interpolate(e3, scale_factor=2, mode="bilinear", align_corners=False)
        u2 = self.u2(torch.cat([u2, e2], 1))
        u1 = F.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False)
        u1 = self.u1(torch.cat([u1, e1], 1))
        return self.h(u1)


def dice_loss(logit, y, eps=1.0):
    p = torch.sigmoid(logit)
    num = 2 * (p * y).sum(dim=(2, 3)) + eps
    den = p.sum(dim=(2, 3)) + y.sum(dim=(2, 3)) + eps
    return (1 - num / den).mean()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, split, device, thr=0.5, dilate=0):
    ds = SegDS(split); ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=6, collate_fn=collate)
    model.eval(); rows = []
    for b in ld:
        logit = model(b["x"].to(device))
        prob = torch.sigmoid(logit).cpu().numpy()[:, 0]
        y = b["y"].numpy()[:, 0].astype(bool); br = b["branch"].numpy()
        roi = b["x"].numpy()[:, 1].astype(bool)
        for j in range(len(prob)):
            pred = prob[j] > thr
            if dilate:
                pred = cv2.dilate(pred.astype(np.uint8),
                                  np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)) > 0
            # remove tiny specks, keep thin slivers
            n, lab, st, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), 8)
            for k in range(1, n):
                if st[k, cv2.CC_STAT_AREA] < 8:
                    pred[lab == k] = False
            clean_pred = roi[j] & (~pred)
            clean_gt = roi[j] & (~y[j])
            inter = (clean_pred & clean_gt).sum(); union = (clean_pred | clean_gt).sum()
            iou = float(inter / max(union, 1))
            miss = float(1 - (pred & y[j]).sum() / max(y[j].sum(), 1)) if y[j].any() else 0.0
            rows.append(dict(branch=int(br[j]), iou=iou, miss=miss, n_dead=int(pred.sum())))
    model.train()
    return rows


def summarize(rows, name):
    names = {v: k for k, v in BRANCH_ID.items()}
    out = {"split": name, "n": len(rows), "clean_iou": float(np.mean([r["iou"] for r in rows]))}
    for bid, nm in names.items():
        rr = [r for r in rows if r["branch"] == bid]
        if rr:
            out[nm] = dict(n=len(rr), iou=float(np.mean([r["iou"] for r in rr])),
                           miss=float(np.mean([r["miss"] for r in rr])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="runs/w5_seg")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--split", default="val,sf_val")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--dilate", type=int, default=0)
    a = ap.parse_args(); set_seed(0); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    model = SegNet().to(device)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.3f}M device={device}", flush=True)
    if a.train:
        ds = SegDS("train", augment=False)
        ld = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                        collate_fn=collate, pin_memory=True, persistent_workers=True)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
        best = 1e9
        for ep in range(a.epochs):
            t0 = time.time(); tot = 0.0; n = 0
            for step, b in enumerate(ld):
                x = b["x"].to(device); y = b["y"].to(device)
                opt.zero_grad(set_to_none=True)
                logit = model(x)
                bce = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(5.0, device=device))
                loss = bce + 0.5 * dice_loss(logit, y)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
                tot += float(loss) * len(x); n += len(x)
                if (step + 1) % 50 == 0:
                    print(f"  ep{ep+1} step{step+1} loss={float(loss):.4f}", flush=True)
            sched.step(); tr = tot / max(n, 1)
            rows = evaluate(model, "val", device, a.thr, a.dilate) if ep >= max(0, a.epochs - 3) else []
            sm = summarize(rows, "val") if rows else {}
            if sm and sm["clean_iou"] > -1:
                score = -sm["clean_iou"] + 2.0 * sm.get("sliver", {}).get("miss", 0)
                if score < best:
                    best = score
                    torch.save(dict(model=model.state_dict(), epoch=ep + 1, val=sm), os.path.join(a.out, "best.pth"))
            print(f"ep{ep+1}/{a.epochs} {time.time()-t0:.0f}s train={tr:.4f} val={json.dumps(sm)}", flush=True)
            json.dump(dict(epoch=ep + 1, train=tr, val=sm), open(os.path.join(a.out, "train_log.json"), "w"), indent=1)
        torch.save(dict(model=model.state_dict()), os.path.join(a.out, "last.pth"))
    if a.eval:
        ck = a.ckpt or os.path.join(a.out, "best.pth")
        c = torch.load(ck, map_location="cpu", weights_only=False); model.load_state_dict(c["model"])
        for sp in a.split.split(","):
            rows = evaluate(model, sp, device, a.thr, a.dilate)
            sm = summarize(rows, sp)
            print(json.dumps(sm, indent=1), flush=True)
            json.dump(sm, open(os.path.join(a.out, f"eval_{sp}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()


