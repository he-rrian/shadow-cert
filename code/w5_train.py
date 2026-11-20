"""w5_train.py -- S5 residual U-Net training (W5-W7).

Three stages, 130 epochs total (following doc 濠碘槅鍋嗘晶妤呪€﹂崶顬?_濠电偞鍨堕幐鎾磻閹剧粯鐓曢柡宥庡亜閸斿鏌″畝鈧崰搴ㄣ€呮總绋块敜闁?.md 闂?.5):
  stage 1 (30 ep): L_phase only
  stage 2 (40 ep): + 0.5 L_fc
  stage 3 (60 ep): + 0.1 L_nll + 0.1 L_cls + 0.05 L_smooth

Run on the GPU server:
  python code/w5_train.py --seed 0 --out runs/w5_seed0
  python code/w5_train.py --seed 0 --out runs/w10_nophasor_seed0 --no-phasor
  python code/w5_train.py --seed 0 --out runs/w10_nofc_seed0 --no-fc
  python code/w5_train.py --seed 0 --out runs/w10_hardsplit_seed0 --hard-split
"""
import os, sys, json, time, argparse, random
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from w5_dataset import W5Dataset, label_from_meta, load_manifest          # noqa: E402
from w5_model import W5UNet, split_output, make_input                     # noqa: E402


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate(batch):
    keys = ("I", "h", "M", "roi", "dead", "phi", "clean", "carrier", "y", "rho", "branch_id")
    out = {}
    for k in keys:
        out[k] = torch.from_numpy(np.stack([b[k] for b in batch]))
    return out


def class_weights(train_split, gamma):
    """inverse-frequency weights from the metadata (fast, no image loading)."""
    _, man = load_manifest(train_split)
    cnt = Counter()
    for s in man["shards"]:
        z = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", train_split, s["path"]))
        for m in z["meta"]:
            y, _ = label_from_meta(json.loads(m), gamma)
            cnt[y] += 1
    n = sum(cnt.values())
    w = np.array([n / (3 * max(cnt[i], 1)) for i in range(3)], np.float32)
    return w / w.mean(), dict(cnt)


def hard_split_weights(train_split, gamma, hard_ratio=0.5):
    """Per-sample weights for 1:1 hard-split oversampling (W10 ablation).

    hard = rho >= 1 at gamma (the W4 diagnostic definition of a hard scene).
    The main S5 table used plain mixed sampling (shuffle=True, no sampler); this
    ablation oversamples hard scenes to `hard_ratio` of every epoch.
    """
    _, man = load_manifest(train_split)
    flags = []
    for s in man["shards"]:
        z = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", train_split, s["path"]))
        for m in z["meta"]:
            _, rho = label_from_meta(json.loads(m), gamma)
            flags.append(bool(np.isfinite(rho) and rho >= 1.0))
    H = int(np.sum(flags)); N = len(flags) - H
    if H == 0 or N == 0:
        return None, dict(hard=H, normal=N)
    w = np.where(flags, hard_ratio / H, (1.0 - hard_ratio) / N).astype(np.float64)
    return w, dict(hard=H, normal=N)


def phase_loss(psi, phi, roi, dead, branch_id=None):
    """Loss on ROI only.

    Normal samples: dead pixels x3, 3 px transition x2, remaining ROI x1.
    cut samples (branch_id==4): each connected dead component is first de-pistoned
    by its own median (a single-frame 2*pi*k across the cut is unobservable); then
    the relative SmoothL1 is averaged over dead pixels only.
    """
    e = F.smooth_l1_loss(psi, phi, beta=0.05, reduction="none")
    roi_f = roi.float(); dead_f = dead.float()
    trans = (F.max_pool2d(dead_f, 7, 1, 3) > 0).float() * roi_f * (1.0 - dead_f)
    w = 1.0 + 2.0 * dead_f + 1.0 * trans
    if branch_id is not None:
        # cut: the 2*pi*k across the cut is unobservable in one frame.  Excluding it
        # from L_phase is the safe fallback explicitly allowed by the P0 plan; cut
        # remains a Class-U failure test at inference instead of poisoning training.
        keep = torch.ones_like(branch_id, dtype=torch.float32)
        keep[branch_id == 4] = 0.0                      # cut: unobservable piston
        keep[(branch_id == 3) | (branch_id == 5)] = 0.2  # edge/degraded: downweight
        keep = keep.view(-1, 1, 1, 1)
    else:
        keep = torch.ones_like(roi_f)
    return (e * w * roi_f * keep).sum() / (w * roi_f * keep).sum().clamp_min(1.0)


def forward_losses(model, batch, weights, cls_w, use_phasor=True):
    I = batch["I"].float().unsqueeze(1) / 255.0
    phi = batch["phi"].float().unsqueeze(1)
    h = batch["h"].float().unsqueeze(1)
    M = batch["M"].float().unsqueeze(1)
    roi = batch["roi"].float().unsqueeze(1)
    dead = batch["dead"].float().unsqueeze(1)
    clean = batch["clean"].float().unsqueeze(1)
    x = make_input(batch, use_phasor)
    out = model(x)
    dpsi, log_sigma, a_logit, b_logit, logits = split_output(out)
    psi = h + M * dpsi

    L_phase = phase_loss(psi, phi, roi, dead, batch.get("branch_id"))
    A = torch.sigmoid(a_logit)
    B = torch.sigmoid(b_logit)
    rec = torch.clamp(A + B * torch.cos(psi), 0.0, 1.0)
    L_fc = ((rec - I).abs() * clean).sum() / clean.sum().clamp_min(1.0)
    L_nll = ((0.5 * torch.exp(-log_sigma) * (psi - phi) ** 2 + 0.5 * log_sigma)
             * roi).sum() / roi.sum().clamp_min(1.0)
    L_cls = F.cross_entropy(logits.mean(dim=(2, 3)), batch["y"], weight=cls_w)
    dx = psi[:, :, :, 1:] - psi[:, :, :, :-1]
    dy = psi[:, :, 1:, :] - psi[:, :, :-1, :]
    mx = M[:, :, :, 1:] * M[:, :, :, :-1]
    my = M[:, :, 1:, :] * M[:, :, :-1, :]
    ix = (I[:, :, :, 1:] - I[:, :, :, :-1]).abs()
    iy = (I[:, :, 1:, :] - I[:, :, :-1, :]).abs()
    L_smooth = ((dx.abs() * mx * torch.exp(-10 * ix)).mean()
                + (dy.abs() * my * torch.exp(-10 * iy)).mean())
    total = (L_phase + weights["fc"] * L_fc + weights["nll"] * L_nll
             + weights["cls"] * L_cls + weights["smooth"] * L_smooth)
    return total, dict(phase=float(L_phase), fc=float(L_fc), nll=float(L_nll),
                       cls=float(L_cls), smooth=float(L_smooth)), psi


@torch.no_grad()
def validate(model, loader, device, use_phasor=True):
    model.eval(); tot = 0.0; n = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        x = make_input(batch, use_phasor)
        dpsi, _, _, _, _ = split_output(model(x))
        psi = batch["h"].float().unsqueeze(1) + batch["M"].float().unsqueeze(1) * dpsi
        tot += float(phase_loss(psi, batch["phi"].float().unsqueeze(1),
                                batch["roi"].float().unsqueeze(1),
                                batch["dead"].float().unsqueeze(1),
                                batch.get("branch_id"))) * len(batch["I"])
        n += len(batch["I"])
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs1", type=int, default=35)
    ap.add_argument("--epochs2", type=int, default=35)
    ap.add_argument("--epochs3", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--gamma", type=float, default=0.75)
    ap.add_argument("--out", default="")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke-test cap per epoch")
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--skip-val", action="store_true")
    # W10 ablations (3 items; logT / scale-aug are deliberately NOT mapped here)
    ap.add_argument("--no-phasor", action="store_true", help="7->3 input channels (I, roi, M)")
    ap.add_argument("--no-fc", action="store_true", help="L_fc weight = 0 in all stages")
    ap.add_argument("--hard-split", action="store_true", help="rho>=1 oversampling vs plain mixed")
    ap.add_argument("--hard-ratio", type=float, default=0.5, help="hard fraction per epoch (1:1 = 0.5)")
    a = ap.parse_args()
    set_seed(a.seed)
    out = a.out or os.path.join("runs", f"w5_seed{a.seed}")
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} seed={a.seed} out={out}", flush=True)

    ds_tr = W5Dataset("train", gamma=a.gamma, augment=False, light=True)   # vertical flip would need a carrier-fy sign fix; disabled for S5
    ds_va = W5Dataset("val", gamma=a.gamma, augment=False, light=True)
    print(f"train={len(ds_tr)} val={len(ds_va)}", flush=True)
    cw, cnt = class_weights("train", a.gamma)
    print(f"train class counts={cnt} weights={cw.round(3).tolist()}", flush=True)
    weights = dict(fc=0.0, nll=0.0, cls=0.0, smooth=0.0)
    cls_w = torch.tensor(cw, dtype=torch.float32, device=device)
    sampler = None; hs = None
    if a.hard_split:
        hw, hs = hard_split_weights("train", a.gamma, a.hard_ratio)
        if hw is None:
            raise SystemExit(f"hard-split sampler unavailable: {hs}")
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.DoubleTensor(hw), num_samples=len(ds_tr), replacement=True)
        print(f"hard-split sampler: hard={hs['hard']} normal={hs['normal']} "
              f"hard_ratio={a.hard_ratio}", flush=True)
    ld_tr = DataLoader(ds_tr, batch_size=a.batch, shuffle=(sampler is None), sampler=sampler,
                       num_workers=a.workers,
                       collate_fn=collate, pin_memory=True, persistent_workers=a.workers > 0,
                       prefetch_factor=2 if a.workers > 0 else None)
    ld_va = DataLoader(ds_va, batch_size=a.batch, shuffle=False, num_workers=max(2, a.workers // 2),
                       collate_fn=collate, pin_memory=True,
                       persistent_workers=a.workers > 0,
                       prefetch_factor=2 if a.workers > 0 else None)

    cin = 3 if a.no_phasor else 7
    model = W5UNet(base=128, cin=cin).to(device)
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params={nparam:.2f}M cin={cin} no_fc={a.no_fc} hard_split={a.hard_split}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    log = dict(args=vars(a), params_m=nparam, cin=cin, class_counts=cnt,
               class_weights=cw.tolist(), hard_split_counts=hs, stages=[], val=[])
    global_step = 0
    e1, e2, e3 = a.epochs1, a.epochs2, a.epochs3
    total_epochs = e1 + e2 + e3
    fc_w = 0.0 if a.no_fc else 0.5
    weights.update(dict(fc=0.0, nll=0.0, cls=0.0, smooth=0.05))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_epochs, 1), eta_min=a.lr * 0.02)
    best = 1e9; best3 = 1e9; prev_stage = 0
    for ep in range(total_epochs):
        stage = 1 if ep < e1 else (2 if ep < e1 + e2 else 3)
        if stage != prev_stage:
            if stage == 1:
                weights.update(dict(fc=0.0, nll=0.0, cls=0.0, smooth=0.05))
            elif stage == 2:
                weights.update(dict(fc=fc_w, nll=0.0, cls=0.0, smooth=0.05))
            else:
                weights.update(dict(fc=fc_w, nll=0.1, cls=0.1, smooth=0.05))
            print(f"=== stage {stage} starts at global epoch {ep+1}: weights={weights}", flush=True)
            prev_stage = stage
        t0 = time.time(); agg = Counter(); n = 0
        for step, batch in enumerate(ld_tr):
            if a.max_steps and step >= a.max_steps:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            total, parts, _ = forward_losses(model, batch, weights, cls_w,
                                              use_phasor=not a.no_phasor)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); global_step += 1
            for k, v in parts.items():
                agg[k] += v * len(batch["I"])
            n += len(batch["I"])
            if (step + 1) % 50 == 0:
                print(f"  s{stage} ep{ep+1} step{step+1} loss={float(total):.4f} "
                      f"phase={parts['phase']:.4f} fc={parts['fc']:.4f} "
                      f"cls={parts['cls']:.4f} lr={opt.param_groups[0]['lr']:.2e}", flush=True)
        sched.step()
        rec = {k: v / max(n, 1) for k, v in agg.items()}
        rec.update(epoch=ep + 1, stage=stage, step=global_step,
                   sec=time.time() - t0, lr=opt.param_groups[0]["lr"])
        if (not a.skip_val) and ((ep + 1) % a.val_every == 0 or ep + 1 == total_epochs):
            vl = validate(model, ld_va, device, use_phasor=not a.no_phasor)
            rec["val_phase"] = vl
            log["val"].append(dict(epoch=ep + 1, stage=stage, val_phase=vl))
            if vl < best:
                best = vl
                torch.save(dict(model=model.state_dict(), args=vars(a), epoch=ep + 1,
                                stage=stage, val_phase=vl, params_m=nparam),
                           os.path.join(out, "best.pth"))
            if stage == 3 and vl < best3:
                best3 = vl
                torch.save(dict(model=model.state_dict(), args=vars(a), epoch=ep + 1,
                                stage=3, val_phase=vl, params_m=nparam),
                           os.path.join(out, "best_stage3.pth"))
        log["stages"].append(rec)
        print(f"stage{stage} ep{ep+1}/{total_epochs} {rec['sec']:.0f}s phase={rec['phase']:.4f} "
              f"fc={rec['fc']:.4f} cls={rec['cls']:.4f} val={rec.get('val_phase', '')}", flush=True)
        if ep + 1 in (e1, e1 + e2):
            torch.save(dict(model=model.state_dict(), args=vars(a), stage=stage, params_m=nparam),
                       os.path.join(out, f"stage{stage}.pth"))
        json.dump(log, open(os.path.join(out, "train_log.json"), "w"), indent=1)
    torch.save(dict(model=model.state_dict(), args=vars(a), epoch=total_epochs, stage=3,
                    params_m=nparam), os.path.join(out, "last.pth"))
    torch.save(dict(model=model.state_dict(), args=vars(a), epoch=total_epochs, stage=3,
                    params_m=nparam), os.path.join(out, "stage3.pth"))
    json.dump(log, open(os.path.join(out, "train_log.json"), "w"), indent=1)
    print(f"DONE best_val={best:.5f} out={out}", flush=True)


if __name__ == "__main__":
    main()








