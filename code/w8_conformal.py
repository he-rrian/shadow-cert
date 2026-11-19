"""w8_conformal.py -- W8 two-level (scene-level) conformal calibration.

Calculates scene scores
    S_i = Quantile_{1-beta}( |psi-phi| / sigma  over GT-dead pixels )
and Mondrian class thresholds
    tau_c = ceil((n_c+1)(1-alpha))-th smallest S_i in class c
then evaluates on the test split:
    phi_hat(x) +/- tau_c * sigma(x)
    success_i = fraction of GT-dead pixels covered >= 1-beta
    P_scene(success) >= 1-alpha

Calibration = val (MATLAB) or sf_val (SF); test = test / sf_test.
This script is calibration/evaluation only -- no retraining.
"""
import os, sys, json, argparse, math
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from w5_dataset import W5Dataset, BRANCH_ID
from w5_model import W5UNet, split_output, make_input

NAMES = {0: "P", 1: "U", 2: "R"}


def collate(batch):
    keys = ("I", "h", "M", "roi", "dead", "phi", "orr", "clean", "carrier", "y", "rho",
            "sid", "idx", "branch_id", "phi_obs", "dead_pred")
    return {k: torch.from_numpy(np.stack([b[k] for b in batch])) for k in keys}


@torch.no_grad()
def collect(ckpt, split, device="cuda", gamma=0.75):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = W5UNet(base=128).to(device); m.load_state_dict(ck["model"]); m.eval()
    ds = W5Dataset(split, gamma=gamma, augment=False)
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate,
                    pin_memory=True, persistent_workers=True)
    scenes = []
    for batch in ld:
        b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        dpsi, log_sigma, _, _, _ = split_output(m(make_input(b)))
        psi = (b["h"].float().unsqueeze(1) + b["M"].float().unsqueeze(1) * dpsi).squeeze(1)
        sig = torch.exp(0.5 * log_sigma).squeeze(1)
        e = (psi - b["phi"].float()).abs()
        dead = b["dead"].bool()
        y = b["y"].cpu().numpy(); sid = b["sid"].cpu().numpy()
        br = b["branch_id"].cpu().numpy()
        for j in range(len(psi)):
            d = dead[j]
            if not d.any():
                continue
            ee = e[j][d].cpu().numpy().astype(np.float64)
            ss = sig[j][d].cpu().numpy().astype(np.float64)
            ss = np.maximum(ss, 1e-6)
            scenes.append(dict(sid=int(sid[j]), y=int(y[j]), branch=int(br[j]),
                               score=(ee / ss), e=ee, s=ss))
    return scenes


def q_high(a, q):
    """empirical quantile with the 'higher' convention (safe for conformal)."""
    return float(np.quantile(np.asarray(a), q, method="higher"))


def tau_for(scores, alpha):
    n = len(scores)
    if n == 0:
        return float("nan")
    k = int(math.ceil((n + 1) * (1 - alpha))) - 1
    k = min(max(k, 0), n - 1)
    return float(np.sort(scores)[k])


def eval_cal(cal, test, alpha, beta):
    out = dict(alpha=alpha, beta=beta, classes={}, global_=None)
    # class-conditional (Mondrian) tau
    for c in (0, 1, 2):
        sc = [q_high([s["score"][np.isfinite(s["score"])]], 1 - beta) if False else
              q_high(s["score"], 1 - beta) for s in cal if s["y"] == c]
        # the expression above is intentionally simple: one scalar per scene
        tau = tau_for(sc, alpha)
        out["classes"][NAMES[c]] = dict(n_cal=len(sc), tau=tau,
                                        n_test=sum(1 for s in test if s["y"] == c))
    # global fallback / baseline
    allsc = [q_high(s["score"], 1 - beta) for s in cal]
    out["global_"] = dict(n_cal=len(allsc), tau=tau_for(allsc, alpha))
    # test coverage per class and global
    for c in (0, 1, 2):
        rr = [s for s in test if s["y"] == c]
        t = out["classes"][NAMES[c]]["tau"]
        cov = [float(np.mean(s["e"] <= t * s["s"])) for s in rr] if rr else []
        out["classes"][NAMES[c]].update(
            scene_success=float(np.mean([v >= 1 - beta for v in cov])) if cov else None,
            mean_coverage=float(np.mean(cov)) if cov else None,
            median_width=float(np.median([2 * t * np.median(s["s"]) for s in rr])) if rr and np.isfinite(t) else None)
    t = out["global_"]["tau"]
    cov = [float(np.mean(s["e"] <= t * s["s"])) for s in test]
    out["global_"].update(scene_success=float(np.mean([v >= 1 - beta for v in cov])),
                          mean_coverage=float(np.mean(cov)),
                          median_width=float(np.median([2 * t * np.median(s["s"]) for s in test])))
    return out


def pixel_baseline(cal, test, alpha, beta):
    """naive pixel-level split conformal: pool all dead pixels, ignore scene grouping."""
    sc = np.concatenate([s["score"] for s in cal])
    tau = tau_for(sc, alpha)
    cov = [float(np.mean(s["e"] <= tau * s["s"])) for s in test]
    return dict(tau=tau, n_cal=int(len(sc)),
                scene_success=float(np.mean([v >= 1 - beta for v in cov])),
                mean_coverage=float(np.mean(cov)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cal", default="val")
    ap.add_argument("--test", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", default="0.05,0.10,0.20")
    ap.add_argument("--betas", default="0.05,0.10,0.20")
    ap.add_argument("--gamma", type=float, default=0.75)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"collect cal={a.cal} ...", flush=True); cal = collect(a.ckpt, a.cal, device, a.gamma)
    print(f"collect test={a.test} ...", flush=True); test = collect(a.ckpt, a.test, device, a.gamma)
    res = dict(ckpt=a.ckpt, cal=a.cal, test=a.test, n_cal=len(cal), n_test=len(test), grid=[])
    for al in [float(x) for x in a.alphas.split(",")]:
        for be in [float(x) for x in a.betas.split(",")]:
            r = eval_cal(cal, test, al, be); r["coverage_target"] = 1 - be
            r["pixel_baseline"] = pixel_baseline(cal, test, al, be)
            res["grid"].append(r)
            print(f"alpha={al:.2f} beta={be:.2f} "
                  f"global_success={r['global_']['scene_success']:.3f} "
                  f"P_success={r['classes']['P']['scene_success']} "
                  f"U_success={r['classes']['U']['scene_success']} "
                  f"R_success={r['classes']['R']['scene_success']} "
                  f"pixel_success={r['pixel_baseline']['scene_success']:.3f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()


