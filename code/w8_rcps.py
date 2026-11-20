"""w8_rcps.py -- W8 RCPS risk-controlled acceptance.

Per-scene normalized counts on the calibration set:
    a_i(lambda) = # {sigma <= lambda, dead} / #dead_i      in [0,1]
    e_i(lambda) = # {sigma <= lambda, |psi-phi| > phi_tol, dead} / #dead_i
    0 <= e_i <= a_i <= 1

Risk is the accepted-pixel error rate:
    R(lambda) = sum_i e_i / sum_i a_i.
Condition R <= alpha is equivalent to mean_i(e_i - alpha*a_i) <= 0.
Because Y_i = e_i - alpha*a_i in [-alpha, 1], the Hoeffding upper bound with n=scenes is strict:
    U(lambda) = mean(Y) + (1+alpha)*sqrt(log(1/delta)/(2*n_scenes)).
A running minimum over the lambda grid restores monotonicity.  The naive pixel
bound treats every dead pixel as independent (n_pixels) and is reported only as
an anti-baseline.
"""
import os, sys, json, argparse, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PHI_TOL = math.pi / 10.0
DELTA = 0.05


def curves(cal, test, alpha, n_grid=200):
    sig = np.concatenate([s["s"] for s in cal])
    lam = np.quantile(sig, np.linspace(0, 1, n_grid))
    lam = np.unique(lam)
    n = len(cal)
    rows = []
    for L in lam:
        a = np.array([np.mean(s["s"] <= L) for s in cal], float)
        e = np.array([np.mean((s["s"] <= L) & (s["e"] > PHI_TOL)) for s in cal], float)
        y = e - alpha * a
        u = float(y.mean() + (1 + alpha) * math.sqrt(math.log(1 / DELTA) / (2 * n)))
        # WSR / empirical Bernstein on W = Y + alpha in [0, 1+alpha]
        W = y + alpha
        nW = len(W)
        varW = float(W.var(ddof=1)) if nW > 1 else 0.0
        L2 = math.log(2.0 / DELTA)
        eps_w = (math.sqrt(2.0 * varW * L2 / nW)
                 + 7.0 * (1.0 + alpha) * L2 / (3.0 * max(nW - 1, 1)))
        u_wsr = float(W.mean() + eps_w - alpha)
        A = float(a.sum()); E = float(e.sum())
        r_pix = E / max(A, 1e-9)
        scene = a > 1e-9
        r_scene = float(np.mean(e[scene] / a[scene])) if scene.any() else 0.0
        # naive pixel-level Hoeffding (wrong independence assumption)
        Apx = float(sum(np.sum(s["s"] <= L) for s in cal))
        Fpx = float(sum(np.sum((s["s"] <= L) & (s["e"] > PHI_TOL)) for s in cal))
        r_naive = Fpx / max(Apx, 1e-9)
        u_naive = r_naive + math.sqrt(math.log(1 / DELTA) / (2 * max(Apx, 1.0)))
        rows.append(dict(lam=float(L), acc_frac=float(a.mean()), r_pix=r_pix,
                         r_scene=r_scene, u=u, u_wsr=u_wsr,
                         r_naive=r_naive, u_naive=u_naive, A=A, E=E))
    U = np.array([r["u"] for r in rows])
    Urun = np.minimum.accumulate(U[::-1])[::-1]
    UW = np.array([r["u_wsr"] for r in rows])
    UWrun = np.minimum.accumulate(UW[::-1])[::-1]
    for i, r in enumerate(rows):
        r["u_run"] = float(Urun[i])
        r["u_wsr_run"] = float(UWrun[i])
        r["r_upper_wsr"] = float(UWrun[i] + alpha)
        r["r_upper_hoef"] = float(Urun[i] + alpha)
    valid = [i for i, r in enumerate(rows) if r["u_wsr_run"] <= 0.0]
    if valid:
        i = max(valid)
        chosen = rows[i]
    else:
        chosen = dict(lam=0.0, acc_frac=0.0, r_pix=0.0, r_scene=0.0, u=0.0, u_run=0.0,
                      u_wsr=0.0, u_wsr_run=0.0, r_upper_wsr=alpha, r_upper_hoef=alpha,
                      r_naive=0.0, u_naive=0.0, A=0.0, E=0.0)
    # test at chosen lambda
    tA = tE = tN = 0.0; t_px = 0.0; t_scene = []
    for s in test:
        aa = float(np.mean(s["s"] <= chosen["lam"]))
        ee = float(np.mean((s["s"] <= chosen["lam"]) & (s["e"] > PHI_TOL)))
        tA += aa; tE += ee; tN += 1.0; t_px += float(np.sum(s["s"] <= chosen["lam"]))
        if aa > 1e-9:
            t_scene.append(ee / aa)
    chosen_test = dict(lam=chosen["lam"], acc_frac=float(tA / max(tN, 1)),
                       r_pix=float(tE / max(tA, 1e-9)),
                       r_scene=float(np.mean(t_scene)) if t_scene else 0.0,
                       n_accepted_px=int(t_px))
    return dict(alpha=alpha, chosen=chosen, chosen_test=chosen_test, grid=rows)


def main():
    import torch
    from w8_conformal import collect
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cal", default="val")
    ap.add_argument("--test", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", default="0.05,0.10,0.20")
    ap.add_argument("--gamma", type=float, default=0.75)
    ap.add_argument("--no-phasor", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("collect cal...", flush=True); cal = collect(a.ckpt, a.cal, device, a.gamma,
                                                       no_phasor=a.no_phasor)
    print("collect test...", flush=True); test = collect(a.ckpt, a.test, device, a.gamma,
                                                         no_phasor=a.no_phasor)
    out = dict(ckpt=a.ckpt, cal=a.cal, test=a.test, phi_tol=PHI_TOL, delta=DELTA,
               n_cal=len(cal), n_test=len(test), alphas=[])
    for al in [float(x) for x in a.alphas.split(",")]:
        r = curves(cal, test, al)
        out["alphas"].append(r)
        c = r["chosen"]; t = r["chosen_test"]
        print(f"alpha={al:.2f} lam={c['lam']:.5f} cal_acc={c['acc_frac']:.3f} "
              f"cal_Rpix={c['r_pix']:.4f} U_run={c['u_run']:.4f} "
              f"test_acc={t['acc_frac']:.3f} test_Rpix={t['r_pix']:.4f} "
              f"naive_R={c['r_naive']:.4f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()



