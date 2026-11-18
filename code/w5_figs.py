"""w5_figs.py -- S5 G5 figures (English labels)."""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VINF = ("sliver", "arc", "blob")


def load_rows(d, name):
    p = os.path.join(d, f"rows_{name}.json")
    return json.load(open(p)) if os.path.exists(p) else []


def fig_bars(rows, out):
    methods = [("har", "Harmonic"), ("bih", "Biharm"), ("orr", "ORR"), ("p95", "Ours")]
    groups = list(VINF) + ["all"]
    fig, ax = plt.subplots(figsize=(8, 4.2)); w = 0.8 / len(methods)
    for k, (key, lab) in enumerate(methods):
        vals = []
        for g in groups:
            rr = rows if g == "all" else [r for r in rows if r["branch"] == g]
            vals.append(np.median([r[key] for r in rr]) if rr else np.nan)
        ax.bar(np.arange(len(groups)) + (k - 1.5) * w, vals, w, label=lab)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups)
    ax.set_ylabel("median P95 error (rad)"); ax.set_title("G5-a: val phase error by shape")
    ax.legend(ncol=4, fontsize=8); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def fig_cdf(rows, out):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for key, lab, c in [("p95", "Ours", "C3"), ("bih", "Biharm", "C0"),
                        ("orr", "ORR", "C2"), ("har", "Harmonic", "C1")]:
        v = np.sort([r[key] for r in rows])
        ax.plot(v, np.linspace(0, 1, len(v)), label=lab, color=c)
    ax.set_xlabel("per-sample P95 error (rad)"); ax.set_ylabel("CDF")
    ax.set_title("G5-a: error CDF (val, vinf samples)")
    ax.set_xlim(0, min(3.0, np.percentile([r["p95"] for r in rows], 99) * 1.2))
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def fig_qual(d, out):
    p = os.path.join(d, "qual", "qual_val.npz")
    if not os.path.exists(p):
        return
    z = np.load(p); n = min(6, len(z["I"]))
    cols = ["input I", "GT phi", "h_input", "Ours psi", "|Ours-GT| (dead)", "M_fix"]
    fig, axes = plt.subplots(n, len(cols), figsize=(2.2 * len(cols), 2.0 * n))
    for i in range(n):
        I, phi, h, psi, M = z["I"][i], z["phi"][i], z["h"][i], z["psi"][i], z["M"][i]
        dead = z["dead"][i].astype(bool)
        err = np.abs(psi - phi); err[~dead] = np.nan
        for j, (im, cm) in enumerate(zip([I, phi, h, psi, err, M],
                                         ["gray", "twilight", "twilight", "twilight", "hot", "viridis"])):
            ax = axes[i, j]; ax.imshow(im, cmap=cm); ax.axis("off")
            if i == 0:
                ax.set_title(cols[j], fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=170); plt.close(fig)


def fig_residual(rows, out):
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.8))
    a = np.array([r["l2_ours"] for r in rows]); b = np.array([r.get("l2_h", r["l2_har"]) for r in rows])
    ax[0].scatter(b, a, s=6, alpha=.5)
    lim = max(np.percentile(b, 99), np.percentile(a, 99))
    ax[0].plot([0, lim], [0, lim], "k--", lw=.8)
    ax[0].set_xlabel("||h_input-ORR||"); ax[0].set_ylabel("||Ours-ORR||")
    ax[0].set_title(f"G5-c: frac(Ours<h_input)={np.mean(a < b):.2f}"); ax[0].grid(alpha=.3)
    ax[1].hist(a / np.maximum(b, 1e-9), bins=40, color="C3", alpha=.8)
    ax[1].axvline(1, color="k", ls="--", lw=.8)
    ax[1].set_xlabel("||Ours-ORR|| / ||h_input-ORR||"); ax[1].set_ylabel("samples")
    ax[1].set_title("G5-c ratio (<1 better)")
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def fig_train(log, out):
    if not log or not os.path.exists(log):
        return
    d = json.load(open(log)); st = d.get("stages", [])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot([r["epoch"] for r in st], [r["phase"] for r in st], label="train phase", color="C0")
    if d.get("val"):
        ax.plot([r["epoch"] for r in d["val"]], [r["val_phase"] for r in d["val"]],
                "o-", label="val phase", color="C1", ms=3)
    for x in (30, 60):
        ax.axvline(x, color="gray", ls=":", lw=.8)
    ax.set_xlabel("epoch"); ax.set_ylabel("region-balanced SmoothL1")
    ax.set_title("S5 training (seed 0, 100 epochs)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def fig_sf(rows_val, rows_sf, out):
    if not rows_sf:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.8)); groups = list(VINF)
    x = np.arange(len(groups)); w = .35
    v = [np.median([r["p95"] for r in rows_val if r["branch"] == g]) for g in groups]
    s = [np.median([r["p95"] for r in rows_sf if r["branch"] == g]) for g in groups]
    ax.bar(x - w / 2, v, w, label="MATLAB val"); ax.bar(x + w / 2, s, w, label="SF test (zero-shot)")
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("median P95 error (rad)")
    ax.set_title("G5-b: cross-domain error"); ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def fig_fair(rows, out):
    methods = [("har_sf", "Harmonic_SF"), ("bih_sf", "Biharm_SF"),
               ("p95_h", "h_input"), ("p95", "Ours")]
    groups = list(VINF) + ["all"]
    fig, ax = plt.subplots(figsize=(8, 4.2)); w = 0.8 / len(methods)
    for k, (key, lab) in enumerate(methods):
        vals = []
        for g in groups:
            rr = rows if g == "all" else [r for r in rows if r["branch"] == g]
            vals.append(np.median([r[key] for r in rr]) if rr else np.nan)
        ax.bar(np.arange(len(groups)) + (k - 1.5) * w, vals, w, label=lab)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups)
    ax.set_ylabel("median P95 error (rad)")
    ax.set_title("G5-a fair tier: single-frame input + learned dead mask")
    ax.legend(ncol=4, fontsize=8); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="probes/results/w5_g5")
    ap.add_argument("--out", default="probes/results/w5_figs")
    ap.add_argument("--train-log", default="runs/w5_bal_seed0/train_log.json")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rv = load_rows(a.dir, "val"); rs = load_rows(a.dir, "sf_test")
    fig_bars(rv, os.path.join(a.out, "g5a_bars.png"))
    fig_fair(rv, os.path.join(a.out, "g5a_fair_bars.png"))
    fig_cdf(rv, os.path.join(a.out, "g5a_cdf.png"))
    fig_qual(a.dir, os.path.join(a.out, "g5_qual.png"))
    fig_residual(rv, os.path.join(a.out, "g5c_residual.png"))
    fig_train(a.train_log, os.path.join(a.out, "train_curve.png"))
    fig_sf(rv, rs, os.path.join(a.out, "g5b_sf.png"))
    print("figures ->", a.out)


if __name__ == "__main__":
    main()


