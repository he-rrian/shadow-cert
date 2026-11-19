"""w8_large_occlusion.py -- large-hole / failure analysis from compact W8 cache (no GPU).

For seed4 val:
  - reconstruct per-sample dead-pixel error maps from compact cache + source dead mask
  - report dead-area bins, P95, failure fraction
  - select largest holes and worst failures; plot input/GT/dead/h_input/Ours error.
"""
import os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"F:\paper\2\P2"
SPLIT = "val"
SEED = 4
PHI_TOL = math.pi / 10
NAMES = {0: "sliver", 1: "arc", 2: "blob", 3: "edge", 4: "cut", 5: "degraded"}

# --- source arrays, in manifest order ---
man = json.load(open(os.path.join(ROOT, "data", SPLIT, "manifest_000.json")))
I_l, phi_l, dead_l, roi_l, meta_l = [], [], [], [], []
for s in man["shards"]:
    z = np.load(os.path.join(ROOT, "data", SPLIT, s["path"]))
    I_l.append(z["I"]); phi_l.append(z["phi"].astype(np.float32))
    dead_l.append(z["dead"].astype(bool)); roi_l.append(z["roi"].astype(bool))
    meta_l.extend(z["meta"])
I = np.concatenate(I_l); phi = np.concatenate(phi_l)
dead = np.concatenate(dead_l); roi = np.concatenate(roi_l)

# --- compact error + sigma, in dataset order (idx maps to source concatenation) ---
c = np.load(os.path.join(ROOT, "data", "preds_compact", f"seed{SEED}_{SPLIT}.npz"))
e_all = c["e"].astype(np.float64); s_all = c["sigma"].astype(np.float64)
n = c["n_dead"].astype(int); off = np.r_[0, np.cumsum(n)]
idx = c["idx"].astype(int); branch = c["branch_id"].astype(int)
assert off[-1] == len(e_all)

# --- frontend h, in source order (for h_input comparison) ---
fman = json.load(open(os.path.join(ROOT, "data", "frontend", SPLIT, "frontend_manifest.json")))
h_l = []
for s in fman["shards"]:
    z = np.load(os.path.join(ROOT, "data", "frontend", SPLIT, s["path"]))
    h_l.append(z["h"].astype(np.float32))
h = np.concatenate(h_l)

rows = []
for k in range(len(idx)):
    i = int(idx[k])
    d = dead[i]
    if d.sum() != n[k]:
        continue
    e = e_all[off[k]:off[k + 1]]
    sig = s_all[off[k]:off[k + 1]]
    p95 = float(np.percentile(e, 95)); emax = float(e.max())
    fail = float(np.mean(e > PHI_TOL))
    he = np.abs(h[i][d] - phi[i][d])
    rows.append(dict(idx=i, branch=int(branch[k]), branch_name=NAMES.get(int(branch[k]), "?"),
                     dead=int(d.sum()), roi=int(roi[i].sum()),
                     dead_ratio=float(d.sum() / max(roi[i].sum(), 1)),
                     p95=p95, max=emax, fail_frac=fail,
                     h_p95=float(np.percentile(he, 95)), sigma_med=float(np.median(sig))))

# bins
bins = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.01)]
print("=== dead-area bins ===")
bin_stats = []
for a, b in bins:
    rr = [r for r in rows if a <= r["dead_ratio"] < b]
    if not rr:
        continue
    st = dict(lo=a, hi=b, n=len(rr), p95_mean=float(np.mean([r["p95"] for r in rr])),
              p95_med=float(np.median([r["p95"] for r in rr])),
              fail_mean=float(np.mean([r["fail_frac"] for r in rr])),
              fail_max=float(np.max([r["fail_frac"] for r in rr])))
    bin_stats.append(st)
    print(f"{a:.2f}-{b:.2f} ROI: n={st['n']:3d} P95 med={st['p95_med']:.3f} "
          f"fail mean={st['fail_mean']:.3f} max={st['fail_max']:.2f}")

json.dump(dict(rows=rows, bins=bin_stats), open(os.path.join(ROOT, "probes", "results", "w8_large_occlusion_stats.json"), "w"), indent=1)

# select: largest dead, largest arc/blob, worst failure with dead_ratio>=2%
rows_sorted = sorted(rows, key=lambda r: r["dead_ratio"], reverse=True)
sel = []
for r in rows_sorted:
    if len(sel) < 2:
        sel.append(r)
    elif r["branch_name"] == "arc" and all(x["branch_name"] != "arc" for x in sel):
        sel.append(r)
    elif len(sel) < 3:
        sel.append(r)
# worst failure among holes >=2%
worst = max([r for r in rows if r["dead_ratio"] >= 0.02], key=lambda r: r["fail_frac"])
if worst not in sel:
    sel.append(worst)
sel = sel[:4]

fig, axes = plt.subplots(len(sel), 5, figsize=(13, 2.6 * len(sel)))
cmap = plt.get_cmap("twilight").copy(); cmap.set_bad("white")
for k, r in enumerate(sel):
    i = r["idx"]; d = dead[i]
    # locate e slice by idx
    kk = [j for j in range(len(idx)) if int(idx[j]) == i][0]
    ee = e_all[off[kk]:off[kk + 1]]
    emap = np.full((256, 256), np.nan); emap[d] = ee
    hemap = np.full((256, 256), np.nan); hemap[d] = np.abs(h[i][d] - phi[i][d])
    vmin, vmax = np.percentile(phi[i][roi[i]], [1, 99])
    axes[k, 0].imshow(I[i], cmap="gray"); axes[k, 0].set_title("input I", fontsize=8)
    axes[k, 1].imshow(np.ma.masked_where(~roi[i], phi[i]), cmap=cmap, vmin=vmin, vmax=vmax)
    axes[k, 1].set_title("GT phase", fontsize=8)
    axes[k, 2].imshow(I[i], cmap="gray"); axes[k, 2].imshow(np.ma.masked_where(~d, d.astype(float)), cmap="Reds", alpha=.55)
    axes[k, 2].set_title(f"dead {r['dead']} px = {100*r['dead_ratio']:.1f}% ROI", fontsize=8)
    axes[k, 3].imshow(hemap, cmap="hot", vmin=0, vmax=0.8)
    axes[k, 3].set_title(f"h_input error P95={r['h_p95']:.2f}", fontsize=8)
    axes[k, 4].imshow(emap, cmap="hot", vmin=0, vmax=0.8)
    axes[k, 4].set_title(f"Ours error P95={r['p95']:.2f}", fontsize=8)
    for ax in axes[k]: ax.axis("off")
    axes[k, 0].text(3, 246, f"{r['branch_name']} | fail {100*r['fail_frac']:.1f}%", color="w", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "probes", "results", "w8_large_occlusion.png"), dpi=180); plt.close(fig)
print("selected:", [(r["branch_name"], round(r["dead_ratio"], 3), round(r["p95"], 3), round(r["fail_frac"], 3)) for r in sel])

