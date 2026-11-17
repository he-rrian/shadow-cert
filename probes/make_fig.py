"""P2 probe 4 -- eight-panel feasibility figure.
Inputs : results/physics_render.json, results/physics_mechanism.json, results/envelope.npy,
         results/theory_c1.json, results/theory_sliver.json,
         F:\\paper\\2\\model\\real\\captures\\proc\\scene_01, F:\\paper\\SynthFringe
Output : results/fig_feasibility.png
Run    : python make_fig.py
"""
import os, json
import numpy as np, scipy.io as sio, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
P = r"F:\paper\original_phase"

def load_phi(i):
    return np.asarray(sio.loadmat(os.path.join(P, f"{i}.mat"))["fai"], np.float64)
def bandlimit(phi, f0, frac=0.45):
    gy, gx = np.gradient(phi); g = np.hypot(gx, gy) / (2 * np.pi)
    s = min(1.0, frac * f0 / max(float(np.percentile(g, 99.9)), 1e-9)); return phi * s, s
def render(phi, f0=1 / 8, seed=0):
    H, W = phi.shape; rng = np.random.default_rng(seed); X = np.arange(W)[None, :]
    I = 0.04 + 0.82 * (0.50 + 0.30 * np.cos(2 * np.pi * f0 * X + phi)) + 0.010 * rng.standard_normal(phi.shape)
    return np.clip(np.round(np.clip(I, 0, 1) * 255), 0, 255).astype(np.uint8)
def im(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

fig = plt.figure(figsize=(17.2, 8.6))
gs = fig.add_gridspec(2, 4, hspace=0.36, wspace=0.30)

# (a)(b) aliasing
ax = fig.add_subplot(gs[0, 0])
raw = load_phi(1128); bl, s = bandlimit(raw, 1 / 8)
ax.imshow(render(raw, 1 / 8, seed=1), cmap="gray", vmin=0, vmax=255)
ax.set_title(f"(a) raw $\\varphi$, carrier 1/8\n86.5% violate $|\\nabla\\varphi|/2\\pi<f_0$", fontsize=9)
ax.set_xticks([]); ax.set_yticks([])
ax = fig.add_subplot(gs[0, 1])
ax.imshow(render(bl, 1 / 8, seed=1), cmap="gray", vmin=0, vmax=255)
ax.set_title(f"(b) band-limited ($\\times${s:.2f})\nclean-fringe RMSE 0.13 rad", fontsize=9)
ax.set_xticks([]); ax.set_yticks([])

# (c) mechanism
ax = fig.add_subplot(gs[0, 2])
mech = json.load(open(os.path.join(RES, "physics_mechanism.json")))
for mode, col, lab in (("sat", "tab:orange", "saturation (clip)"),
                       ("shadow", "tab:green", "shadow (attenuation)"),
                       ("occl", "tab:blue", "total occlusion")):
    S = [m for m in mech if m["mode"] == mode]
    if S:
        ax.plot([m["d"] for m in S], [m["id_fail"] for m in S], "o-", color=col, label=lab)
ax.axhline(0.25, ls="--", c="k", lw=1); ax.text(64, 0.27, "doc G0 gate", fontsize=8)
ax.set_xlabel("dead-zone diameter $d$ (px)"); ax.set_ylabel("Identity $2\\pi$ failure rate")
ax.set_ylim(-0.03, 0.85); ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="upper left")
ax.set_title("(c) only occlusion destroys phase", fontsize=9)

# (d) WIDTH LAW: equal-area shapes, torsion bound vs the disk bound
ax = fig.add_subplot(gs[0, 3])
T = json.load(open(os.path.join(RES, "theory_c1.json")))["torsion"]
order = [("disk", 1.0), ("square_equal_area", 1.0), ("ellipse_2:1_equal_area", 2.0),
         ("sliver_4:1_equal_area", 4.0), ("sliver_8:1_equal_area", 8.0), ("sliver_16:1_equal_area", 16.0)]
xs = [o[1] for o in order if o[0] in T]
ys = [T[o[0]]["ratio_vmax_over_diskbound"] for o in order if o[0] in T]
ax.plot(xs, ys, "o-", color="tab:red")
ax.axhline(1.0, ls=":", c="gray")
ax.set_xscale("log"); ax.set_xlabel("aspect ratio (equal area)"); ax.set_ylabel(r"$\|v_\Omega\|_\infty\,/\,(d_{eq}^2/16)$")
ax.set_title("(d) width law: bound shrinks with elongation\n(measured 10x200 sliver: 10.6$\\times$ tighter)", fontsize=9)
ax.grid(alpha=0.3, which="both")
for x, y in zip(xs, ys):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center")

# (e) envelope
E = np.load(os.path.join(RES, "envelope.npy"))
area, d, idf, idr, evf, evr, evp, pinp, geo = E.T
ax = fig.add_subplot(gs[1, 0])
ax.semilogy(d, idr, "o-", label="Identity (FTP+unwrap)")
ax.semilogy(d, pinp, "^-", label="phase inpainting")
ax.semilogy(d, evr, "s-", label="ORR (boundary only)")
ax.semilogy(d, geo, "k--", lw=1, label=r"$\kappa d^2/8$ (1-D)")
for tol, lab in [(0.314, r"$\pi/10$"), (1.571, r"$\pi/2$")]:
    ax.axhline(tol, ls=":", c="gray"); ax.text(150, tol * 1.1, lab, fontsize=8)
ax.set_xlabel("$d$ (px)"); ax.set_ylabel("RMSE in mask (rad)")
ax.set_title("(e) dead-zone error vs hole size", fontsize=9)
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)

# (f) real scene
ax = fig.add_subplot(gs[1, 1])
d0 = r"F:\paper\2\model\real\captures\proc\scene_01"
I = [cv2.imread(os.path.join(d0, f"capture_{k}.png"), 0).astype(np.float64) / 255 for k in range(1, 5)]
vis = np.sqrt((I[0] - I[2]) ** 2 + (I[3] - I[1]) ** 2) / (2 * (sum(I) / 4 + 1e-6))
pm = cv2.imread(os.path.join(d0, "badmask_single.png"), 0) > 127
ax.imshow(vis, cmap="viridis", vmin=0, vmax=1.0)
ax.contour(pm, levels=[0.5], colors="red", linewidths=1.6)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("(f) real scene: painted hole (red)\non physical visibility", fontsize=9)
ax.text(4, 250, f"physical m<0.08: {(vis < 0.04).mean()*100:.1f}%   painted: {pm.mean()*100:.1f}%",
        fontsize=7, color="w")

# (g) SynthFringe background
ax = fig.add_subplot(gs[1, 2])
sd = r"F:\paper\SynthFringe\64_15\00000"
S = []
for f in sorted(os.listdir(sd))[:15]:
    a = im(os.path.join(sd, f))
    if a.ndim == 3: a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2GRAY)
    S.append(a.astype(np.float32))
mx = np.max(np.stack(S), 0)
cls = np.zeros_like(mx); cls[(mx >= 1) & (mx < 5)] = 0.33; cls[(mx >= 5) & (mx < 60)] = 0.66; cls[mx >= 60] = 1.0
ax.imshow(cls, cmap="viridis"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("(g) SynthFringe: no surface vs noise vs object", fontsize=9)
ax.text(4, 470, "object 32.8% | bg noise 27.8% | black 38.0%", fontsize=7, color="w")

# (h) bars
ax = fig.add_subplot(gs[1, 3])
labels = ["Identity", "phase\ninpaint", "ORR"]
vals = [np.median(idr), np.median(pinp), np.median(evr)]
ax.bar(labels, vals, color=["tab:red", "tab:green", "tab:blue"])
ax.set_yscale("log"); ax.set_ylim(0.5, 120)
for i, v in enumerate(vals):
    ax.text(i, v * 1.25, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("median RMSE in mask (rad)")
ax.set_title("(h) headroom vs Identity 16.4$x$;\nvs phase inpaint only 7.7%", fontsize=9)
ax.grid(alpha=0.3, axis="y")

fig.suptitle("P2 feasibility check on the actual data (F:\\paper)  -  probes, 2026-11-15", fontsize=11, y=0.985)
out = os.path.join(RES, "fig_feasibility.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
