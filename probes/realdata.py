"""P2 probe 3 -- the real / public assets.

A. six real scenes (2\\model\\real\\captures\\proc\\scene_01..06): GT sanity, painted mask vs physical visibility
B. SynthFringe: decode N-step PS -> phase GT, fringe period, valid domain, dead-zone census
C. folder-1 real scenes (1\\phase\\实测1/实测2/实验3/实验4/Train): sizes and GT availability
Run:  python realdata.py
"""
import os, json
import numpy as np, cv2, scipy.io as sio
from skimage.restoration import unwrap_phase

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
REAL = r"F:\paper\2\model\real\captures\proc"
SYN = r"F:\paper\SynthFringe"

def im(p):
    b = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(b, cv2.IMREAD_UNCHANGED)

def gray(p):
    a = im(p)
    if a is None: return None
    if a.ndim == 3: a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2GRAY)
    return a.astype(np.float32)

# ---------------------------------------------------------------- A
print("=" * 78, "\nA. six real scenes")
rows = []
for s in sorted(os.listdir(REAL)):
    d = os.path.join(REAL, s)
    g = np.load(os.path.join(d, "gt_phase.npy"))
    fin = np.isfinite(g)
    I = [gray(os.path.join(d, f"capture_{k}.png")) / 255.0 for k in range(1, 5)]
    vis = np.sqrt((I[0] - I[2]) ** 2 + (I[3] - I[1]) ** 2) / (2 * (sum(I) / 4 + 1e-6))
    pm = cv2.imread(os.path.join(d, "badmask_single.png"), 0) > 127
    ss = gray(os.path.join(d, "single_shot.png"))
    gy, gx = np.gradient(np.where(fin, g, 0.0))
    mag = np.hypot(gy, gx)
    inner = fin.copy(); inner[0] = inner[-1] = False; inner[:, 0] = inner[:, -1] = False
    r = dict(scene=s, gt_min=float(np.nanmin(g)), gt_max=float(np.nanmax(g)),
             nan=float(1 - fin.mean()), tear=float((mag[inner] > np.pi).mean()),
             painted=float(pm.mean()), phys_low=float((vis < 0.04).mean()),
             sat=float((np.maximum.reduce([gray(os.path.join(d, f"capture_{k}.png")) for k in range(1, 5)]) >= 250).mean()),
             black_in_mask=float((ss[pm] == 0).mean()) if pm.any() else 0.0)
    rows.append(r)
    print(f"  {s}: GT[{r['gt_min']:7.1f},{r['gt_max']:7.1f}] nan={r['nan']:.3f} tear={r['tear']*100:.2f}% "
          f"| painted={r['painted']*100:5.1f}% (black in mask {r['black_in_mask']*100:.0f}%) "
          f"| physical low={r['phys_low']*100:4.1f}%  sat={r['sat']*100:4.2f}%")
json.dump(rows, open(os.path.join(RES, "real6.json"), "w"), indent=1)

# ---------------------------------------------------------------- B
print("=" * 78, "\nB. SynthFringe decode")
def seq(sub, sid):
    p = os.path.join(SYN, sub, sid)
    fs = sorted(f for f in os.listdir(p) if f.lower().endswith((".bmp", ".png", ".jpg")))
    out = []
    for f in fs:
        g = gray(os.path.join(p, f))
        if g is not None: out.append(g)
    return np.stack(out)

def nstep(I):
    N = I.shape[0]; k = np.arange(N); d = 2 * np.pi * k / N
    num = sum(I[i] * (-np.sin(d[i])) for i in range(N))
    den = sum(I[i] * np.cos(d[i]) for i in range(N))
    return np.arctan2(num, den), np.sqrt(num ** 2 + den ** 2) / N

syn = []
for sub in ("1_12", "3_16", "8_6", "64_15"):
    I = seq(sub, "00000"); psi, mod = nstep(I)
    mx = I.max(0)
    roi = cv2.morphologyEx((mx >= 25).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    un = unwrap_phase(psi)
    gy, gx = np.gradient(un); g = np.hypot(gx, gy)[roi] / (2 * np.pi)
    r = dict(sub=sub, steps=I.shape[0], roi=float(roi.mean()), period=float(1 / max(np.median(g), 1e-9)),
             phase_span=float(np.percentile(un[roi], 99) - np.percentile(un[roi], 1)),
             low_vis=float((mod[roi] < 0.15).mean()), sat=float((I.max(0)[roi] >= 250).mean()),
             bg_exact0=float((mx == 0).mean()), bg_noise=float(((mx >= 1) & (mx < 5)).mean()),
             obj=float((mx >= 60).mean()))
    syn.append(r)
    print(f"  {sub}: steps={r['steps']:2d} period={r['period']:6.1f}px phase_span={r['phase_span']:6.1f}rad "
          f"roi={r['roi']*100:5.1f}% low_vis={r['low_vis']*100:4.1f}% sat={r['sat']*100:4.2f}% "
          f"| bg0={r['bg_exact0']*100:4.1f}% bg_noise={r['bg_noise']*100:4.1f}% obj={r['obj']*100:4.1f}%")
json.dump(syn, open(os.path.join(RES, "synthfringe.json"), "w"), indent=1)
cal = {}
for f in sorted(os.listdir(os.path.join(SYN, "calibData"))):
    a = np.loadtxt(os.path.join(SYN, "calibData", f))
    cal[f] = np.asarray(a).tolist()
print(f"  calibration: Cam fx={cal['CamIntrinsicMatrix.txt'][0][0]:.2f} "
      f"Proj fx={cal['ProjIntrinsicMatrix.txt'][0][0]:.2f} "
      f"t={np.round(cal['ProjTranslationVector.txt'], 2).ravel()}")
json.dump(cal, open(os.path.join(RES, "calib.json"), "w"), indent=1)

# ---------------------------------------------------------------- C
print("=" * 78, "\nC. folder-1 real scenes")
F1 = r"F:\paper\1\phase"
for d in ("实测1", "实测2", "实验3", "实验4", "Train"):
    p = os.path.join(F1, d)
    if not os.path.isdir(p): continue
    fs = sorted(os.listdir(p))
    steps = [f for f in fs if f.lower().endswith((".bmp", ".jpg", ".png")) and ("shice" in f or f[:2].isdigit())]
    mats = [f for f in fs if f.endswith(".mat")]
    gt = [f for f in mats if "purphase" in f]
    print(f"  {d}: frames={len(steps)} mats={len(mats)} gt_mat={len(gt)} "
          f"first={steps[0] if steps else '-'}")
print("\nsaved to", RES)
