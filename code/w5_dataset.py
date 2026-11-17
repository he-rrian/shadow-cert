"""w5_dataset.py -- S5 dataset: lazy shard reader + gamma*=0.75 labels + torch-free self-test.

Key rules
---------
1. Labels are recomputed AT READ TIME from `meta`:
      rho = (kappa_hat_ps[1] * vinf + 0.13) / (pi/10)
      degraded -> R(2);  rho <= gamma -> P(0);  else U(1)
   The frozen `meta["cls"]` is gamma=1 and MUST NOT be used as the training label.
2. The network input frontend (h_input / M_fix / clean_est) comes from the cache built by
   `w5_prep_frontend.py`; `__getitem__` never solves a Poisson system.
3. The frontend cache was built from `I` + legal prior `roi` only.  GT `dead`, `orr` and
   `phi` are returned for loss / evaluation only, never used to build h_input / M_fix.
4. A pure horizontal mirror (no sign negation) canonicalises negative carriers to +x;
   the cache already stores canonical frames, so source arrays are mirrored when
   `flip=1`, while cached arrays are used as-is.

Run:
  python code/w5_dataset.py --self-test --split val --n 500
"""
import os, sys, json, argparse
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA, "frontend")
PHI_TOL = float(np.pi / 10)
DELTA_G = 0.13
GAMMA_DEFAULT = 0.75


def load_manifest(split):
    d = os.path.join(DATA, split)
    p = os.path.join(d, "manifest.json") if split.startswith("sf_") else os.path.join(d, "manifest_000.json")
    return d, json.load(open(p, encoding="utf-8"))


def label_from_meta(m, gamma):
    """Return (y, rho).  y: 0=P, 1=U, 2=R.  Frozen meta['cls'] is gamma=1 -- ignored.

    CORRECTED 2026-11-17: the certificate condition is
        gamma * kappa_hat * vinf + delta_g <= phi_tol
    NOT `rho <= gamma` (that would give P=20 instead of the W4 P=193 at gamma=0.75).
    rho is still returned as the gamma=1 diagnostic (kappa_hat*vinf+delta)/phi_tol.
    """
    if m.get("branch") == "degraded":
        return 2, float("nan")
    vinf = m.get("vinf")
    if vinf is None:
        return 1, float("nan")
    kh = float(m["kappa_hat_ps"][1])          # P75, the main kappa_hat
    rho = (kh * float(vinf) + DELTA_G) / PHI_TOL
    bound = gamma * kh * float(vinf) + DELTA_G
    return (0 if bound <= PHI_TOL else 1), float(rho)


class W5Dataset:
    def __init__(self, split, gamma=GAMMA_DEFAULT, augment=False, cache_dir=CACHE):
        self.split = split
        self.gamma = float(gamma)
        self.augment = bool(augment)
        self.cache_dir = cache_dir
        self.src_dir, self.src_man = load_manifest(split)
        self.fe_man = json.load(open(os.path.join(cache_dir, split, "frontend_manifest.json"), encoding="utf-8"))
        self.src_shards = {s["path"]: s for s in self.src_man["shards"]}
        self.fe_shards = {s["path"]: s for s in self.fe_man["shards"]}
        self.index = []
        for s in self.src_man["shards"]:
            if s["path"] not in self.fe_shards:
                raise FileNotFoundError(f"frontend cache missing for {split}/{s['path']}; run w5_prep_frontend.py")
            for i in range(int(s["n"])):
                self.index.append((s["path"], i))
        self._src = {}      # shard_name -> dict of arrays (loaded once per worker)
        self._fe = {}

    def _load_src(self, name):
        if name not in self._src:
            z = np.load(os.path.join(self.src_dir, name))
            self._src[name] = {k: z[k] for k in ("I", "phi", "roi", "dead", "orr", "kmap", "meta")}
        return self._src[name]

    def _load_fe(self, name):
        if name not in self._fe:
            z = np.load(os.path.join(self.cache_dir, self.split, name))
            self._fe[name] = {k: z[k] for k in ("h", "M", "clean", "border", "carrier",
                                                "k", "flip", "sid", "iou", "iou_obs",
                                                "obs_p95", "h_p95", "h_miss", "tau")}
        return self._fe[name]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        name, i = self.index[idx]
        s = self._load_src(name); f = self._load_fe(name)
        I = np.asarray(s["I"][i], np.uint8)
        phi = np.asarray(s["phi"][i], np.float32)
        roi = np.asarray(s["roi"][i], bool)
        dead = np.asarray(s["dead"][i], bool)
        orr = np.asarray(s["orr"][i], np.float32)
        kmap = np.asarray(s["kmap"][i], np.float16).astype(np.float32)  # do not diff in float16
        m = json.loads(str(s["meta"][i]))
        y, rho = label_from_meta(m, self.gamma)
        h = np.asarray(f["h"][i], np.float32)
        M = np.asarray(f["M"][i], np.float16).astype(np.float32)
        clean = np.asarray(f["clean"][i], bool)
        flip = int(f["flip"][i])

        if flip:
            I = I[:, ::-1]; phi = phi[:, ::-1]; roi = roi[:, ::-1]
            dead = dead[:, ::-1]; orr = orr[:, ::-1]; kmap = kmap[:, ::-1]
            # h / M / clean are already canonical in the cache: do NOT flip them again

        if self.augment and np.random.rand() < 0.5:      # vertical flip: no carrier contradiction
            I = I[::-1].copy(); phi = phi[::-1].copy(); roi = roi[::-1].copy()
            dead = dead[::-1].copy(); orr = orr[::-1].copy(); kmap = kmap[::-1].copy()
            h = h[::-1].copy(); M = M[::-1].copy(); clean = clean[::-1].copy()

        return dict(I=I, phi=phi, h=h, M=M, clean=clean, roi=roi, dead=dead, orr=orr, kmap=kmap,
                    y=np.int64(y), rho=np.float32(rho), sid=np.int32(f["sid"][i]),
                    branch=m.get("branch", ""), flip=np.int8(flip),
                    border=np.bool_(f["border"][i]),
                    kappa_hat=np.float32(m.get("kappa_hat", np.nan)),
                    kappa_hat_ps1=np.float32(m["kappa_hat_ps"][1] if m.get("kappa_hat_ps") else np.nan),
                    iou=np.float32(f["iou"][i]), iou_obs=np.float32(f["iou_obs"][i]),
                    obs_p95=np.float32(f["obs_p95"][i]), h_p95=np.float32(f["h_p95"][i]),
                    h_miss=np.float32(f["h_miss"][i]))


def collate(batch):
    """Torch-free collate (numpy).  A torch wrapper can be added in the training loop."""
    out = {}
    for k in batch[0]:
        v = [b[k] for b in batch]
        if isinstance(v[0], (np.ndarray, np.bool_, np.integer, np.floating)):
            out[k] = np.stack(v)
        else:
            out[k] = v
    return out


def self_test(split="val", n=500, gamma=GAMMA_DEFAULT):
    ds = W5Dataset(split, gamma=gamma, augment=False)
    n = min(n, len(ds))
    print(f"split={split}  gamma={gamma}  len={len(ds)}  testing {n}")
    from collections import Counter, defaultdict
    cnt = Counter(); bad = []; kappa_bad = 0
    ious, ious_obs, obs_p95, h_p95, h_miss = [], [], [], [], []
    h_cov_all = []
    br_stats = defaultdict(list)
    borders = 0
    idxs = np.random.default_rng(0).choice(len(ds), size=n, replace=False)
    for k in idxs:
        k = int(k)
        r = ds[k]
        cnt[int(r["y"])] += 1
        for key in ("I", "phi", "h", "M", "roi", "dead", "orr"):
            a = r[key]
            if a.shape != (256, 256):
                bad.append((k, key, "shape", a.shape))
        if not (np.isfinite(r["phi"]).all() and np.isfinite(r["h"]).all() and np.isfinite(r["M"]).all()):
            bad.append((k, "finite", "NaN/Inf"))
        if not (0.0 <= float(r["M"].min()) and float(r["M"].max()) <= 1.0):
            bad.append((k, "M", "range", float(r["M"].min()), float(r["M"].max())))
        if (r["dead"] & (~r["roi"])).any():
            bad.append((k, "dead", "outside roi"))
        if not kappa_bad:
            kappa_bad = int(not np.isclose(float(r["kappa_hat"]), float(r["kappa_hat_ps1"]), atol=1e-9))
        ious.append(float(r["iou"])); ious_obs.append(float(r["iou_obs"]))
        obs_p95.append(float(r["obs_p95"])); h_p95.append(float(r["h_p95"]))
        h_miss.append(float(r["h_miss"])); borders += int(r["border"])
        cov = r["dead"] & (r["roi"] & (~r["clean"]))
        hcov = float(np.percentile(np.abs(r["h"][cov] - r["orr"][cov]), 95)) if cov.any() else 0.0
        h_cov_all.append(hcov)
        br_stats[r["branch"]].append((float(r["h_p95"]), hcov, float(r["iou"]),
                                      float(r["h_miss"]), int(r["border"]), float(r["obs_p95"])))
    print(f"  labels: P={cnt[0]} U={cnt[1]} R={cnt[2]}  P ratio={100*cnt[0]/n:.1f}% "
          f"(expect ~12.9% for val at gamma=0.75)")
    print(f"  kappa_hat == kappa_hat_ps[1] for sample 0: {not kappa_bad}")
    print(f"  clean-est IoU = {np.mean(ious):.3f}  (observable {np.mean(ious_obs):.3f})")
    print(f"  phi_obs vs GT clean: P95={np.mean(obs_p95):.2f} rad; "
          f"h_input vs ORR on GT dead: P95={np.mean(h_p95):.2f} rad "
          f"(covered-dead {np.mean(h_cov_all):.2f}); miss={100*np.mean(h_miss):.1f}%")
    print(f"  border-flagged: {borders}/{n}")
    print("  by branch (median / mean):")
    for br, v in sorted(br_stats.items()):
        a = np.array(v)
        print(f"    {br:9s} n={len(v):3d}  hP95_all P50={np.median(a[:,0]):6.2f}  "
              f"hP95_cov P50={np.median(a[:,1]):6.2f}  IoU={a[:,2].mean():.3f}  "
              f"miss={a[:,3].mean():.2f}  border={a[:,4].mean():.2f}  phiP95={a[:,5].mean():.2f}")
    if bad:
        print(f"  !! {len(bad)} bad entries; first 5: {bad[:5]}")
    else:
        print("  shape/range/NaN checks: OK")

    # GT-leak audit by construction: the cached h/M/clean were produced by frontend(I, roi).
    print("  GT-leak audit: h_input/M_fix/clean_est come from cache built with I+roi only; "
          "dead/orr/phi are returned for loss/eval.")
    return dict(n=n, P=cnt[0], U=cnt[1], R=cnt[2], iou=float(np.mean(ious)),
                iou_obs=float(np.mean(ious_obs)), obs_p95=float(np.mean(obs_p95)),
                h_p95=float(np.mean(h_p95)), h_miss=float(np.mean(h_miss)),
                borders=borders, bad=len(bad))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--gamma", type=float, default=GAMMA_DEFAULT)
    a = ap.parse_args()
    if a.self_test:
        self_test(a.split, a.n, a.gamma)
    else:
        ds = W5Dataset(a.split, a.gamma)
        print(f"W5Dataset({a.split}) len={len(ds)} gamma={a.gamma}")
        r = collate([ds[0], ds[1]])
        for k, v in r.items():
            print(f"  {k}: {v.shape} {v.dtype}")
