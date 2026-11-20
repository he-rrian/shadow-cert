"""w9_real_c1.py -- C1 P/U labels on the 11 real scenes (CPU only, no model).

The C1 certificate is model-independent: it depends on the estimated ROI/dead zone,
the torsion width vinf, and the deployed kappa_hat from the observed phase.  This
script runs the single-frame frontend on each real scene and records:

    rho = (gamma * kappa_hat * vinf + delta_g) / phi_tol
    P if rho <= 1 and the dead zone is interior; U otherwise.

No GT is used here; this is exactly the deployment-time gate.  The result is used to
update N_eff: a scene that C1 rejects as U needs no pixel-level GT and counts as
handled in G7.

Run:
  python code/w9_real_c1.py
"""
import os, sys, json, math
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from w5_prep_frontend import frontend
from w3_gen import torsion_vinf
from w9_roi import estimate_roi

GAMMA = 0.75
DELTA_G = 0.13
PHI_TOL = math.pi / 10.0
COLLAR = 2
P1 = r"F:\paper\1\phase"
REAL = r"F:\paper\2\model\real\captures\proc"

SCENES = [
    ("scene_01", os.path.join(REAL, "scene_01"), "single_shot.png"),
    ("scene_02", os.path.join(REAL, "scene_02"), "single_shot.png"),
    ("scene_03", os.path.join(REAL, "scene_03"), "single_shot.png"),
    ("scene_04", os.path.join(REAL, "scene_04"), "single_shot.png"),
    ("scene_05", os.path.join(REAL, "scene_05"), "single_shot.png"),
    ("scene_06", os.path.join(REAL, "scene_06"), "single_shot.png"),
    ("Train", os.path.join(P1, "Train"), "input00.png"),
    ("实测1", os.path.join(P1, "实测1"), "fpp_shice_1_1.bmp"),
    ("实测2", os.path.join(P1, "实测2"), "fpp_shice_2_1.bmp"),
    ("实验3", os.path.join(P1, "实验3"), "cropped_fpp_shice_3_1.jpg"),
    ("实验4", os.path.join(P1, "实验4"), "cropped_fpp_shice_3_1.jpg"),
]


def im(p):
    a = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(p)
    if a.ndim == 3:
        a = cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2GRAY)
    return a.astype(np.float32)


def deployable_kappa(phi_obs, dead, roi):
    """Fixed-kernel P75 on the delta_g collar of the observed phase (no GT)."""
    lap = cv2.Laplacian(phi_obs.astype(np.float32), cv2.CV_32F)
    k = np.ones((2 * COLLAR + 1, 2 * COLLAR + 1), np.uint8)
    dom = (cv2.dilate(dead.astype(np.uint8), k) > 0) & roi
    collar = dom & (~dead)
    v = np.abs(lap[collar])
    if v.size < 20:
        return float("nan"), int(v.size)
    return float(np.percentile(v, 75)), int(v.size)


def main():
    rows = []
    for name, d, f in SCENES:
        p = os.path.join(d, f)
        I = im(p)
        est, fb, empty = estimate_roi(I)
        rec = dict(scene=name, file=f, shape=list(I.shape), roi_frac=float(est.mean()),
                   roi_fallback=bool(fb), roi_empty=bool(empty))
        if empty or est.sum() < 100:
            rec.update(c1="U", reason="empty_roi", rho=None, vinf=None, kappa=None)
            rows.append(rec); print(rec, flush=True); continue
        try:
            fe = frontend(I, est)
        except Exception as ex:
            rec.update(c1="U", reason=f"frontend_error:{type(ex).__name__}", rho=None, vinf=None, kappa=None)
            rows.append(rec); print(rec, flush=True); continue
        dead = fe["dead"].astype(bool); roi = est.astype(bool)
        vinf = torsion_vinf(dead, roi)
        kappa, ncol = deployable_kappa(fe["phi_obs"], dead, roi)
        border = bool(fe.get("border", False))
        rho = None
        if vinf is None or not np.isfinite(kappa):
            lab = "U"; reason = "no_vinf_or_kappa"
        else:
            rho = (GAMMA * kappa * vinf + DELTA_G) / PHI_TOL
            if border:
                lab = "U"; reason = "border_dead"
            elif rho <= 1.0:
                lab = "P"; reason = "rho<=1"
            else:
                lab = "U"; reason = "rho>1"
        rec.update(c1=lab, reason=reason, rho=None if rho is None else float(rho),
                   vinf=None if vinf is None else float(vinf),
                   kappa=None if not np.isfinite(kappa) else float(kappa),
                   n_collar=ncol, dead_frac=float(dead.mean()),
                   roi_frac_used=float(roi.mean()), border=border)
        rows.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    out = dict(date="2026-11-20", gamma=GAMMA, delta_g=DELTA_G, phi_tol=PHI_TOL,
               scenes=rows)
    os.makedirs(os.path.join(ROOT, "probes", "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "probes", "results", "w9_real_c1.json"), "w"),
              indent=1, ensure_ascii=False)
    print("saved probes/results/w9_real_c1.json")


if __name__ == "__main__":
    main()
