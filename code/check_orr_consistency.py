"""check_orr_consistency.py -- S3 gate #5.  15-minute numerical alignment.

Four checks the reviewer asked for:
  A. same-domain / same-solver : ORR (w3_gen.orr_reference) vs Harmonic (w4_baseline.harmonic_fill)
     fed with the EXACT boundary -> must agree to machine precision.
  B. linear-carrier invariance:  adding a plane to psi must NOT change (ORR - psi).
                                 A plane has Delta = 0 both in continuum and in the 5-point stencil.
  C. discrete maximum principle: ratio = max|psi-ORR| / (max|Delta_5pt psi| * ||v||inf) <= 1.
                                 This separates a SOLVER bug from a kappa-ESTIMATION error.
  D. background contamination : how far is the phase just outside Omega_roi?  If a hole touches
     the ROI border, an uninformed nearest-neighbour fill can drag Omega_bg (a DIFFERENT plane,
     tens of rad away) into the hole -- that is the only way to get a 100 rad error on a 30 px hole.

Run: python check_orr_consistency.py [--shard ...]
"""
import os, json, argparse, glob
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "probes", "results")

import importlib.util
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

W3 = load("w3", os.path.join(HERE, "w3_gen.py"))
W4 = load("w4", os.path.join(HERE, "w4_baseline.py"))

def lap5(z):
    return cv2.Laplacian(z.astype(np.float64), cv2.CV_64F)

def check_one(phi, dead, roi, vinf, tag):
    H, W = phi.shape
    out = {"tag": tag, "area": int(dead.sum()), "vinf": vinf}
    # ---------- border bookkeeping (the reviewer's key point)
    out["touch_img_border"] = bool(dead[0, :].any() or dead[-1, :].any()
                                   or dead[:, 0].any() or dead[:, -1].any())
    out["touch_roi_border"] = bool((cv2.dilate(dead.astype(np.uint8),
                                               np.ones((3, 3), np.uint8)) > 0)[~roi].any())
    out["interior"] = not (out["touch_img_border"] or out["touch_roi_border"])
    # ---------- A. two code paths, SAME exact boundary (ring-extrapolated)
    orr = W3.orr_reference(dead, roi, phi).astype(np.float64)
    har = W4.harmonic_fill(dead, phi)
    out["A_max_abs_diff"] = float(np.max(np.abs(orr[dead] - har[dead]))) if dead.any() else 0.0
    # ---------- A'. clean Oracle: harmonic on the hole with the GT everywhere outside
    orr_direct = W3.harmonic_on(dead, phi)
    out["Ax_orrdirect_vs_ring"] = float(np.max(np.abs(orr_direct[dead] - orr[dead]))) if dead.any() else 0.0
    # ---------- B. linear-carrier invariance (on the CLEAN oracle)
    yy, xx = np.mgrid[0:H, 0:W]
    plane = 0.37 * xx + 0.11 * yy + 12.0
    orr2 = W3.harmonic_on(dead, phi + plane)
    r1 = orr_direct[dead] - phi[dead]
    r2 = orr2[dead] - (phi + plane)[dead]
    out["B_carrier_inv_max"] = float(np.max(np.abs(r1 - r2))) if dead.any() else 0.0
    # ---------- C. discrete maximum principle (on the CLEAN oracle)
    e = np.abs(orr_direct[dead] - phi[dead])
    L = float(np.max(np.abs(lap5(phi)[dead]))) if dead.any() else 0.0
    out["C_maxerr"] = float(np.max(e)) if dead.any() else 0.0
    out["C_maxlap"] = L
    out["C_ratio"] = float(out["C_maxerr"] / (L * max(vinf, 1e-9))) if L > 0 else np.nan
    # ---------- D. nearest-neighbour sanity
    nn = W4.nearest_fill(dead, phi)
    out["D_nearest_p95"] = float(np.percentile(np.abs(nn[dead] - phi[dead]), 95)) if dead.any() else 0.0
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=os.path.join(ROOT, "data", "train", "shard_098_0000.npz"))
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    d = np.load(a.shard)
    metas = [json.loads(s) for s in d["meta"]]
    rows = []
    print(f"{'#':>3s}{'branch':>9s}{'cls':>4s}{'area':>7s}{'vinf':>7s}"
          f"{'A(ring)':>10s}{'A2(direct)':>12s}{'B(carr)':>11s}{'C(ratio)':>10s}"
          f"{'D(nn95)':>9s}{'imgB':>6s}{'roiB':>6s}")
    for i in range(min(len(metas), a.n)):
        m = metas[i]; dead = d["dead"][i]; roi = d["roi"][i]
        vinf = m.get("vinf")
        if dead.sum() < 30 or vinf is None:
            continue
        r = check_one(d["phi"][i].astype(np.float64), dead, roi, float(vinf), m["id"])
        rows.append(r)
        print(f"{i:3d}{m['branch']:>9s}{m['cls']:>4s}{r['area']:7d}{vinf:7.1f}"
              f"{r['A_max_abs_diff']:10.2e}{r['Ax_orrdirect_vs_ring']:12.2e}"
              f"{r['B_carrier_inv_max']:11.2e}{r['C_ratio']:10.4f}{r['D_nearest_p95']:9.2f}"
              f"{str(r['touch_img_border']):>6s}{str(r['touch_roi_border']):>6s}")
    if not rows:
        raise SystemExit("no samples with vinf")
    print("\n--- summary ---")
    inter = [r for r in rows if r["interior"]]
    edge = [r for r in rows if not r["interior"]]
    print(f"  interior holes: {len(inter)}   border-touching: {len(edge)}")
    for grp, nm in ((inter, "INTERIOR"), (rows, "ALL")):
        if not grp:
            continue
        a_ = np.array([r["A_max_abs_diff"] for r in grp])
        b_ = np.array([r["B_carrier_inv_max"] for r in grp])
        c_ = np.array([r["C_ratio"] for r in grp], float)
        print(f"  [{nm}] A max={a_.max():.2e}  B max={b_.max():.2e}  "
              f"C max={np.nanmax(c_):.3f}  C>1: {int(np.nansum(c_ > 1.0))}/{len(grp)}")
    bad = [r for r in rows if not (r["A_max_abs_diff"] < 1e-6 and r["B_carrier_inv_max"] < 1e-6
                                   and r["C_ratio"] <= 1.0)]
    print(f"\n  samples failing the green criteria: {len(bad)}/{len(rows)}")
    for r in bad:
        print(f"     {r['tag']:26s} A={r['A_max_abs_diff']:.1e} B={r['B_carrier_inv_max']:.1e} "
              f"C={r['C_ratio']:.2f} imgB={r['touch_img_border']} roiB={r['touch_roi_border']}")
    json.dump(rows, open(os.path.join(RES, "orr_consistency.json"), "w"), indent=1, default=float)
    print("\nsaved", os.path.join(RES, "orr_consistency.json"))
