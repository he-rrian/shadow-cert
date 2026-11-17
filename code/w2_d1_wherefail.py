"""W2/D1 diagnosis -- WHERE do the 8->64 residual outliers sit?

For the failing scenes, characterise pixels with |e| > pi/2 by
  * distance to the ROI outline  (dist transform)
  * 64-level modulation percentile
  * local phase gradient
Decides the M_gt rule: outline-driven (erode) vs concave-interreflection-driven (residual mask).
Run: python w2_d1_wherefail.py
"""
import os
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
g = {"__file__": os.path.join(HERE, "w2_d1_fix.py"), "__name__": "m"}
exec(open(g["__file__"], encoding="utf-8").read().split('if __name__')[0], g)
seq, phase_theory = g["seq"], g["phase_theory"]
poly_guided_unwrap, wrap, RATIO, SUBS = g["poly_guided_unwrap"], g["wrap"], g["RATIO"], g["SUBS"]

def analyse(sid, gate=0.5):
    P, A, MX = {}, {}, None
    for sub, N in SUBS:
        I = seq(sub, sid)
        P[sub], A[sub] = phase_theory(I)
        MX = I.max(0) if MX is None else np.maximum(MX, I.max(0))
    base = MX >= 25
    roi = base.copy()
    for s in A:
        roi &= (A[s] > gate * np.percentile(A[s][base], 50))
    roi = cv2.morphologyEx(roi.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    phi = {"1_12": poly_guided_unwrap(P["1_12"], roi)}
    res = {}
    for lo, hi in (("1_12", "3_16"), ("3_16", "8_6"), ("8_6", "64_15")):
        a = RATIO[(lo, hi)]
        d = a * phi[lo] - P[hi]
        off = float(np.angle(np.mean(np.exp(1j * wrap(d)[roi]))))
        k = np.round((d - off) / (2 * np.pi))
        phi[hi] = 2 * np.pi * k + P[hi] + off
        res[hi] = np.abs(wrap(d - off))
    e = res["64_15"]
    bad = roi & (e > np.pi / 2)
    # distance to the ROI outline (inside)
    dist = cv2.distanceTransform(roi.astype(np.uint8), cv2.DIST_L2, 5)
    # a pixel is "near outline" if dist < 3 px
    near = roi & (dist < 3.0)
    a64 = A["64_15"]
    p = np.percentile(a64[roi], [50, 90])
    print(f"=== {sid}: roi={roi.mean()*100:.1f}%  bad={bad.sum()} px ({bad.sum()/max(roi.sum(),1)*100:.2f}% of roi) ===")
    if bad.sum() == 0:
        return
    print(f"   dist-to-outline of BAD px : P50={np.median(dist[bad]):5.2f} P90={np.percentile(dist[bad],90):5.2f} "
          f"max={dist[bad].max():5.2f}")
    print(f"   dist-to-outline of ROI px : P50={np.median(dist[roi]):5.2f} P90={np.percentile(dist[roi],90):5.2f}")
    frac_near = float((dist[bad] < 3).mean())
    print(f"   fraction of bad px within 3 px of the outline : {frac_near*100:.1f}%   "
          f"(outline band is {near.sum()/roi.sum()*100:.1f}% of the ROI)")
    print(f"   amp64 of bad px  P50={np.median(a64[bad]):6.1f}   ROI amp64 P50={p[0]:6.1f} P90={p[1]:6.1f}")
    print(f"   amp64 of bad px relative to P50 : {np.median(a64[bad])/p[0]:.2f}x")

if __name__ == "__main__":
    for sid in ("01200", "00900", "03000", "01800", "00000"):
        analyse(sid)
