"""W2/D1 -- cascade with a MODULATION-GATED ROI.

Finding: the 1->3 failures came from low-modulation pixels (penumbra / interreflection),
not from topology.  Gating the ROI at amp > 0.5*P50 on every level removes them.

Run: python w2_d1_modroi.py
"""
import os, json
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
g = {"__file__": os.path.join(HERE, "w2_d1_fix.py"), "__name__": "m"}
exec(open(g["__file__"], encoding="utf-8").read().split('if __name__')[0], g)
seq = g["seq"]; phase_theory = g["phase_theory"]; poly_guided_unwrap = g["poly_guided_unwrap"]
wrap = g["wrap"]; RATIO = g["RATIO"]; SUBS = g["SUBS"]; RES = g["RES"]
os.makedirs(RES, exist_ok=True)

IDS = ["00000", "01200", "02399", "00360", "00600", "00900", "01500", "01800", "02400", "03000"]

def run(sid, gate=0.5, save=True):
    P, A, MX = {}, {}, None
    for sub, N in SUBS:
        I = seq(sub, sid)
        P[sub], A[sub] = phase_theory(I)
        MX = I.max(0) if MX is None else np.maximum(MX, I.max(0))
    base = MX >= 25
    roi = base.copy()
    for s in A:
        p50 = np.percentile(A[s][base], 50)
        roi &= (A[s] > gate * p50)
    roi = cv2.morphologyEx(roi.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    phi = {"1_12": poly_guided_unwrap(P["1_12"], roi)}
    rec = {"sid": sid, "roi": float(roi.mean())}
    for lo, hi in (("1_12", "3_16"), ("3_16", "8_6"), ("8_6", "64_15")):
        a = RATIO[(lo, hi)]
        d = a * phi[lo] - P[hi]
        off = float(np.angle(np.mean(np.exp(1j * wrap(d)[roi]))))
        k = np.round((d - off) / (2 * np.pi))
        phi[hi] = 2 * np.pi * k + P[hi] + off
        e = np.abs(wrap(d - off)[roi])
        rec[f"{lo}->{hi}"] = (float(np.percentile(e, 99)), float(np.mean(e > np.pi / 2)))
    p = phi["64_15"]

    # ---------------- M_gt : GT validity mask (decided from the residual analysis) ------
    #   |cascade residual| < pi/2      (order decision self-consistent)
    # & amp_64 > 0.5 * P50             (enough modulation at the finest level)
    # & dist to the ROI outline >= 2   (kills the 1-2 px foreground/background mixing band)
    e64 = np.abs(wrap(RATIO[("8_6", "64_15")] * phi["8_6"] - P["64_15"]))
    dist = cv2.distanceTransform(roi.astype(np.uint8), cv2.DIST_L2, 5)
    m_gt = roi & (e64 < np.pi / 2) & (A["64_15"] > 0.5 * np.percentile(A["64_15"][roi], 50)) & (dist >= 2.0)
    m_gt = cv2.morphologyEx(m_gt.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    rec["gt"] = float(m_gt.mean())

    dx = np.abs(np.diff(p, axis=1))[roi[:, :-1] & roi[:, 1:]]
    dy = np.abs(np.diff(p, axis=0))[roi[:-1, :] & roi[1:, :]]
    rec["tear"] = int((dx > np.pi).sum() + (dy > np.pi).sum())
    rec["ncomp"] = int(cv2.connectedComponents(roi.astype(np.uint8), 8)[0] - 1)
    if save:
        out = os.path.join(RES, "gt"); os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, f"{sid}.npz"),
                            phi64=p.astype(np.float32), m_roi=roi, m_gt=m_gt,
                            amp64=A["64_15"].astype(np.float32), e64=e64.astype(np.float32))
    return rec

if __name__ == "__main__":
    hdr = f"{'scene':7s}{'comp':>5s}{'roi%':>6s}{'gt%':>6s} | {'1->3 P99':>9s}{'f':>7s} | {'3->8 P99':>9s}{'f':>7s} | {'8->64 P99':>10s}{'f':>7s}{'tear':>6s}"
    print(hdr)
    rows = []
    for sid in IDS:
        try:
            r = run(sid)
        except Exception as ex:
            print(f"{sid}: ERR {type(ex).__name__} {ex}"); continue
        a1, a2, a3 = r["1_12->3_16"], r["3_16->8_6"], r["8_6->64_15"]
        print(f"{sid:7s}{r['ncomp']:5d}{r['roi']*100:6.1f}{r['gt']*100:6.1f} | {a1[0]:9.4f}{a1[1]*100:6.2f}% "
              f"| {a2[0]:9.4f}{a2[1]*100:6.2f}% | {a3[0]:10.4f}{a3[1]*100:6.2f}%{r['tear']:6d}")
        rows.append(r)
    if rows:
        print(f"\ntear total = {sum(r['tear'] for r in rows)} px over {len(rows)} scenes")
        print(f"M_gt coverage of the modulation ROI: "
              f"{100*np.mean([r['gt']/max(r['roi'],1e-9) for r in rows]):.2f}% (mean)")
        print(f"worst 8->64 frac>pi/2 = {max(r['8_6->64_15'][1] for r in rows)*100:.2f}%")
    json.dump(rows, open(os.path.join(RES, "w2_d1_modroi.json"), "w"), indent=1)
