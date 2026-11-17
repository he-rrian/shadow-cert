"""w4_sf_recal.py -- zero-retrain, target-domain gamma recalibration on SynthFringe.

Question: the MATLAB-calibrated certificate does not transfer to SynthFringe
(gamma=1: 0/450).  If we keep the SAME model-free certificate and only
re-calibrate gamma on SF-val (no network, no retraining), does a usable SF
certificate exist?  SF-val -> SF-test (test is one-shot, not used to tune).

Protocol is the same as W4:
  cert      : gamma*kappa_hat*vinf + delta_g <= phi_tol
  A-cert    : certified and per-pixel MAX error > gamma*kappa_hat*vinf + delta_g
  B         : certified and per-pixel MAX error > phi_tol
  rate      : one-sided 95% Clopper-Pearson upper bound
  gamma*    : smallest gamma with n_cert >= N_MIN and CP95(A-cert) <= eps

2026-11-17 S5 revision:
  * one sample per scene (450 unique vinf scenes per split), so n is not inflated
    by repeated scenes;
  * per-family CP (sliver / arc / blob) is the MAIN presentation, pooled CP is a
    reference only -- the three families have 88% / 40% / 0% certification rates
    and mixing them makes the pooled CP uninformative;
  * 3-block diagnostics (certification rate and A-cert violation rate);
  * violation attribution: oracle (ORR itself > phi_tol) vs boundary-noise
    (ORR <= phi_tol but harmonic > phi_tol) vs bound-only;
  * required-N table for a CP-legal 20% certificate.

Run: python w4_sf_recal.py
Writes probes/results/w4_sf_recal.json / .md
"""
import os, json, time, importlib.util, sys
from collections import Counter
import numpy as np, cv2
from scipy.stats import beta

try:                                    # Windows console may be GBK; never crash on print
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "probes", "results")
PHI_TOL = float(np.pi / 10)
DELTA_G = 0.13
N_MIN = 50
VINF = ["sliver", "arc", "blob"]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


W4 = load("w4_ref", os.path.join(HERE, "w4_baseline.py"))


def cp_upper(k, n, alpha=0.05):
    if n <= 0:
        return float("nan")
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, k + 1.0, n - k))


def eval_split(split):
    d = os.path.join(DATA, f"sf_{split}")
    man_path = os.path.join(d, "manifest.json")
    man = json.load(open(man_path))
    cache = os.path.join(RES, f"sf_rows_{split}.json")
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(man_path):
        rows = json.load(open(cache))
        print(f"  {split}: {len(rows)} rows (cache)", flush=True)
        return rows
    rows = []
    t0 = time.time()
    for s in man["shards"]:
        z = np.load(os.path.join(d, s["path"]))
        for i in range(len(z["meta"])):
            m = json.loads(z["meta"][i])
            vinf = m.get("vinf")
            if vinf is None:
                continue
            dead = np.asarray(z["dead"][i], bool)
            if dead.sum() < 30:
                continue
            phi = np.asarray(z["phi"][i], np.float64)
            orr = np.asarray(z["orr"][i], np.float64)
            rng = np.random.default_rng(20261117 + int(m["sid"]) * 13 + VINF.index(m["branch"]))
            src = phi + rng.normal(0.0, DELTA_G / 2.0, phi.shape) * (~dead)
            har = W4.harmonic_fill(dead, src)
            e = np.abs(har[dead] - phi[dead])
            lap = np.abs(cv2.Laplacian(phi, cv2.CV_64F)[dead]).max()
            eo = np.abs(orr[dead] - phi[dead]).max()
            rows.append(dict(split=split, idx=i, id=m["id"], sid=int(m["sid"]),
                             block=int(m.get("block", int(m["sid"]) // 200)),
                             branch=m["branch"],
                             npx=int(dead.sum()), vinf=float(vinf),
                             k_hat=float(m["kappa_hat"]),
                             err_max=float(e.max()), err_p95=float(np.percentile(e, 95)),
                             err_mean=float(e.mean()),
                             orr_max=float(eo), k_max=float(lap),
                             c5=float(eo / (lap * vinf)) if lap > 0 else None))
        print(f"  {split}: {len(rows)} rows  {time.time() - t0:.0f}s", flush=True)
    json.dump(rows, open(cache, "w"))
    return rows


def curve(rows):
    out = []
    for g in np.round(np.arange(0.0, 1.6001, 0.05), 4):
        bounds = np.array([g * r["k_hat"] * r["vinf"] + DELTA_G for r in rows])
        errs = np.array([r["err_max"] for r in rows])
        cert = bounds <= PHI_TOL
        n_cert = int(cert.sum())
        kA = int(np.sum(cert & (errs > bounds)))
        kB = int(np.sum(cert & (errs > PHI_TOL)))
        out.append(dict(gamma=float(g), n=len(rows), n_cert=n_cert,
                        cert_rate=n_cert / max(len(rows), 1),
                        A=kA, A_rate=kA / max(n_cert, 1), A_cp=cp_upper(kA, n_cert),
                        B=kB, B_rate=kB / max(n_cert, 1), B_cp=cp_upper(kB, n_cert)))
    return out


def pick(curve_, eps, n_min=N_MIN):
    for rec in curve_:
        if rec["n_cert"] >= n_min and np.isfinite(rec["A_cp"]) and rec["A_cp"] <= eps:
            return rec
    return None


def at_gamma(rows, gamma):
    """Pooled or per-family statistics at one gamma."""
    if not rows or gamma is None:
        return dict(n=0, n_cert=0, cert_rate=None, A=0, A_rate=None, A_cp=None,
                    B=0, B_rate=None, B_cp=None)
    bounds = np.array([gamma * r["k_hat"] * r["vinf"] + DELTA_G for r in rows])
    errs = np.array([r["err_max"] for r in rows])
    cert = bounds <= PHI_TOL
    n, n_cert = len(rows), int(cert.sum())
    kA = int(np.sum(cert & (errs > bounds)))
    kB = int(np.sum(cert & (errs > PHI_TOL)))
    return dict(n=n, n_cert=n_cert, cert_rate=n_cert / n,
                A=kA, A_rate=kA / max(n_cert, 1), A_cp=cp_upper(kA, n_cert),
                B=kB, B_rate=kB / max(n_cert, 1), B_cp=cp_upper(kB, n_cert))


def attribution(rows, gamma):
    """Split certified violations (err > bound) by their physical cause."""
    cats = Counter(); by_fam = Counter(); items = []
    if gamma is None:
        return dict(counts={}, by_family={}, items=[])
    for r in rows:
        bound = gamma * r["k_hat"] * r["vinf"] + DELTA_G
        if bound > PHI_TOL or r["err_max"] <= bound:
            continue
        if r["orr_max"] > PHI_TOL:
            cat = "oracle"          # even the ORR (exact-boundary harmonic) fails tolerance
        elif r["err_max"] > PHI_TOL:
            cat = "boundary_noise"  # ORR OK, harmonic (noisy boundary) exceeds tolerance
        else:
            cat = "bound_only"      # bound violated, tolerance still met
        cats[cat] += 1; by_fam[(cat, r["branch"])] += 1
        items.append(dict(id=r["id"], branch=r["branch"], block=r["block"],
                          vinf=r["vinf"], rho=(r["k_hat"] * r["vinf"] + DELTA_G) / PHI_TOL,
                          k_hat=r["k_hat"], err_max=r["err_max"], orr_max=r["orr_max"],
                          bound=float(bound), cat=cat))
    return dict(counts=dict(cats), by_family={f"{k[0]}/{k[1]}": v for k, v in by_fam.items()},
                items=items)


def n_needed(p, eps=0.20):
    """Smallest n_cert whose CP95 upper bound is <= eps for violation rate p."""
    for n in range(10, 20001):
        k = int(np.ceil(p * n - 1e-12))
        if cp_upper(k, n) <= eps:
            return n, k
    return None, None


def best_cp(cv, n_min=N_MIN):
    cands = [r for r in cv if r["n_cert"] >= n_min and np.isfinite(r["A_cp"])]
    return min(cands, key=lambda r: r["A_cp"]) if cands else None


def fmt_pct(x):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.1f}%"


def cp_cell(rec, eps):
    if rec["n_cert"] <= 0:
        return "—"
    s = f"{rec['A']}/{rec['n_cert']} = {100 * rec['A_rate']:.1f}% (CP {100 * rec['A_cp']:.1f}%)"
    if rec["A_cp"] > eps:
        s += " ⚠ 点估计参考，CP 不合法"
    return s


def main():
    val = eval_split("val")
    test = eval_split("test")
    cv, ct = curve(val), curve(test)

    by_eps = {}
    for eps in (0.05, 0.10, 0.20):
        rec = pick(cv, eps)
        ent = {"val": rec, "test": None, "best_val_cp": best_cp(cv)}
        if rec is not None:
            ent["test"] = next(r for r in ct if abs(r["gamma"] - rec["gamma"]) < 1e-9)
        by_eps[f"{eps:.2f}"] = ent

    g20 = by_eps["0.20"]["val"]
    gamma = g20["gamma"] if g20 else None

    fam_val = {br: at_gamma([r for r in val if r["branch"] == br], gamma) for br in VINF}
    fam_test = {br: at_gamma([r for r in test if r["branch"] == br], gamma) for br in VINF}
    pool_val = at_gamma(val, gamma)
    pool_test = at_gamma(test, gamma)

    # per-family CP-safe gamma (diagnostic only: the deployment gamma is the pooled one)
    fam_cv = {br: curve([r for r in val if r["branch"] == br]) for br in VINF}
    fam_ct = {br: curve([r for r in test if r["branch"] == br]) for br in VINF}
    fam_safe = {}
    for eps in (0.10, 0.20):
        key = f"{eps:.2f}"
        fam_safe[key] = {}
        for br in VINF:
            rec = pick(fam_cv[br], eps, 30)
            if rec is None:
                fam_safe[key][br] = None
            else:
                t = next(x for x in fam_ct[br] if abs(x["gamma"] - rec["gamma"]) < 1e-9)
                fam_safe[key][br] = dict(gamma=rec["gamma"], val=rec, test=t)

    blocks = sorted({r["block"] for r in val + test})
    blk = {}
    for b in blocks:
        blk[str(b)] = {
            "val": at_gamma([r for r in val if r["block"] == b], gamma),
            "test": at_gamma([r for r in test if r["block"] == b], gamma)}
    attr_val = attribution(val, gamma)
    attr_test = attribution(test, gamma)

    # conservative operating point: the gamma val selects for the 10% budget
    g_safe = by_eps["0.10"]["val"]["gamma"] if by_eps["0.10"]["val"] else None
    fam_val_s = {br: at_gamma([r for r in val if r["branch"] == br], g_safe) for br in VINF} if g_safe else {}
    fam_test_s = {br: at_gamma([r for r in test if r["branch"] == br], g_safe) for br in VINF} if g_safe else {}
    pool_val_s = at_gamma(val, g_safe) if g_safe else None
    pool_test_s = at_gamma(test, g_safe) if g_safe else None
    blk_s = {}
    if g_safe:
        for b in blocks:
            for sp, rows_sp in (("val", val), ("test", test)):
                if any(r["block"] == b for r in rows_sp):
                    blk_s[f"{b}/{sp}"] = at_gamma([r for r in rows_sp if r["block"] == b], g_safe)
    attr_val_s = attribution(val, g_safe) if g_safe else {}
    attr_test_s = attribution(test, g_safe) if g_safe else {}

    # trade-off frontier on SF-test: can any gamma give n_cert>=180 AND CP<=20%?
    front = []
    if g_safe is not None:
        for g in np.arange(0.10, 0.301, 0.01):
            r = at_gamma(test, float(g))
            front.append(dict(gamma=float(g), n_cert=r["n_cert"], A=r["A"], A_cp=r["A_cp"]))
    feasible = [f for f in front if f["n_cert"] >= 180 and f["A_cp"] <= 0.20]
    frontier_note = None
    if not feasible and front:
        c1 = [f for f in front if f["A_cp"] <= 0.20]
        c2 = [f for f in front if f["n_cert"] >= 180]
        best_n = max(c1, key=lambda f: f["n_cert"]) if c1 else None
        best_cp_row = min(c2, key=lambda f: f["A_cp"]) if c2 else None
        frontier_note = dict(best_n=best_n, best_cp=best_cp_row)

    # required-N table: use the observed test / val violation rates
    ntab = []
    obs = [("SF-val 激进口径", pool_val["A_rate"] if pool_val["n_cert"] else None, "val", pool_val),
           ("SF-test 激进口径", pool_test["A_rate"] if pool_test["n_cert"] else None, "test", pool_test)]
    if pool_test_s and pool_test_s["n_cert"]:
        obs.append(("SF-test 保守口径", pool_test_s["A_rate"], "test", pool_test_s))
    for name, p, split, rec in obs + [("参考 12.0%", 0.120, None, None)]:
        if p is None or p <= 0:
            continue
        n, k = n_needed(p, 0.20)
        if n is None:
            ntab.append(dict(label=name, p=float(p), n_cert=None, k=None,
                             note="p>eps：任何 n 都不可达"))
            continue
        row = dict(label=name, p=float(p), n_cert=n, k=k)
        if split and rec and rec["cert_rate"]:
            row["total"] = int(np.ceil(n / rec["cert_rate"]))
            row["split"] = split
        ntab.append(row)

    out = dict(protocol=dict(phi_tol=PHI_TOL, delta_g=DELTA_G, eps_grid=[0.05, 0.10, 0.20],
                             n_min=N_MIN, n_vinf=len(val), n_vinf_test=len(test),
                             unique_scenes_val=len({r["sid"] for r in val}),
                             unique_scenes_test=len({r["sid"] for r in test})),
               gamma_star=gamma, val_curve=cv, test_curve=ct,
               by_eps=by_eps, fam_val=fam_val, fam_test=fam_test,
               pool_val=pool_val, pool_test=pool_test,
               blocks=blk, attribution=dict(val=attr_val, test=attr_test),
               n_needed=ntab, fam_safe=fam_safe,
               safe=dict(gamma=g_safe, fam_val=fam_val_s, fam_test=fam_test_s,
                         pool_val=pool_val_s, pool_test=pool_test_s, blocks=blk_s,
                         attribution=dict(val=attr_val_s, test=attr_test_s)),
               frontier=front, frontier_note=frontier_note)
    json.dump(out, open(os.path.join(RES, "w4_sf_recal.json"), "w"), indent=1, default=float)

    L = []
    L += ["# SynthFringe 目标域重标定（零重训，唯一场景版）", "",
          f"phi_tol={PHI_TOL:.6f}  delta_g={DELTA_G}  n_min={N_MIN}", ""]
    L += [f"SF-val {len(val)} 个含 vinf 样本（唯一场景 {len({r['sid'] for r in val})}） / "
          f"SF-test {len(test)} 个（唯一场景 {len({r['sid'] for r in test})}）；每族 150。",
          "edge/cut/degraded 只作零样本误差评测，不进证书统计。", ""]
    L += ["> **C4 边界**：SF 只作 G5-b 零样本误差评测与目标域重标定诊断；"
          "MATLAB 的 ε=0.05 主口径（25.7%/25.2%，CP 3.23%/4.05%）才是 C4 的主证据。"
          "SF 的证书不得写成“跨域覆盖率保证”。", ""]

    L += ["## 1. 结论（先看这段）", ""]
    e05, e10 = by_eps["0.05"], by_eps["0.10"]
    L += [f"- **数据**：SF-val {len(val)} / SF-test {len(test)} 个含 vinf 样本，全部唯一场景；"
          f"γ=1 时 **0/{len(val)}**（sound 但零覆盖）。", ""]
    if gamma is None:
        L += ["**ε=0.20 也没有 CP 合法的 γ。** 如实记录，不调 γ 凑数。", ""]
    else:
        ok = pool_test["A_cp"] <= 0.20
        L += ["**两个工作点都由 SF-val 选出（test 只做一次性验证，未按 test 回调 γ）**：", "",
              "| 口径 | γ | val 认证 / CP95 | test 认证 / CP95 | 判定 |",
              "|---|---|---|---|---|",
              f"| 20% 最优（激进） | {gamma:.2f} | {pool_val['n_cert']}/{pool_val['n']}，"
              f"{fmt_pct(pool_val['A_cp'])} | {pool_test['n_cert']}/{pool_test['n']}，"
              f"{fmt_pct(pool_test['A_cp'])} | {'test 合法' if ok else '**test 不合法**'} |"]
        if g_safe:
            ok_s = pool_test_s["A_cp"] <= 0.20
            L += [f"| 10% 最优（保守，推荐 SF） | {g_safe:.2f} | {pool_val_s['n_cert']}/{pool_val_s['n']}，"
                  f"{fmt_pct(pool_val_s['A_cp'])} | {pool_test_s['n_cert']}/{pool_test_s['n']}，"
                  f"{fmt_pct(pool_test_s['A_cp'])} | "
                  f"{'pooled 合法（block 15 ⚠）' if ok_s else 'test 不合法'} |", ""]
            L += [f"- **分族（γ={g_safe:.2f}，推荐口径）**：sliver test "
                  f"{fam_test_s['sliver']['n_cert']}/150，CP {fmt_pct(fam_test_s['sliver']['A_cp'])}；"
                  f"arc {fam_test_s['arc']['n_cert']}/150，CP {fmt_pct(fam_test_s['arc']['A_cp'])}；"
                  f"blob {fam_test_s['blob']['n_cert']}/150（宽度律，见 §2.2）。"]
            b15 = blk_s.get("15/test")
            if b15:
                L += [f"- **但 block 15 不合法**：test block 15 {b15['n_cert']} 认证 / {b15['A']} 违例，"
                      f"CP {fmt_pct(b15['A_cp'])}；block 16 {fmt_pct(blk_s['16/test']['A_cp'])}、"
                      f"block 17 {fmt_pct(blk_s['17/test']['A_cp'])}。20% 只在 pooled/分族成立，"
                      "**block 15 上不成立**。"]
        if frontier_note:
            bn, bc = frontier_note["best_n"], frontier_note["best_cp"]
            L += [f"- **n_cert≥180 与 CP≤20% 在 SF-test 上不可同时满足**："
                  f"CP 合法的最大认证数 γ={bn['gamma']:.2f} → n_cert={bn['n_cert']}；"
                  f"n_cert≥180 的最小 CP γ={bc['gamma']:.2f} → {fmt_pct(bc['A_cp'])}。"
                  "不要为过门禁回调 γ。"]
        if e05["val"] is None:
            b = e05["best_val_cp"]
            L += [f"- **ε=0.05：无解**（n_cert≥{N_MIN} 下最好可达 CP "
                  f"{fmt_pct(b['A_cp'])} @ γ={b['gamma']:.2f}，n_cert={b['n_cert']}）。"]
        if e10["val"] is None:
            b = e10["best_val_cp"]
            L += [f"- **ε=0.10：无解**（最好 CP {fmt_pct(b['A_cp'])} @ γ={b['gamma']:.2f}）。"]
        else:
            t10 = e10["test"]
            L += [f"- **ε=0.10：val pooled {fmt_pct(e10['val']['A_cp'])}，test pooled "
                  f"{fmt_pct(t10['A_cp'] if t10 else None)}；分族 test sliver "
                  f"{fmt_pct(fam_test_s['sliver']['A_cp'])}、arc {fmt_pct(fam_test_s['arc']['A_cp'])} "
                  "→ **没有跨 split 的 10% 分族证书**。"]
        at = attr_test
        cnt = at["counts"]; tot = sum(cnt.values())
        L += [f"- **违例归因（γ={gamma:.2f}，test {tot} 个 A 违例）**：oracle {cnt.get('oracle', 0)}、"
              f"bound_only {cnt.get('bound_only', 0)}、boundary {cnt.get('boundary_noise', 0)}；"
              "主因不是边界 δ_g，而是 CAD 锐边使洞内真实 κ 超出环带 κ̂（光滑类假设被触发）。", ""]

    gstr = f"{gamma:.2f}" if gamma is not None else "—"
    L += ["## 2. 分族主表", "",
          f"### 2.1 激进口径（γ={gstr}，ε=0.20 最优）", "",
          "A-cert = 认证集内 `max_err > γκ̂vinf+δ_g`；B = 认证集内 `max_err > phi_tol`；括号为单侧 95% CP 上界。", "",
          "| 族 | split | n | 认证 n_cert | 认证率 | A-cert (CP95) | B (CP95) |",
          "|---|---|---|---|---|---|---|"]
    for br in VINF:
        for sp, fam in [("val", fam_val), ("test", fam_test)]:
            r = fam[br]
            L.append(f"| {br} | {sp} | {r['n']} | {r['n_cert']} | {100*r['cert_rate']:.1f}% | "
                     f"{cp_cell(r, 0.20)} | {r['B']}/{r['n_cert']} = {100*r['B_rate']:.1f}% (CP {100*r['B_cp']:.1f}%) |")
    for sp, r in [("val", pool_val), ("test", pool_test)]:
        L.append(f"| **pooled（参考）** | {sp} | {r['n']} | {r['n_cert']} | {100*r['cert_rate']:.1f}% | "
                 f"{cp_cell(r, 0.20)} | {r['B']}/{r['n_cert']} = {100*r['B_rate']:.1f}% (CP {100*r['B_cp']:.1f}%) |")
    L += ["", "> 表注：CP 合法 = `CP95(A-cert) ≤ ε`。本表 ε=0.20；带 ⚠ 的行是点估计参考，CP 不合法。"
          "pooled 行只说明“三族混在一起会怎样”，不能替代分族结论。", ""]

    if g_safe:
        L += [f"### 2.2 保守口径（γ={g_safe:.2f}，val 的 ε=0.10 工作点；推荐 SF）", "",
              "| 族 | split | n | n_cert | 认证率 | A-cert (CP95) |",
              "|---|---|---|---|---|---|"]
        for br in VINF:
            for sp, fam in [("val", fam_val_s), ("test", fam_test_s)]:
                r = fam[br]
                L.append(f"| {br} | {sp} | {r['n']} | {r['n_cert']} | {100*r['cert_rate']:.1f}% | "
                         f"{cp_cell(r, 0.20)} |")
        for sp, r in [("val", pool_val_s), ("test", pool_test_s)]:
            L.append(f"| **pooled（参考）** | {sp} | {r['n']} | {r['n_cert']} | {100*r['cert_rate']:.1f}% | "
                     f"{cp_cell(r, 0.20)} |")
        L += ["", "> 该口径下 sliver/arc 的分族 CP 均 ≤20%，blob 0 认证；但 **block 15 仍 >20%**（见 §3.2）。"
              "这是本报告最重要的限定：pooled/分族合法 ≠ block 条件合法。", ""]

    L += ["### 2.3 分族 CP-safe γ（各自校准，仅诊断）", "",
          "| family | ε=0.10：γ / n_cert / CP95 | ε=0.20：γ / n_cert / CP95 |",
          "|---|---|---|"]
    for br in VINF:
        cells = []
        for key in ("0.10", "0.20"):
            rec = fam_safe[key][br]
            if rec is None:
                cells.append("无解")
            else:
                cells.append(f"γ={rec['gamma']:.2f}，n_cert={rec['val']['n_cert']}，"
                             f"CP {fmt_pct(rec['val']['A_cp'])}")
        L.append(f"| {br} | {cells[0]} | {cells[1]} |")
    L += ["", "> 这张表只回答“若单独给某一族校准，能否拿到合法 CP”；部署**不能**按形状族分别调 γ"
          "（那会变成按形状族挑参数），它只用来诊断哪一族是瓶颈。", ""]

    L += ["## 3. 分 block 稳健性", "",
          f"### 3.1 激进口径（γ={gamma if gamma is not None else '—'}）", "",
          "| block | split | n | 认证 n_cert | 认证率 | A-cert | A rate | CP95 |",
          "|---|---|---|---|---|---|---|---|"]
    for b in blocks:
        for sp, rows_sp in (("val", val), ("test", test)):
            if not any(r["block"] == b for r in rows_sp):
                continue
            r = blk[str(b)][sp]
            flag = " ⚠" if r["A_cp"] > 0.20 else ""
            L.append(f"| {b} | {sp} | {r['n']} | {r['n_cert']} | {100*r['cert_rate']:.1f}% | "
                     f"{r['A']}/{r['n_cert']} | {100*r['A_rate']:.1f}% | {100*r['A_cp']:.1f}%{flag} |")
    L += ["", "> 3 个 block 用的是同一 γ*；任何单 block 的认证率/违例率偏离总体 >10 个百分点都应视为分布不稳。", ""]

    if g_safe:
        L += [f"### 3.2 保守口径（γ={g_safe:.2f}，推荐）", "",
              "| block | split | n | 认证 n_cert | 认证率 | A-cert | A rate | CP95 |",
              "|---|---|---|---|---|---|---|---|"]
        for b in blocks:
            for sp, rows_sp in (("val", val), ("test", test)):
                if not any(r["block"] == b for r in rows_sp):
                    continue
                r = blk_s[f"{b}/{sp}"]
                flag = " ⚠" if r["A_cp"] > 0.20 else ""
                L.append(f"| {b} | {sp} | {r['n']} | {r['n_cert']} | {100*r['cert_rate']:.1f}% | "
                         f"{r['A']}/{r['n_cert']} | {100*r['A_rate']:.1f}% | {100*r['A_cp']:.1f}%{flag} |")
        b15 = blk_s.get("15/test")
        if b15:
            L += ["", f"> **即使 pooled/分族合法，block 15 仍不合法**：{b15['n_cert']} 认证 / "
                  f"{b15['A']} 违例（{100*b15['A_rate']:.1f}%），CP95 {100*b15['A_cp']:.1f}% > 20%。"
                  "所以 20% 只能写成 **pooled 保证**，不能写成 block 条件保证；"
                  "这是 SF 域内异质性，不是样本量问题。", ""]

    L += ["## 4. 违例归因（认证却超证书界）", "",
          "把 A-cert 违例拆成三类：**oracle**（连 ORR 都超 φ_tol，说明洞内真实曲率超出光滑类/κ̂ 环带假设）、"
          "**boundary_noise**（ORR 没超 φ_tol 但带噪调和超了，说明 δ_g 低估/边界畸变）、"
          "**bound_only**（只超证书界、未超容差，属保守界本身偏紧，不是容差失败）。", "",
          "| split | category | count | 占 A-cert 违例 | 物理解释 |",
          "|---|---|---|---|---|"]
    for sp, at in [("val", attr_val), ("test", attr_test)]:
        total = sum(at["counts"].values())
        expl = {"oracle": "洞内真实 κ 远超环带 κ̂（锐边/台阶），光滑类假设被触发",
                "boundary_noise": "边界 δ_g 超过 0.13 rad（引理 2 低估）",
                "bound_only": "证书界偏紧，容差仍满足"}
        for cat in ("oracle", "boundary_noise", "bound_only"):
            c = at["counts"].get(cat, 0)
            L.append(f"| {sp} | {cat} | {c} | {100*c/max(total,1):.0f}% | {expl[cat]} |")
    L += ["", "分族计数：", ""]
    for sp, at in [("val", attr_val), ("test", attr_test)]:
        L.append(f"- {sp}: " + ("，".join(f"{k}={v}" for k, v in at["by_family"].items()) or "无"))
    if g_safe:
        ats = attr_test_s
        tots = sum(ats["counts"].values())
        L += ["", f"> 保守口径 γ={g_safe:.2f} 的 test 只有 {tots} 个 A 违例："
              f"oracle {ats['counts'].get('oracle', 0)}、bound_only {ats['counts'].get('bound_only', 0)}、"
              f"boundary {ats['counts'].get('boundary_noise', 0)}。"
              f"oracle 仍占 {100*ats['counts'].get('oracle', 0)/max(tots,1):.0f}%；"
              "说明主因是锐边导致的 κ 类违反，不是 δ_g 边界噪声。"]
    L += [""]

    L += ["## 5. 严格协议下（每个 eps 各自的 val 最优 γ）的 test 结果", "",
          "| eps | γ*（val） | val pooled CP95 | test pooled CP95 | 分族 test（CP95） | 判定 |",
          "|---|---|---|---|---|---|"]
    for eps in (0.05, 0.10, 0.20):
        key = f"{eps:.2f}"
        rec = by_eps[key]["val"]
        if rec is None:
            b = by_eps[key]["best_val_cp"]
            L.append(f"| {eps} | — | 最好 {fmt_pct(b['A_cp'])} @ γ={b['gamma']:.2f} | — | — | **无解** |")
            continue
        t = by_eps[key]["test"]
        famsrc = {br: at_gamma([r for r in test if r["branch"] == br], rec["gamma"]) for br in VINF}
        famtxt = "，".join(f"{br} {fmt_pct(famsrc[br]['A_cp'])}" for br in VINF)
        verdict = "**test 不合法**" if (t and t["A_cp"] > eps) else "test 合法"
        L.append(f"| {eps} | {rec['gamma']:.2f} | {fmt_pct(rec['A_cp'])} | "
                 f"{fmt_pct(t['A_cp'] if t else None)} | {famtxt} | {verdict} |")
    if g_safe:
        L += ["", f"> 唯一 CP 合法的 SF-test 20% 证书出现在**保守口径 γ={g_safe:.2f}**"
              f"（它是 val 的 ε=0.10 工作点，不是 ε=0.20 的 val 最优）："
              f"test {pool_test_s['n_cert']} 认证，CP {fmt_pct(pool_test_s['A_cp'])}。"
              f"严格协议下 ε=0.20 的 val 最优 γ={gamma:.2f} 在 test 上 CP {fmt_pct(pool_test['A_cp'])}，不合法。"
              "这不是调 γ 凑出来的：两个 γ 都在 val 上选出，test 只做验证。", ""]
    L += ["> 分族列用的是该 eps 自己的 γ*（不是别的口径）；报告时必须同时给出 γ，不能只写一个 CP。", ""]

    L += ["## 6. 前沿与所需 N", "", "### 6.1 n_cert vs CP95 前沿（SF-test，γ 细网格）", "",
          "| γ | n_cert | A-cert | CP95 | 备注 |",
          "|---|---|---|---|---|"]
    for f in front:
        if 0.14 <= f["gamma"] <= 0.2201:
            note = ""
            if f["n_cert"] >= 180 and f["A_cp"] > 0.20:
                note = "n_cert 达标，CP 超"
            elif f["n_cert"] < 180 and f["A_cp"] <= 0.20:
                note = "CP 达标，n_cert 不足"
            L.append(f"| {f['gamma']:.2f} | {f['n_cert']} | {f['A']} | {fmt_pct(f['A_cp'])} | {note} |")
    if frontier_note:
        L += ["", "> **没有任何 γ 同时满足 `n_cert≥180` 与 `CP95≤20%`。**"
              f"CP 合法的最大认证数是 γ={frontier_note['best_n']['gamma']:.2f} → n_cert={frontier_note['best_n']['n_cert']}；"
              f"n_cert≥180 的最小 CP 是 γ={frontier_note['best_cp']['gamma']:.2f} → {fmt_pct(frontier_note['best_cp']['A_cp'])}。"
              "这不是“样本量再大一点”能解决的：在激进 γ 上违例率 21.9% > 20%，N 再大 CP 也只会收敛到 21.9%。"
              "要么用更保守的 γ（n_cert 下降），要么改进 κ̂（见 §4 的 oracle 违例）。", ""]

    L += ["### 6.2 所需 N（若违例率 p 已定）", "",
          "| 口径 | 违例率 p | 需要 n_cert | CP95 | 按实测认证率折算总样本 |",
          "|---|---|---|---|---|"]
    for row in ntab:
        if row.get("n_cert") is None:
            L.append(f"| {row['label']} | {100*row['p']:.1f}% | 不可达 | — | {row.get('note','')} |")
            continue
        tot = f"{row['total']}（{row.get('split','—')}）" if row.get("total") else "—"
        L.append(f"| {row['label']} | {100*row['p']:.1f}% | {row['n_cert']} | ≤20% | {tot} |")
    L += ["", "> 折算用同一 split 的认证率；SF 现有 450 个唯一 vinf 样本（每族 150）不够时，"
          "论文写“该域需要 >N 个同域标注样本”，不要调 γ。", ""]

    text = "\n".join(L)
    open(os.path.join(RES, "w4_sf_recal.md"), "w", encoding="utf-8").write(text)
    print(text)
    print("\nsaved", os.path.join(RES, "w4_sf_recal.json"))


if __name__ == "__main__":
    main()
