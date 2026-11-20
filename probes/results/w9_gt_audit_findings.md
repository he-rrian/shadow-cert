# W9 真实 11 场景 GT audit v2 — 锁定口径与当前状态

**日期**：2026-11-20 ｜ **代码**：`code/w9_gt_audit.py` ｜ **结果**：`w9_gt_audit.json` / `w9_gt_audit.md`

**状态**：GT audit 完成；按锁定规则 **N_eff = 6 < 10 → G7 = insufficient**（不是 fail，也不是 pass）。W9 六项零重训继续有效；W10 三项可推进；G7 等 scene_02/03/06 的 badmask/相位修复与 re-audit。

---

## 0. 当前审计表（锁定后）

| 场景 | Tier | 判定 | 关键数字 |
|---|---|---|---|
| scene_01 | A | PASS | median 0.0020, P95 0.0062 |
| scene_02 | A | **GT-insufficient** | median 0.0104, P95 12.57；42–48% 像素差 2π |
| scene_03 | A | **GT-insufficient** | median 0.0132, P95 12.55；42–48% 像素差 2π |
| scene_04 | A | PASS | median 0.0020, P95 0.0184 |
| scene_05 | A | PASS | median 0.0062, P95 0.0211 |
| scene_06 | A | **GT-insufficient** | median 0.0093, P95 6.28；42–48% 像素差 2π |
| Train | A | PASS | 偏差相位+平面：median 0.0016, P95 0.0048 |
| 实测1 | A | **Tier A FAIL** | 偏差相位+平面：median 0.1162, P95 0.5063 vs 0.1/0.3 |
| 实测2 | B | PASS | model 0.0816；raw dir BAD → canonical ok；first5 N/A |
| 实验3 | B | **FAIL** | model 0.132 > 0.10；raw dir BAD → canonical ok |
| 实验4 | B | PASS | model 0.0681；dir ok |

**N_eff 计算**：
```
N_eff = 11 - n_fail_A - n_fail_B
n_fail_A = 4   (scene_02/03/06 GT-insufficient + 实测1 Tier A FAIL)
n_fail_B = 1   (实验3 model residual 0.132 > 0.10)
N_eff = 6
```
> 注：不是 7。实验3 在 Q4/Q5 修正后仍然 model residual 0.132 > 0.10，必须算 n_fail_B=1。状态仍是 insufficient，但分母必须写对。

---

## 1. 锁定决定（2026-11-20）

| # | 决定 | 备注 |
|---|---|---|
| 1 | Train 允许“偏差相位 + 拟合载频平面” | 残差 median 0.0016 / P95 0.0048；不修改 GT 数据；平面拟合确定性、预先定义 |
| 2 | 实测1 保持 **Tier A FAIL**，只作定性 | 原始 0.1162 / 0.5063 vs 0.1 / 0.3 完整保留；**不降 Tier B、不放宽阈值**；“验证失败 ≠ 没有验证” |
| 3 | scene_02/03/06 先按“不可像素级评价” | 不按连通域自动对齐；不用质量导向解包当 GT（循环论证）；修好 badmask 再 re-audit |
| 4 | first≈fifth = **N/A** | 第 5 帧不是 2π 重复（相关 0.36–0.68），`fringe.m` 只用 4 帧；协议修正，不是看结果删检查 |
| 5 | Tier B 先 canonicalization 再查方向 | 镜像到 +x、翻符号匹配 carrier；raw/canonical 两套都写进 audit 表 |
| 6 | N_eff<10 = **insufficient** | 不是 fail/pass；不能在“把失败场景算进去”的分母上算 G7 |
| 7 | “P 但 GT 不可评”单列 | `P-class, GT-insufficient`；不进 P_pass 分子，也不进分母；不默认挂/过 |
| 8 | scene_02/03/06 re-audit 预注册 | 相位门槛 + mask 门槛，见 §3 |

---

## 2. 三条已确认的 GT 口径事实（有数字）

1. **Train GT 是偏差相位**：`gt15_rad` 梯度 dx 0.0003 rad/px，15 步总相位 dx -0.63 rad/px；`phi ≈ -gt15_rad + a + bx·x + by·y`，残差 0.0016/0.0048 rad。
2. **实测1 mat 是偏差相位但略超阈值**：符号 -1 + 平面，median 0.1162 / P95 0.5063；8 种转置/翻转都没有更好方向；它是外部 four-step 方法输出，不是独立总相位 GT。
3. **scene_02/03/06 是旧 GT 管线的 2π 分支**：`gt_phase.npy` 由 `make_single_shot.py` 的 `skimage.restoration.unwrap_phase` 生成；与质量导向解包在一个大连通域上差 2π（42–48% 像素；P95 12.57），且 badmask 与相位是同一次旧管线产物，不对齐。归因是系统性旧管线问题，不是随机场景难；**不用于免责**。

---

## 3. scene_02/03/06 re-audit 预注册（锁定）

**相位门槛**：
- 按连通域 `k_c = round(median(Δ / 2π))`，`residual = Δ − 2π·k_c`；
- `P95(residual) ≤ 0.1 rad` 且 `|residual| > 0.5 rad` 的像素 `<1%`。

**mask 门槛**（用户 2026-11-20 追加）：
- 修复后的 badmask 与质量导向解包 invalid mask（或独立调制度 mask）`IoU ≥ 0.95`；
- ROI 边界偏差 `≤2 px`。

**规则**：两条都过才恢复像素级评价；否则维持“不可像素级评价/定性”。不按连通域自动对齐，不用质量导向解包当 GT。

---

## 4. 当前状态与下一步

- **W9 六项零重训**：✅ 完成；官方 C3 float32 主表未动。
- **GT audit v2**：✅ 完成；N_eff=6，G7 insufficient。
- **G7**：⏸ 等 scene_02/03/06 的 badmask/相位修复与 re-audit；若其中被 C1 判 U，可回到 N_eff 并计 handled；判 P 且 GT 不可评 → 单列 `P-class, GT-insufficient`。
- **W10**：3 项可继续（去 phasor 7→3、去 L_fc、hard-split 1:1 上采样 vs 主表纯混合）；`log T̂` 不硬映射，等 code audit；尺度增广按“S5 配方本就没有”作废。
- **服务器**：真实推理仍需 GPU；等用户通知后开。G7 暂时 insufficient，不阻塞 W10 代码准备。
