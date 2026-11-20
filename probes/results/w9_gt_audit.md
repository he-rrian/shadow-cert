# 真实 11 场景 GT audit（Tier A/B，G7 前置）

日期：2026-11-20 ｜ 代码：`code/w9_gt_audit.py` ｜ JSON：`probes/results/w9_gt_audit.json`

> 口径：Tier A 用独立 GT（scene_01–06 的 gt_phase.npy、Train 的 gt15_rad+mask、实测1 的 fourstep_purphase.mat）做符号/2π/平面（Train、实测1）对齐后的交叉验证；Tier B（实测2、实验3/4）没有独立 GT，只有通过预注册自洽套件才进 G7。**没有一个场景可以“看起来对”就进 G7。**

## 0. 预注册阈值（看 G7 前写死）

| 项 | 阈值 |
|---|---:|
| valid_frac_min | 0.1 |
| wrap_p95_max | 0.05 |
| model_resid_med_max | 0.1 |
| dir_sign_match | True |
| unwrap_jump_frac_max | 0.005 |
| cross_med_max | 0.05 |
| cross_p95_max | 0.2 |
| cross_med_max_tierA_real | 0.1 |
| cross_p95_max_tierA_real | 0.3 |

## 1. 审计表

| 场景 | Tier | 帧 | shape | 有效 mask | 相位范围 (rad) | 交叉验证 | median | P95 | max | wrap P95 | jump% | model resid | 1st≈5th | dir raw→canon | 判定 |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|
| scene_01 | A | 4 | 256×256 | 0.960 | [-89.580, 79.921] | gt_phase.npy | 0.001953 | 0.006226 | 12.58 | 2.694e-05 | 0.064 | 0.104 | N/A | BAD→BAD | **PASS** |
| scene_02 | A | 4 | 256×256 | 0.921 | [-178.144, 38.883] | gt_phase.npy | 0.01038 | 12.57 | 163.4 | 6.986e-05 | 1.48 | 0.27 | N/A | BAD→BAD | **GT-insufficient (old skimage 2pi branch; not pixel-evaluable)** |
| scene_03 | A | 4 | 256×256 | 0.916 | [-157.494, 10.888] | gt_phase.npy | 0.01318 | 12.55 | 31.42 | 6.318e-05 | 0.8 | 0.0912 | N/A | ok→ok | **GT-insufficient (old skimage 2pi branch; not pixel-evaluable)** |
| scene_04 | A | 4 | 256×256 | 0.930 | [-92.306, 84.681] | gt_phase.npy | 0.001953 | 0.01843 | 43.98 | 4.506e-05 | 0.34 | 0.166 | N/A | ok→ok | **PASS** |
| scene_05 | A | 4 | 256×256 | 1.000 | [-99.264, 98.997] | gt_phase.npy | 0.006226 | 0.02112 | 25.15 | 3.505e-05 | 0.337 | 0.109 | N/A | BAD→BAD | **PASS** |
| scene_06 | A | 4 | 256×256 | 1.000 | [-49.557, 143.557] | gt_phase.npy | 0.009277 | 6.283 | 12.57 | 5.412e-05 | 0.295 | 0.138 | N/A | BAD→BAD | **GT-insufficient (old skimage 2pi branch; not pixel-evaluable)** |
| Train | A | 15 | 480×640 | 0.391 | [-164.948, 196.257] | gt15_rad.npy + gt15_mask.npy | 0.001616 | 0.004803 | 12.56 | 6.175e-05 | 0.0193 | 0.0357 | N/A | ok→ok | **PASS** |
| 实测1 | A | 5 | 700×700 | 0.970 | [-93.537, 179.465] | fpp_shice_1_fourstep_purphase.mat | 0.1162 | 0.5063 | 64.45 | 0.0001376 | 0 | 0.0552 | N/A | BAD→BAD | **Tier A FAIL (mat deviation+plane 0.116/0.506 > 0.1/0.3; qualitative only)** |
| 实测2 | B | 5 | 595×595 | 0.969 | [-152.296, 166.256] | — | — | — | — | 0.0001218 | 0.00205 | 0.0816 | N/A | BAD→ok | **PASS** |
| 实验3 | B | 4 | 667×640 | 0.991 | [-194.154, 193.607] | — | — | — | — | 0.0002964 | 0.023 | 0.132 | N/A | BAD→ok | **FAIL** |
| 实验4 | B | 4 | 497×530 | 0.996 | [-157.648, 34.463] | — | — | — | — | 7.606e-05 | 0.156 | 0.0681 | N/A | ok→ok | **PASS** |

**判定说明**：PASS = 通过预注册阈值；FAIL = 失败；GT-insufficient / Tier A FAIL = 保留原始数字但不进 G7 的 P 像素统计。

## 2. Tier A/B、N_eff 与 G7

```
N_eff = 11 - n_fail_A - n_fail_B
n_fail_A = 4  (scene_02/03/06: GT-insufficient; 实测1: Tier A FAIL)
n_fail_B = 1  (实验3: model residual 0.132 > 0.10)
N_eff   = 6
```
- **G7 = insufficient**（N_eff < 10；不是 fail，也不是 pass；不允许靠缩小分母过关）
- “P 但 GT 不可评”场景单列 `P-class, GT-insufficient`：不进 P_pass 分子，也不进分母；不默认挂、不默认过。
- 若 scene_02/03/06 后续被 C1 判 U，它们可回到 N_eff 并计 handled；判 P 则只能等 badmask 修复。

## 3. 失败/风险备注

- **scene_02**：GT-insufficient (old skimage 2pi branch; not pixel-evaluable)；cross={'ref': 'gt_phase.npy', 'sign': 1, 'k': -163, 'offset': 992.7471923828125, 'median': 0.0103759765625, 'p95': 12.5650634765625, 'max': 163.3756103515625, 'mean': 4.7214531898498535, 'model': 'offset'}；valid=0.921；wrap=6.986e-05；jump=1.48%；model_resid=0.27；dir raw/canon=False/False
- **scene_03**：GT-insufficient (old skimage 2pi branch; not pixel-evaluable)；cross={'ref': 'gt_phase.npy', 'sign': 1, 'k': -163, 'offset': 1068.1478271484375, 'median': 0.01318359375, 'p95': 12.5538330078125, 'max': 31.4202880859375, 'mean': 3.4629640579223633, 'model': 'offset'}；valid=0.916；wrap=6.318e-05；jump=0.80%；model_resid=0.0912；dir raw/canon=True/True
- **scene_06**：GT-insufficient (old skimage 2pi branch; not pixel-evaluable)；cross={'ref': 'gt_phase.npy', 'sign': 1, 'k': -163, 'offset': 879.6505126953125, 'median': 0.00927734375, 'p95': 6.2830810546875, 'max': 12.5653076171875, 'mean': 2.639371395111084, 'model': 'offset'}；valid=1.000；wrap=5.412e-05；jump=0.29%；model_resid=0.138；dir raw/canon=False/False
- **实测1**：Tier A FAIL (mat deviation+plane 0.116/0.506 > 0.1/0.3; qualitative only)；cross={'ref': 'fpp_shice_1_fourstep_purphase.mat', 'sign': -1, 'a': -93.360166584972, 'bx': 0.3854492212836418, 'by': 0.0040074199443477215, 'median': 0.11619104668435654, 'p95': 0.5062942912433431, 'max': 64.45252059408962, 'mean': 0.17765572917680966, 'model': 'deviation+plane'}；valid=0.970；wrap=0.0001376；jump=0.00%；model_resid=0.0552；dir raw/canon=False/False
- **实验3**：Tier B FAIL；cross=None；valid=0.991；wrap=0.0002964；jump=0.02%；model_resid=0.132；dir raw/canon=False/True

## 4. scene_02/03/06 re-audit 预注册（锁定 2026-11-20）

- **相位门槛**：按连通域 `k_c=round(median(Δ/2π))`，`residual=Δ−2πk_c`；`P95(residual) ≤ 0.1 rad` 且 `|residual|>0.5 rad` 的像素 `<1%`。
- **mask 门槛**：修复后的 badmask 与质量导向解包 invalid mask（或独立调制度 mask）`IoU ≥ 0.95`，且 ROI 边界偏差 `≤2 px`。
- 两条都过才恢复像素级评价；否则维持“不可像素级评价/定性”。**不按连通域自动对齐，不用质量导向解包当 GT。**

## 5. 归因（不用于免责）

- 6 个 captures 里 3 个（scene_02/03/06）挂在同一条旧 `skimage.unwrap_phase` GT 管线上；失败不是随机分布，而是旧管线在大连通域上的系统性 2π 跳变。
- badmask 与相位是同一次旧管线产物，所以 mask 也不对齐；处置仍是“修好 / 降级 / insufficient”，不是免责。

> 注意：实测2 / 实验3 / 实验4 的“GT”是 4 步自洽参考，不是独立标定；只有通过上表套件才进 G7。Tier B 失败场景只做定性，并从 N_eff 中剔除。