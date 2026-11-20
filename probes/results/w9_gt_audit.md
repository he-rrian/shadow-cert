# 真实 11 场景 GT audit（Tier A/B，G7 前置）

日期：2026-11-20 ｜ 代码：`code/w9_gt_audit.py` ｜ JSON：`probes/results/w9_gt_audit.json`

> 口径：Tier A 用独立 GT（scene_01–06 的 gt_phase.npy、Train 的 gt15_rad+mask、实测1 的 fourstep_purphase.mat）做 2π/offset/符号对齐后的交叉验证；Tier B（实测2、实验3/4）没有独立 GT，只有通过预注册自洽套件才进 G7。**没有一个场景可以“看起来对”就进 G7。**

## 0. 预注册阈值（看 G7 前写死）

| 项 | 阈值 |
|---|---:|
| valid_frac_min | 0.1 |
| wrap_p95_max | 0.05 |
| model_resid_med_max | 0.1 |
| first5_rms_max | 0.05 |
| dir_sign_match | True |
| unwrap_jump_frac_max | 0.005 |
| cross_med_max | 0.05 |
| cross_p95_max | 0.2 |
| cross_med_max_tierA_real | 0.1 |
| cross_p95_max_tierA_real | 0.3 |

## 1. 审计表

| 场景 | Tier | 帧/文件 | shape | 有效 mask | 相位范围 (rad) | unwrapped | 交叉验证 | median | P95 | max | wrap P95 | jump% | model resid | 1st≈5th | dir | pass |
|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| scene_01 | A | 4 frames | 256×256 | 0.960 | [-89.580, 79.921] | yes | gt_phase.npy | 0.001953 | 0.006226 | 12.58 | 2.694e-05 | 0.064 | 0.104 | — | BAD | **PASS** |
| scene_02 | A | 4 frames | 256×256 | 0.921 | [-178.144, 38.883] | yes | gt_phase.npy | 0.01038 | 12.57 | 163.4 | 6.986e-05 | 1.48 | 0.27 | — | BAD | **FAIL** |
| scene_03 | A | 4 frames | 256×256 | 0.916 | [-157.494, 10.888] | yes | gt_phase.npy | 0.01318 | 12.55 | 31.42 | 6.318e-05 | 0.8 | 0.0912 | — | ok | **FAIL** |
| scene_04 | A | 4 frames | 256×256 | 0.930 | [-92.306, 84.681] | yes | gt_phase.npy | 0.001953 | 0.01843 | 43.98 | 4.506e-05 | 0.34 | 0.166 | — | ok | **PASS** |
| scene_05 | A | 4 frames | 256×256 | 1.000 | [-99.264, 98.997] | yes | gt_phase.npy | 0.006226 | 0.02112 | 25.15 | 3.505e-05 | 0.337 | 0.109 | — | BAD | **PASS** |
| scene_06 | A | 4 frames | 256×256 | 1.000 | [-49.557, 143.557] | yes | gt_phase.npy | 0.009277 | 6.283 | 12.57 | 5.412e-05 | 0.295 | 0.138 | — | BAD | **FAIL** |
| Train | A | 15 frames | 480×640 | 0.391 | [-164.948, 196.257] | yes | gt15_rad.npy + gt15_mask.npy | 0.001616 | 0.004803 | 12.56 | 6.175e-05 | 0.0193 | 0.0357 | 0.4836 | ok | **PASS** |
| 实测1 | A | 5 frames | 700×700 | 0.970 | [-93.537, 179.465] | yes | fpp_shice_1_fourstep_purphase.mat | 64.32 | 149.6 | 163.8 | 0.0001376 | 0 | 0.0552 | 0.6785 | BAD | **FAIL** |
| 实测2 | B | 5 frames | 595×595 | 0.969 | [-166.256, 152.296] | yes | — | — | — | — | 0.0001194 | 0.0019 | 0.0816 | 0.5426 | BAD | **FAIL** |
| 实验3 | B | 4 frames | 667×640 | 0.991 | [-193.607, 194.154] | yes | — | — | — | — | 0.0002897 | 0.0246 | 0.132 | — | BAD | **FAIL** |
| 实验4 | B | 4 frames | 497×530 | 0.996 | [-157.648, 34.463] | yes | — | — | — | — | 7.606e-05 | 0.156 | 0.0681 | — | ok | **PASS** |

## 2. Tier A/B 与 N_eff

- Tier B 失败场景数：**2**；N_eff = 11 − 2 = **9**
- **G7 = insufficient**（N_eff < 10；不允许靠缩小分母过关）

## 3. 失败/风险备注

- **scene_02**：FAIL；cross={'ref': 'gt_phase.npy', 'sign': 1, 'k': -163, 'offset': 992.7471923828125, 'median': 0.0103759765625, 'p95': 12.5650634765625, 'max': 163.3756103515625, 'mean': 4.7214531898498535, 'model': 'offset'}；valid=0.921；wrap=6.986e-05；jump=1.48%；model_resid=0.27；first5=None；dir_ok=False
- **scene_03**：FAIL；cross={'ref': 'gt_phase.npy', 'sign': 1, 'k': -163, 'offset': 1068.1478271484375, 'median': 0.01318359375, 'p95': 12.5538330078125, 'max': 31.4202880859375, 'mean': 3.4629640579223633, 'model': 'offset'}；valid=0.916；wrap=6.318e-05；jump=0.80%；model_resid=0.0912；first5=None；dir_ok=True
- **scene_06**：FAIL；cross={'ref': 'gt_phase.npy', 'sign': 1, 'k': -163, 'offset': 879.6505126953125, 'median': 0.00927734375, 'p95': 6.2830810546875, 'max': 12.5653076171875, 'mean': 2.639371395111084, 'model': 'offset'}；valid=1.000；wrap=5.412e-05；jump=0.29%；model_resid=0.138；first5=None；dir_ok=False
- **实测1**：FAIL；cross={'ref': 'fpp_shice_1_fourstep_purphase.mat', 'sign': 1, 'k': -155, 'offset': 958.246826171875, 'median': 64.31643676757812, 'p95': 149.57565307617188, 'max': 163.78887939453125, 'mean': 69.18114471435547, 'model': 'offset'}；valid=0.970；wrap=0.0001376；jump=0.00%；model_resid=0.0552；first5=0.6785022020339966；dir_ok=False
- **实测2**：FAIL；cross=None；valid=0.969；wrap=0.0001194；jump=0.00%；model_resid=0.0816；first5=0.5426180958747864；dir_ok=False
- **实验3**：FAIL；cross=None；valid=0.991；wrap=0.0002897；jump=0.02%；model_resid=0.132；first5=None；dir_ok=False

> 注意：实测2 / 实验3 / 实验4 的“GT”是 4 步自洽参考，不是独立标定；只有通过上表套件才进 G7。Tier B 失败场景只做定性，并从 N_eff 中剔除。