# 题目2 工作区（P2）

论文：**Certified Recoverability and Distribution-Free Intervals for Single-Shot Fringe Projection Profilometry**
（包 C：可行解集合辨识 + 条件覆盖 + 风险约束的接受/拒绝）

> 规则：题目 2 的所有产物只放本目录；题目 3 的东西不要放进来。临时文件随手删。

## 当前进度（2026-11-20 19:30）

S0 ✅ → S1 ✅ → S2 ✅ → S3 ✅ → S4 ✅ → **S5 ✅（5 seeds，公平档 5/5）** → **S6/W8 ✅（A 口径：G6 OOD 未达 / C4 abstain 降级附录）** → **S7/W9 零重训消融 ✅（6/6，本地 CPU）** → **S7 GT audit v2 ✅（N_eff=6，G7 insufficient）** → **W10 代码就绪（3 项，等 GPU 服务器）** → S8 ⬜

### S5（已完成）
- 数据：train 12,000 / val 1,500 / test 1,500（12.84 GB），SHA256 全绿；前端缓存 24 shard。
- 模型：3.53 M depthwise-separable U-Net；P0 加 **0.009 M dead 分割头**（val clean IoU 0.9846、sliver miss 1.83%；SF-val 0.9838 / 0.51%）。
- 5 seeds（0–4，100 epoch）：公平单帧档 **Ours/Biharm_SF = 0.273 ± 0.013**；sliver **0.263 ± 0.012**、arc **0.250 ± 0.012**、blob **0.321 ± 0.012**；SF 零样本 **−3.4% ± 6.4%**；G5-c vs h_input **82.5% ± 0.5%**。**5/5 全过。**
- 最优 seed 4：Ours 0.2020 rad，Ours/Biharm_SF **0.2559**。
- Oracle 参考档（GT-dead Biharm/ORR）总体 **1.537 ± 0.073**，按约定只作不可达上界，不作门禁。
- 报告：`probes/results/w5_5seeds_report.md`；图：`probes/results/w5_5seeds/w5_5seeds_ratios.png`；最优参数图：`probes/results/w5_best_seed4_figs/`。

### W8（S6 收口完成，A 口径；C4 降级附录）
- 两级 Conformal **P-only**（β=未覆盖率，β=0.05 → 95% 场景内覆盖）：5-seed **0.9344/0.8857/0.7841**（α=0.05/0.10/0.20）；P-only 自检 **0.9564/0.9133/0.8056**；OOD 掉 1.4–1.6 pt，G6 OOD 未达，如实报。
- 旧 all-class 0.9318/0.8760/0.7691 只作附录；像素级基线 α=0.05 仅 **0.715 ± 0.013**。
- SF：零样本全局 0.756；目标域 sf_val 重标定后 **0.966 ± 0.005**（P=0.975/U=0.956；γ=0.20 分组，n_P=162 / n_U=448）。
- C4 P-only RCPS：α=0.05/.10 均 **abstain**（Bentkus UCB 0.1167/0.1150；CP 0.4238）；**降级为附录 negative/limitation**；pooled 70.6%/3.83% 只作诊断。
- 报告：`probes/results/w8_summary.md`、`w8_s6_main_table.md`；图：`probes/results/w8_conformal/`、`w8_rcps/`。
- 预测缓存：**compact dead-only 5-seed** `data/preds_compact/`（141 MB）；full 20 个 `.npz` 已下载本地 `data/preds_full/`（3,014 MB）。

### W9（S7 零重训消融 6 项，2026-11-20 完成）
- 代码：`code/w9_zero_retrain.py`（1–5）、`code/w9_roi.py`（6）；报告：`probes/results/w9_zero_retrain.md`；汇总 JSON：`w9_zero_retrain_all.json`。
- 全部本地 CPU，无 GPU/无重训；**官方 C3 float32 主表未动**（0.9344/0.8857/0.7841）。
- 六个结论：① 去 Mondrian → P 过覆盖 0.932→0.959、宽度 +13.6%，不能修 OOD；② 去 RCPS → P 经验风险 3.77%、scene-any 32.2%，但官方 Bentkus/CP α=.05/.10 仍 abstain（负对照，不写 C4 有效）；③ 像素级 α=.05 **0.7147 ± 0.0132**，复现 0.715 崩盘；④ 去 ρ 门控 → U 混入，α=.10 pooled 风险 **10.02%**、U 风险 10.58% vs P 3.58%；⑤ κ̂ fixed vs MLQ → P 集 193→167 / 189→171（**P 成员变化 15.5%/12.7%**），C3 覆盖不变，C1 CP95 2.43/2.49→2.81/2.74%；⑥ Ω_roi 单帧估计 → 中位 IoU 0.976、mean 0.828、fallback 52.3%、dead 保留 0.828；P 场景 193→147 / 189→135，C1 CP95 4.22/4.59%，39–40% 场景因 dead 触边界判 U。
- 口径：κ̂ “SG”只是 fixed-kernel proxy，不是真 SG7；Ω_roi 是消融，不进官方主表；所有阈值在看结果前固定，未用 test 回调。

### GT audit（S7 真实 11 场景，2026-11-20 v2 锁定）
- 代码：`code/w9_gt_audit.py`；报告：`probes/results/w9_gt_audit.md`；发现/决定：`w9_gt_audit_findings.md`；JSON：`w9_gt_audit.json`。
- **6 PASS**：scene_01/04/05、Train（偏差相位+平面）、实测2（canonical）、实验4。
- **3 GT-insufficient**：scene_02/03/06（旧 `skimage.unwrap_phase` 在一个大连通域上差 2π；badmask 同源不对齐）→ 不按连通域自动对齐、不用质量导向解包当 GT，修好 badmask 再 re-audit。
- **1 Tier A FAIL**：实测1（mat 偏差+平面 median 0.116 / P95 0.506 vs 0.1/0.3）→ 只作定性，不进 G7 像素统计；不降 Tier B、不放宽阈值。
- **1 Tier B FAIL**：实验3（4 步模型残差 0.132 > 0.10）。
- **N_eff = 11 − 4 − 1 = 6 < 10 → G7 = insufficient**（不是 fail/pass）；“P 但 GT 不可评”单列 `P-class, GT-insufficient`，不进 P_pass 分子/分母。
- **真实 C1 gate（`w9_real_c1.py`，CPU，无模型）：11/11 U，n_P=0**。真实遮挡 `vinf` 88–654（合成 P 小内孔 ~10–20），`rho` 2.5–44；GT-badmask 诊断 6/6 仍 U，GT/紧 ROI 诊断 11/11 仍 U（最小 rho 2.32）。**G7 的 binding 约束变成 `n_P≥5`，不是 N_eff**；即使 GT audit 全过也 insufficient。真实只做定性/点误差。
- re-audit 预注册（相位 + mask 双门槛）：见 `w9_gt_audit_findings.md` §3；报告：`probes/results/w9_real_c1.md`。

### W10（需重训消融，已在服务器训练）
- **只做 3 项**（用户 2026-11-20 决定）：去 phasor（7→3 通道：I/roi/M）、去 L_fc（权重 0）、hard-split 1:1 上采样 vs 主表纯混合。
- 代码：`w5_train.py --no-phasor / --no-fc / --hard-split --hard-ratio 0.5`；`w5_model.make_input(use_phasor=False)` + `W5UNet(cin=3)`；`run_w10_ablations.sh`；评测端已加 `--no-phasor`（`w5_eval.py`、`w8_export_preds.py`、`w8_conformal.py`、`w8_rcps.py`）。
- 审计：`probes/results/w10_code_audit.md`；`log T̂` 当前模型没有这个头，**不硬映射**；尺度增广 S5 配方本就没有，**not applicable**。
- **100 epoch（30/30/40），与 S5 主表同配方；1 seed（seed 0）；~9 GPU·h**；已在 AutoDL 5090 顺序训练（no-phasor 先）。G8 与 5-seed 主表同向。

### 背景（S4/SF，历史）
- W4 主口径（γ*=0.75，5% 风险）：val 认证 25.7% / test 25.2%，CP95 3.23% / 4.05%；严格 γ=1 0/109。
- SynthFringe 唯一场景 450/450；严格协议 ε=0.05 无解，唯一合法 SF-test 20% 证书在 γ=0.20（162/450，CP 15.3%，block 15 CP 32.3%）→ SF 只作 G5-b 误差评测。
- 口径文档：`doc/题目2_定义与实施前置.md` v3.2；S4 主报告 `doc/W4_基线与校准报告.md`；交接 `doc/交接_S4到S5_2026-11-17.md`。
- 大白话版：`doc/给非专业读者的项目说明.md`（2026-11-20 更新到 S6 A 口径 + W9 + GT audit）。

### 下一步
1. S6 收口已完成（A 口径）；交接 `doc/交接_S6到S7_2026-11-20.md`。
2. **S7/W9 零重训 6 项 + 真实 11 场景 GT audit v2 已完成**（2026-11-20）；**N_eff=6 → G7 insufficient**。下一步不是“硬算 G7”，而是：修 scene_02/03/06 的 badmask/相位并 re-audit（相位+mask 双门槛已预注册）；真实推理仍需 GPU，服务器等用户通知。
3. **W10 只做 3 项**（用户 2026-11-20 决定）：去 phasor（7→3 通道）、去 L_fc、hard-split 1:1 上采样 vs 主表纯混合。`log T̂` 更名/待 code audit；尺度增广按“S5 训练配方本就没有”作废，不硬映射。
4. G7 口径：**empirical acceptance rule，不是 C4 证书**；`handled ≥ ceil(8·N_eff/11)`、`n_P ≥ 5`、`P_pass ≥ ceil(0.8·n_P)`、`N_eff < 10 → insufficient`；P 空接受集 = abstain ≠ pass。
5. 文档同步：标题 “Certified Recoverability and Distribution-Free Intervals...”；C4 只作附录 negative。



## 目录

```
2\P2\
  README.md              本文件（唯一索引）
  doc\
    给非专业读者的项目说明.md        ★ 大白话版：每一步在做什么、为什么
    W4_基线与校准报告.md              ★★★ S4 主报告（基线表 + 风险曲线 + 门禁）
    交接_S6到S7_2026-11-20.md          ★★★ 当前交接：S6 收口（A 口径）+ S7/W9 入口
    交接_S5到S6_2026-11-20.md          ★★ S5→S6 历史交接
    交接_S4到S5_2026-11-17.md          ★ S4→S5 历史交接
    启动S4_新会话开局包.md            ★ S4 开局包（S4 已完成，留作历史）
    阶段划分与交接规范.md             ★★ 阶段 S0-S8 + 出口门禁 + 交接文件规范
    交接_S3到S4_2026-11-16.md         S3 交接（历史）
    题目2_定义与实施前置.md            ★★ 唯一口径（v3.1，只读；变更须走待变更条目）
    题目2_一区方案_v2.md              ★ 主方案（v2.5，执行依据）
    理论_W1_最优恢复证书.md             ★ W1 理论稿（定理 1–6 + 证明 + 验证）
    探针A_B_结果.md                   δ_g(d) 曲线 + 真实数据普查
    W2_D1_状态与阻塞.md               W2 D1–D2 记录（含 mm 冻结裁定）
    W3_pilot_G2报告.md                W3 pilot 的六族/三分类/背景配额门禁
    题目2_可行性复核与数据方案_v1.md    证据记录（为什么这么改）
  probes\                复核探针（可复现，results/ 存结果）
    physics.py           相位统计 / 渲染合法性 / 哪种退化毁相位
    envelope.py          边界信息包络 vs 孔径 → results/envelope.npy
    realdata.py          6 个实测场景 + SynthFringe 解码与普查 + 标定
    theory_check.py      圆盘 κd²/16 界的 soundness（60/60）
    theory_c1.py         C⁰ vs C¹ 常数（1.4426 = 1/ln2）+ Talenti
    theory_sliver.py     宽度律（7 形状，零违例）
    w1_verify.py         V6 可达性 / V7 闭式界 / V7b 收敛 / V8 κ̂ 校准
    make_fig.py          八面板图
    results\
      w9_zero_retrain.md          ★ W9 六项消融报告（当前）
      w9_zero_retrain_all.json    W9 汇总 JSON
      w8_s6_main_table.md         S6 C1–C4 主表
  code\                  生成器 / 基线 / 校准 / SF 导出与校验（w3_gen.py, w4_calibrate.py, w4_decide.py, w4_sf_recal.py, w3_gt_blocks.py, w3_export_sf.py, verify_w3.py, verify_sf.py）
    w9_zero_retrain.py    W9 第 1–5 项（torch-free，compact）
    w9_roi.py             W9 第 6 项（单帧 ROI 估计 + 代价）
    w8_conformal.py / w8_rcps.py   已改 torch-lazy，CPU 可导入校准数学
  data\                  train/val/test 分片 + manifest（SHA256 冻结，12.84 GB）；sf_val/sf_test 各 690（450 vinf + 240 误差评测）
```

## W4 结论（S4 完成，2026-11-17）

- **主口径**：`γ*=0.75`（认证集 + 5% 风险），val 认证 **25.7%**、test **25.2%**；认证集内超界 **1.04% / 1.59%**（CP95 **3.23% / 4.05%**）。
- **严格对照**：`γ=1` 时 0/109、0/114（CP 2.71% / 2.59%）；`C_5pt_max` 最大 **0.332**（0/193 >1）→ 定理层面 sound。
- **基线**：Biharm 总体 `Rel-ORR=0.876`，强于 ORR；ORR 只在 sliver 最优（Biharm 1.527）。Identity 2π 失败 60.4%（门禁 >25%）。
- **门禁**：G4 零违例（CP 口径）✅；`n_cert≥1000`（train 1,364）✅；认证率目标由 35–55% 修订为 `ε=0.05` 下 ≥20%。
- **SF 外部基准（v3.2 唯一场景版）**：SF-val/SF-test 各 450 个含 `vinf` 样本（每族 150，零复用）；γ=1 均为 0/450。重标定后唯一 CP 合法的 test 20% 证书在保守 γ=0.20（162/450，CP 15.3%），但 **block 15 CP 32.3%**；严格协议下 ε=0.10/0.20 的 val 最优 γ 在 test 均不合法（15.3% > 10%、26.4% > 20%）。G5-b 只测误差。
- 细节：`doc/W4_基线与校准报告.md`；交接：`doc/交接_S4到S5_2026-11-17.md`；SF 重标定：`probes/results/w4_sf_recal.md`。

## S5 开工前收尾（2026-11-17，指令 1–3）

| 指令 | 状态 | 产物 / 验收 |
|---|---|---|
| 1 SF 扩样 → 唯一场景 450 | ✅ | `data/sf_val`、`data/sf_test` 各 690 样本（450 vinf + 240 误差评测），SHA256 全绿；`verify_sf.py` 通过；重跑 `w4_sf_recal.py` |
| 2 CP 失效行打标记 | ✅ | `W4_基线与校准报告.md` §5（MATLAB ε=0.20 test 行）、`题目2_定义与实施前置.md` v3.2 §4bis.1、`w4_sf_recal.md`（所有 CP>ε 行） |
| 3 C4 措辞红线 | ✅ | `题目2_一区方案_v2.md` §7.1：禁止/允许措辞 + 必须同框的三个数 + block 15 / oracle 归因 |

> 服务器仍未定（用户：需要传数据时再说，届时提供 SSH）。S5 训练暂不阻塞代码阶段。

## W1 结论（已完成，G1 全过）

- **定理链**：引理 1（比较引理）→ 定理 1（minimax 最优 + 可达）→ 推论 1（证书）/ 定理 6（不可能性）；定理 2（圆盘 `d²/16`）、定理 3（Talenti `d_eq²/16`）、**定理 4（宽度律 `w_min²/8`）**、定理 5（C¹ 只值 `1/ln2 = 1.4427`）。
- **数值**：引理零违例 60/60、7/7、24/24；极值面可达性 **1.0000**；离散收敛 1/R；`κ̂` 在 γ=1 时**零违例**（认证 17%）。
- **两个坑已固定**：① 离散约定 `effective size = extent + 2`（否则假报违反）；② `κ̂_SG/κ_true` 的 P10 = **0.35**（低估 3 倍）→ `γ` 必须在校准集上取，不能取 1 拍脑袋。

## 主方案要点（v2.3）

- **题目**：Certified Recoverability and Risk-Controlled Intervals for Single-Shot FPP
- **理论核**：死区相位的最优恢复用**圣维南扭转函数** —— 任意死区 `Ω`，任何可行表面与调和延拓偏差 ≤ `κ‖v_Ω‖_∞`；圆盘退化为 `κd²/16`；Talenti 给兜底（但对细长/环状形状松 15–32×，只作粗筛）。
- **可证伪判据**：`ρ = κ‖v_Ω‖_∞/φ_tol ≤ 1` ⟺ 被证明可恢复；否则**不存在**任何只依赖边界的估计器能保证容差（定理 6）。
- **"宽度律"的正确表述**：由**扭转宽度** `w_eff=√(8‖v‖)` 决定，**不是凸包宽度**（香蕉形阴影凸包界松 275×）。
- **三个前提已形式化**：引理 2（边界噪声 `δ_g`）、引理 3（拓扑：贯穿型只给相对相位）、`κ` 为像平面量。
- **两层测量**：Tier-1 证书 + Tier-2 **两级 conformal**（`P_场景(覆盖率≥1−β) ≥ 1−α`）+ RCPS（λ 网格 + WSR + running-min）。
- **网络**：`ψ = h_input + M_d·Δψ`，`M_d` 为 **C¹ smoothstep**；`Δψ` 的两条职责 = 补起伏 + 抵消边界噪声引起的调和基底偏差。
- **GT 与 mm**：**级联解包 `1→3→8→64`**（不许直跳）+ `W_p` 定标 + 相机光线/投影平面求交；遮挡用**免 Mesh 的三维着色**生成。
- **数据**：三级标签 Synth-Occlusion / Synthetic-Cutout / Real-Physical-Shadow；死区连通域 ≥3000。
- **周期 12 周**；消融拆"零重训 6 项 / 需重训 5 项"。
- ⚠️ **分区**：中科院 1 区当前不可达（无台架无标定）；现实上限 IEEE TIM（JCR Q1 / 仪器仪表 T1）。见方案 §13.1。

## 探针（全部可复现）

| 脚本 | 回答什么 |
|---|---|
| `physics.py` | 相位梯度统计、渲染合法性、哪种退化毁相位 |
| `envelope.py` | 边界信息包络 vs 孔径（Identity / 相位修补 / ORR） |
| `realdata.py` | 6 个实测场景 + SynthFringe 解码与普查 + 标定 |
| `theory_check.py` | 圆盘上 `κd²/16` 界的 soundness（60/60 零违例） |
| `theory_c1.py` | C⁰ vs C¹ 极值常数（1.4426 = 1/ln2）+ Talenti 验证 |
| `theory_sliver.py` | **宽度律**：7 种形状，细条比 `d_eq` 紧 2.9–10.6×，零违例 |
| `w1_verify.py` | V6 可达性 1.0000 / V7 闭式界 24/24 / V7b 收敛 / V8 κ̂ 校准 |
| `w1b_stress.py` | **T1 弯曲死区（凸包界松 275×）/ T2 边界噪声 / T3 贯穿型死区** |
| `make_fig.py` | 八面板复核图 |


## 数据资产（只读，勿改）

| 路径 | 内容 | 用途 |
|---|---|---|
| `F:\paper\SynthFringe` | 4 密度（1/3/8/64 周期）× 3600 场景 × 12/15/16 步 + **完整相机/投影仪标定** | 主力基准：精确 GT 相位 + mm 几何 + 载波轴 |
| `F:\paper\original_phase` | 10,000 个 256² 相位，max∈{10..190} | 受控 κ / 孔径扫描（相变图的合成支线） |
| `F:\paper\2\model\real\captures` | 6 场景，4 步 GT，256² | 真实相位级验证 |
| `F:\paper\2\extra`、`2\phase` | 真实图，无配准 GT | 定性 / OOD |
| `F:\paper\1\phase\Train` | 15 步 GT，480×640，39.1% 有效 | 真实定性 + 相位级定量 |
| `F:\paper\1\phase\实测1/2、实验3/4` | 4–5 步实测 | 真实相位级定量（共 11 场景；实验3/4 独立采集，文件名误导） |

## 命令

```powershell
# 数据校验（SHA256 / 分片 / 分类计数）
python F:\paper\2\P2\code\verify_w3.py

# W4（S4）：逐 split 评估 → 校准汇总 → 决策表 → 64 样本回归
python F:\paper\2\P2\code\w4_calibrate.py eval --split val --delta-g 0.13
python F:\paper\2\P2\code\w4_calibrate.py report
python F:\paper\2\P2\code\w4_decide.py
python F:\paper\2\P2\code\w4_calibrate.py regression

# SynthFringe 外部基准（S5 前置）：blocks 12..17 GT → SF-val / SF-test（唯一场景 450 vinf/ split）
python F:\paper\2\P2\code\w3_gt_blocks.py
python F:\paper\2\P2\code\w3_export_sf.py --split val  --n-vinf 150 --n-other 80
python F:\paper\2\P2\code\w3_export_sf.py --split test --n-vinf 150 --n-other 80
python F:\paper\2\P2\code\verify_sf.py
python F:\paper\2\P2\code\w4_sf_recal.py

# 探针（S1/S2）
python F:\paper\2\P2\probes\physics.py
python F:\paper\2\P2\probes\envelope.py
python F:\paper\2\P2\probes\realdata.py
python F:\paper\2\P2\probes\make_fig.py
```

## 门禁（G0–G4 修订版，见 doc §5.8 / 交接 S4）

| 门 | 判据 |
|---|---|
| V1 渲染合法 | 干净条纹 FTP+unwrap 在有效区 2π 失败 <0.5%、RMSE<0.5 rad |
| V4 Identity | mask 内 2π 失败 >5%（hard split >25%） |
| V5 包络 | d=30–60 px 时包络 RMSE 落在 0.2–0.8 rad |
| G2 校准 | 类条件 PICP 与 1−α 偏差 <5% |
| **G4 证书（W4，已完成）** | 认证集内超界 CP95 <5%（val 3.23% / test 4.05%）；`n_cert≥1000`（train 1,364）；Identity 2π 失败 60.4% >25% ✅ |
| G5 网络（S5） | 总体 `Ours/Biharm≤0.95`；分形状 sliver `Ours/ORR≤1.05`、arc/blob `Ours/Biharm≤1.05`；SF 掉点≤20% |
| C 理论门（W6） | 二维界 vs 实测包络 Spearman>0.85 且常数在不同 d/κ 档稳定（1.0±0.25） |
