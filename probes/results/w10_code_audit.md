# W10 需重训消融 — code audit 与可执行定义

**日期**：2026-11-20 ｜ **状态**：3 项代码已就绪；**已在服务器执行（no-phasor 先跑，100 epoch 配方）**

**用户 2026-11-20 决定**：W10 只做 **3 项**；`log T̂` 不硬映射（等 code audit）；尺度增广按“S5 训练配方本就没有”**作废**。

---

## 1. 三项与代码对应

| # | 消融 | 定义 | 代码 | 状态 |
|---|---|---|---|---|
| 1 | **去 phasor** | 输入 7 通道 → 3 通道：`I`, `roi`, `M_fix`；不喂 `sin/cos(h−carrier)` 与 `sin/cos(carrier)` | `w5_model.make_input(..., use_phasor=False)`；`W5UNet(cin=3)`；`w5_train.py --no-phasor` | ✅ 可跑 |
| 2 | **去 L_fc** | 三阶段 `L_fc` 权重全部 = 0（S5 主表是 0.5） | `w5_train.py --no-fc`（stage 2/3 `fc=0`） | ✅ 可跑 |
| 3 | **hard-split 1:1 vs 混合** | 主表 = 纯混合（`shuffle=True`，无 sampler）；消融 = `ρ≥1` 场景上采样到每个 epoch 的 50% | `w5_train.py --hard-split --hard-ratio 0.5`；`hard_split_weights()` | ✅ 可跑 |

**方向写死**：主表是**混合采样**；W10-3 的消融臂是**ρ≥1 1:1 上采样**。报告不能写成“主表是 hard split”。

---

## 2. 两项“不硬映射”的审计结论

### 2.1 `log T̂`：当前模型没有这个头
`w5_model.split_output` 的实际输出是：
```
dpsi, log_sigma, A_logit, B_logit, logits
```
没有 `log T` / `log T̂`。`log_sigma` 是**不确定度**（C3/C4 的 σ̂ 来源），不是 log modulation/period。
- **不映射**成“去 σ/NLL 头”（那是另一条有价值的消融，但现在不冒充 `log T̂`）。
- 处理：W10 暂不做 `log T̂`；等 W9 后的 code audit 再决定“改名 / 删除 / 替换”。本文件就是这次 audit 的记录。

### 2.2 尺度增广：S5 配方里没有
`w5_train.py` 的 S5 正式训练 `augment=False`；代码里唯一的增广是 vertical flip（且被禁用）。方案的 `0.6–1.6×`/`0.55–1.8×` 缩放从未实现。
- 报告写：**“Scale augmentation was not part of the S5 training recipe; therefore this ablation is not applicable.”**
- 不把“按 T 过滤到 9–12 px”冒充“去尺度增广”（那是窄 T 训练，是另一件事）。

---

## 3. 运行

```bash
# on the GPU server, from 2/P2
bash run_w10_ablations.sh
```

或逐条：
```bash
python code/w5_train.py --seed 0 --out runs/w10_nophasor_seed0  --no-phasor
python code/w5_train.py --seed 0 --out runs/w10_nofc_seed0       --no-fc
python code/w5_train.py --seed 0 --out runs/w10_hardsplit_seed0  --hard-split --hard-ratio 0.5
```

- 每项 100 epoch（30/30/40），batch 16，与 S5 主表同配方；1 seed（seed 0）。
- 预计 **~9 GPU·h**（5090 实测约 3 h/项；原 36 GPU·h 估计基于 130 epoch/5 项）。
- 输出：`runs/w10_*_seed0/best_stage3.pth`、`train_log.json`。

---

## 4. 评测（G8）

训练后需要与 5-seed 主表同向比较，至少报：

| 指标 | 脚本 | 说明 |
|---|---|---|
| G5 公平档（Ours/Biharm_SF、sliver/arc/blob） | `w5_eval.py --no-phasor`（仅 no-phasor 需要） | **公平比值 = `gate_val.ours / gate_val.bih_sf`（val）**；主表 5-seed **0.2734 ± 0.0130**，seed0 单点 **0.2652** |
| C3 P-only 覆盖（α=.05/.10/.20） | `w8_conformal.py --no-phasor` | 主表 0.9344/0.8857/0.7841；seed0 单点 0.931/0.884/0.799；只作消融，不改官方主表 |
| C4 附录风险（如仍 abstain） | `w8_rcps.py --no-phasor` | 不把 C4 写成正面 |
| 训练日志/参数量 | `train_log.json` | 记录 cin/参数量/FPS |

**一键评测**：
```bash
# on the GPU server, from 2/P2, after training
bash run_w10_eval.sh
```
输出：`probes/results/w10_{nophasor,nofc,hardsplit}_g5/`、`w10_*_c3.json`、`w10_*_rcps.json`。
`run_w10_waiter.sh` 已排队：等 `runs/w10_all.log` 出现 `done.` 后自动执行 `run_w10_eval.sh`，写到 `runs/w10_eval.log`。

**G8 判据**：消融结论与 5-seed 主表**同向**（不是“消融必须更差才合格”，而是“结论方向一致”）。方向写进报告：
- 去 phasor：预期点误差↑/覆盖↓；
- 去 L_fc：预期边缘/纹理区变差；
- **hard-split 1:1 是反转项**：主表本身就是纯混合（`shuffle=True`，无 sampler）；消融是 `ρ≥1` 1:1 上采样。若上采样版没有显著更好，结论是 **“主表混合采样足够”**，不是“hard split 更好”。报告不能写反方向。

---

## 5. 当前状态

- [x] `w5_model.make_input(use_phasor=False)` + `W5UNet(cin=3)`
- [x] `w5_train.py --no-phasor / --no-fc / --hard-split`
- [x] `w5_eval.py --no-phasor`、`w8_export_preds.py --no-phasor`、`w8_conformal.py --no-phasor`、`w8_rcps.py --no-phasor`
- [x] `run_w10_ablations.sh`（100 epoch 30/30/40，与 S5 主表同配方）
- [x] `run_w10_eval.sh` + `run_w10_waiter.sh`（训练完自动 G8）
- [x] 服务器执行 3 项重训（100 epoch，与 S5 主表同配方；~9 GPU·h）
- [x] G8 表 + 与主表同向结论：`probes/results/w10_g8_report.md`
  - 去 phasor **0.3766（+37.7%）** → 必要；
  - 去 L_fc **0.2753（+0.7%）**，SF −11.9% → 弱/跨域；
  - hard-split **0.2758（+0.9%）** → 无收益，主表混合足够。
