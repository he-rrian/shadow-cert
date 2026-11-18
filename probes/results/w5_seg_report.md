# P0 轻量 dead 分割头结果（确认用）

日期：2026-11-18  
模型：3 层 depthwise-separable U-Net，输入 `[I/255, roi, m_hilb]`，输出 dead_prob  
参数量：**0.009 M**（9k；比计划的 0.3M 更小，已过线）  
训练：8 epoch，batch 32，AdamW 1e-3，BCE(pos_weight=5)+0.5 Dice；**约 4 min**（RTX 5090）  
后处理：`prob > 0.5`，再 **1 px dilation**（对应方案里的 “dead_pred ⊕ 1–2 px”）

## 结果

| split | clean IoU | sliver miss | arc miss | blob miss | edge/cut/degraded miss |
|---|---:|---:|---:|---:|---:|
| **MATLAB val** | **0.9846** | **1.83%** | 0.79% | 0.08% | 0.46% / 0.31% / 0.00% |
| **SF val（零样本）** | **0.9838** | **0.51%** | 0.04% | 0.00% | 0.56% / 0.77% / 0.00% |

门槛：clean IoU ≥ 0.85，sliver miss ≤ 5% → **两项都过**。

> 对照：不加 1 px dilation 时 val sliver miss 6.4%（未过 5%）；1 px dilation 后 1.83%，clean IoU 仍 0.985。SF-val 零样本也过，说明不是靠 MATLAB 纹理记忆。

## 定性

![seg](seg_qual.png)

上排 MATLAB val：GT dead 是两条细缝，预测的 dead_prob 只亮在细缝上，1 px dilation 后覆盖细缝；下排 SF-val 同样只覆盖两条斜缝，没有把整块多边形吞掉。C 版本里“半张物体亮黄色”的过覆盖在这版消失。

## 下一步（等确认后执行）

1. 用 `dead_pred ⊕ 1 px` 重算 `h_input`（harmonic fill 只在紧凑 dead 上做），替代 C 的 over-cover h；
2. 解开 `Ω_clean` 的 `M_fix=0` 硬锁：在 `Ω_roi` 全域预测 Δψ，dead 内权重 ×3、3 px 过渡带 ×2、其余 ×1，配合观测域 `L_fc`；
3. cut 样本只对主连通域算绝对相位 loss（或每个连通域去 median piston），消除 1.0 rad 假平台；
4. G5 基线拆两档：单帧公平版 `Harmonic_SF/Biharm_SF/h_input`，Oracle 版 `ORR/Biharm_oracle`；G5-c 改为 `||Ours-ORR|| < ||h_input-ORR||`；
5. 100–120 epoch，先 1 seed 过门禁。
