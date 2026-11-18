# W8 预检报告（seed 0, last.pth）

日期：2026-11-18  
checkpoint：`runs/w5_p0_seed0/last.pth`（stage 3，epoch 100）  
数据：MATLAB val 1,500 场景 / 2,027,191 个 GT-dead 像素

---

## 1. σ̂ 头：未塌缩，场景级排序强

`log_sigma` 是 **log variance**；下表已换算成标准差 `σ = exp(0.5·logσ)`。

| 指标 | 数值 | 判定 |
|---|---:|---|
| Spearman(|Ours−GT|, σ̂) **pooled（全部 dead 像素）** | **0.476** | ✅ ≥0.45 |
| Spearman **场景级**（每场景 median error vs median σ̂） | **0.692** | ✅ 强 |
| Spearman **per-sample 像素级（中位）** | 0.323 | ⚠️ 偏弱（见下） |

σ̂ 中位数（rad）：

| 形状 | σ̂ median |
|---|---:|
| sliver | 0.0574 |
| arc | 0.0565 |
| blob | **0.0841** |
| edge | 0.1020 |
| cut | 0.0842 |

- **blob > sliver/arc**，符合“blob/U 的不确定度高于 sliver/R”的预期；edge 最高，也合理。
- **没有塌缩**：σ̂ 量级 0.06–0.10 rad，与 Ours 的 P95 0.16–0.31 rad 同一量级；不是常数、不是 0。
- **注意**：W8 的 `S_i` 先对每个场景取 `(1−β)` 分位，再做场景级 conformal；**场景级 Spearman 0.692** 才是 W8 直接依赖的排序。像素级 per-sample 0.323 偏弱，说明单像素 σ̂ 的排序有噪声，但场景汇总后很强；如果 W8 后续直接用像素级 σ̂ 做逐像素覆盖，需要再校准（当前设计是场景级，不受影响）。

## 2. cls 辅助头：未塌缩

| 类 | recall |
|---|---:|
| **P** | **0.741** ✅ ≥0.5 |
| U | 0.948 |
| R | 0.972 |
| 总体 acc | 0.925 |

混淆矩阵（行=GT P/U/R，列=Pred P/U/R）：

```
[[143,  50,   0],
 [ 53,1002,   2],
 [  4,   3, 243]]
```

P 类 193 个中 143 个召回，50 个被判 U；没有全塌到 U/R。

## 3. checkpoint 保存规则（已改代码）

- `best_stage3.pth` **只在 `stage == 3`（epoch 61–100）保存**，且按 val phase 最优；不会误存 stage 1/2 的随机 σ̂ 头。
- `last.pth` / `stage3.pth` 现在带 `epoch`、`stage=3` 元数据。
- seed 0 的 `best_stage3.pth` 由 `last.pth`（stage 3, epoch 100）复制，供 5-seed 汇总统一使用。
- 4 个新 seed 的训练脚本已使用新规则；跑完检查 `best_stage3.pth` 的 `stage` 字段应为 3。

---

## 结论

**W8 预检绿（按场景级口径）**：
- σ̂ 未塌缩，pooled 0.476 / 场景级 0.692，形状单调；
- P recall 0.741；
- checkpoint 规则已锁死 stage 3。

唯一保留项：**像素级 per-sample Spearman 0.323**；W8 的场景级 `S_i` 不受影响，但如果后续要做像素级 coverage，需要单独校准 σ̂。

已按此放行：seeds 1–4 于 19:45 起跑，预计 **06:45 左右**跑完；之后自动跑 5-seed 评估与 mean±std 汇总（`/root/AGG_DONE`）。
