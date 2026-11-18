# 5-seed 汇总（S5 公平单帧档）

日期：2026-11-19  
seeds：0, 1, 2, 3, 4；每个 100 epoch（30/30/40），RTX 5090  
评估：`best_stage3.pth`（seed 0 为 stage 3 的 last），val 750 个 vinf 场景 + SF test 450

---

## 1. 公平单帧档（主门禁）：5/5 全过

| 指标 | 要求 | 5-seed mean ± std | 最差 seed | 判定 |
|---|---:|---:|---:|---|
| 总体 Ours/Biharm_SF | ≤ 0.95 | **0.273 ± 0.013** | 0.290 | ✅ |
| sliver Ours/Biharm_SF | ≤ 1.05 | **0.263 ± 0.012** | 0.280 | ✅ |
| arc Ours/Biharm_SF | ≤ 1.05 | **0.250 ± 0.012** | 0.266 | ✅ |
| blob Ours/Biharm_SF | ≤ 1.05 | **0.321 ± 0.012** | 0.337 | ✅ |
| SF 零样本掉点 | ≤ 20% | **−3.4% ± 6.4%** | +0.96% | ✅ |
| 比 h_input 更近 ORR | ≥ 80% | **82.5% ± 0.5%** | 81.7% | ✅ |

绝对误差（median P95，rad）：

| 指标 | mean ± std |
|---|---:|
| Ours 总体 | **0.216 ± 0.010** |
| Biharm_SF（公平基线） | 0.789（固定） |
| sliver Ours | 0.169 ± 0.007 |
| arc Ours | 0.193 ± 0.009 |
| blob Ours | 0.330 ± 0.012 |
| SF Ours | 0.208 ± 0.014 |

![5seed](w5_5seeds_ratios.png)

## 2. Oracle 参考档（不可达上界，不作主门禁）

| 指标 | 要求 | mean ± std | 判定 |
|---|---:|---:|---|
| 总体 Ours/Biharm_oracle | ≤ 0.95 | **1.537 ± 0.073** | 参考（未打平） |
| sliver Ours/ORR | ≤ 1.05 | **1.808 ± 0.080** | 参考 |
| arc Ours/Biharm_oracle | ≤ 1.05 | **1.165 ± 0.055** | 参考 |
| blob Ours/Biharm_oracle | ≤ 1.05 | **0.435 ± 0.016** | 优于 ORR |

> Oracle 档吃 GT dead / 无纹理误差边界，是物理上界；公平档才是 C2 的部署口径。按用户 2026-11-18 的决定：**公平档为主门禁，Oracle 只作理论参考，不硬刷**。

## 3. 结论

- 5 个 seed 的公平档 **全部通过**，最差 seed 的总体比也只有 0.290（要求 0.95），余量很大。
- 5 个 seed 的 SF 掉点都在 +1% 以内（均值 −3.4%），跨域稳定。
- G5-c 对 h_input 的胜率 **82.5% ± 0.5%**，每个 seed 都 ≥81.7%。
- Oracle 档稳定在 1.44–1.63×，属于“接近但不可达”，与单帧纹理信噪本底一致，不再作为失败项。

## 4. 产物

- 逐 seed gate/rows：`probes/results/w5_5seeds/seed{0..4}/`
- 汇总 JSON：`probes/results/w5_5seeds/summary.json`
- 图：`probes/results/w5_5seeds/w5_5seeds_ratios.png`
- 5 个 checkpoint：服务器 `runs/w5_p0_seed{0..4}/best_stage3.pth`
