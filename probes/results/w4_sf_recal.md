# SynthFringe 目标域重标定（零重训）

phi_tol=0.314159  delta_g=0.13  eps=0.05  n_min=50
SF-val 240 / SF-test 240 个含 vinf 样本

| gamma | val n_cert | val cert% | val A-cert (CP95) | val B (CP95) |
|---|---|---|---|---|
| 0.50 | 4 | 1.7% | 0/4 = 0.0% (CP 52.7%) | 0/4 = 0.0% (CP 52.7%) |
| 0.75 | 0 | 0.0% | 0/0 = 0.0% (CP nan%) | 0/0 = 0.0% (CP nan%) |
| 1.00 | 0 | 0.0% | 0/0 = 0.0% (CP nan%) | 0/0 = 0.0% (CP nan%) |
| 1.30 | 0 | 0.0% | 0/0 = 0.0% (CP nan%) | 0/0 = 0.0% (CP nan%) |

**没有 gamma 满足 n_cert>=50 且 CP95(A-cert)<=0.05**；SF 上连重标定后的 5% 证书也不存在，只能报误差或更宽的风险档。

## 不同风险档的 SF 可认证性（SF-val 校准，SF-test 一次性）

| eps | gamma*_SF | val n_cert | val cert% | val A-cert (CP95) | test n_cert | test A-cert (CP95) |
|---|---|---|---|---|---|---|
| 0.05 | — | — | — | 无解（n_cert>=50 且 CP95<=0.05） | — | — |
| 0.1 | — | — | — | 无解（n_cert>=50 且 CP95<=0.1） | — | — |
| 0.2 | 0.20 | 102/240 | 42.5% | 11/102 (10.8%, CP 17.2%) | 71/240 (14.1%, CP 22.7%) |