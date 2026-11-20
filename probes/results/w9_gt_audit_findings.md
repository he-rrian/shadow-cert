# W9 真实 11 场景 GT audit v1 — 发现与待定口径

**日期**：2026-11-20 ｜ **代码**：`code/w9_gt_audit.py` ｜ **结果**：`w9_gt_audit.json` / `w9_gt_audit.md`

**状态**：11 场景已全部过一遍；按**预注册阈值**严格判，**5 通过、6 失败/待定**。这不是“GT 都坏了”，而是三类口径问题必须先定，否则 G7 会在错误的分母上算。

---

## 0. 严格审计结果（预注册阈值）

| 场景 | Tier | 交叉验证 / 自洽 | 关键数字 | 判定 |
|---|---|---|---|---|
| scene_01 | A | gt_phase.npy | median 0.0020, P95 0.0062 | PASS |
| scene_02 | A | gt_phase.npy | median 0.0104, **P95 12.57**；45.2% 像素差 >π | **FAIL** |
| scene_03 | A | gt_phase.npy | median 0.0132, **P95 12.55**；48.2% 像素差 >π | **FAIL** |
| scene_04 | A | gt_phase.npy | median 0.0020, P95 0.0184 | PASS |
| scene_05 | A | gt_phase.npy | median 0.0062, P95 0.0211 | PASS |
| scene_06 | A | gt_phase.npy | median 0.0093, **P95 6.28**；42.0% 像素差 >π | **FAIL** |
| Train | A | gt15_rad + mask | 偏差相位+平面：median 0.0016, P95 0.0048 | PASS（需 plane 口径） |
| 实测1 | A | fourstep_purphase.mat | 偏差相位+平面：**median 0.116, P95 0.506** | **FAIL**（略超 0.1/0.3） |
| 实测2 | B | self-consistency | 1st≈5th 0.543；model 0.0816；dir BAD | **FAIL**（5th/方向） |
| 实验3 | B | self-consistency | model 0.132；dir BAD | **FAIL**（model 残差） |
| 实验4 | B | self-consistency | model 0.068；dir ok | PASS |

**严格后果**：Tier B 失败 2 个 → N_eff=9 → **G7 = insufficient**；即使 N_eff=10，Tier A 的 scene_02/03/06 + 实测1 也需要先定口径，否则 G7 的 8/11 分母没有意义。

---

## 1. 三类已确认的口径问题（有数字，不是猜）

### 1.1 Train 的 GT 是“偏差相位”，不是总相位
- `gt15_rad.npy` 梯度中位 dx = 0.0003 rad/px；15 步总相位梯度 dx = -0.63 rad/px。
- 拟合 `phi_total ≈ -gt15_rad + a + bx·x + by·y`：**a=-196.355, bx=0.63025, by=2.62e-5**；残差 **median 0.0016、P95 0.0048 rad**。
- 结论：Train GT 是**去载频的偏差相位**；审计必须允许“偏差相位 + 载频平面”，否则 median 会假报 76 rad。当前 audit 已用 `deviation+plane`，Train 通过。

### 1.2 实测1 的 fourstep mat 也是偏差相位，但平面后仍略超阈值
- 符号 -1 + 平面：`median 0.116, P95 0.506, max 64.45 rad`；梯度相关约 -0.12/-0.07（identity）。
- 试过 8 种转置/翻转，没有更好的方向；它不是总相位，而是外部 four-step 方法的偏差相位输出。
- 预注册 Tier A real 阈值是 median 0.1 / P95 0.3；**0.116/0.506 超线**。这是“略超”，不是数量级错误；但按规则不能算 pass。

### 1.3 scene_02/03/06 是“解包分支”差 2π，不是平滑相位误差
- `gt_phase.npy` 由旧脚本 `make_single_shot.py` 的 `skimage.restoration.unwrap_phase` 生成（第 387 行）；本次 audit 用质量导向解包。
- 全局 offset 对齐后 median 只有 0.01 rad，但 **42–48% 的像素差 >π**；最大差值 163 rad（多圈）。
- 差 2π 的大连通域：scene_02 左侧 110×256（34% valid）、scene_03 中央 206×198（44%）、scene_06 左侧 112×256（42%）；与 `badmask_single` 重叠只有 1.6%/6.2%/0%（**不是遮挡区**）。
- 结论：两条解包路径在一个大连通域上选了不同的 2π 分支；这是**不可观测活塞**，不是模型能补的平滑误差。严格按 GT 像素比会判 scene_02/03/06 全错，但错的是分支，不是形状。

### 1.4 实测1/2 的第 5 帧不是 2π 复查
- 实测2 帧间相关：第 5 帧与第 1–4 帧相关都约 0.36，**不是第 1 帧的 2π 重复**；`fringe.m` 也只用 1–4 帧（4 步）。
- 实测1 的 `first5` = 0.679，同样不是 2π 重复。
- 所以“首帧≈第五帧”这一条对这两场景**不适用**（5th 是富裕帧/不同图），不是采集失败。

### 1.5 方向检查是“符号约定”，不是物理失败
- `scene_01/02/05/06` 的 raw `dir` 也可能是 BAD，但 Tier A 用 GT 交叉验证定符号，不受影响。
- Tier B 的 4 步符号可由条纹方向（carrier）确定；当前 raw 检查把“公式符号”当成失败。正确做法是：先按 carrier 镜像到 +x，再翻转 4 步符号使 `median dφ/dx` 与 carrier 同号，然后做方向检查；报告“修正前/后”。

---

## 2. 待你拍板的 5 件事

| # | 问题 | 选项（推荐在前） |
|---|---|---|
| Q1 | Train GT 是偏差相位，是否允许“偏差相位 + 拟合载频平面”作为交叉验证口径？ | **A. 允许，报告平面系数与残差（推荐）**；B. 重建总相位后再比；C. 不用 Train 做像素级 GT |
| Q2 | 实测1 的 mat 平面后 median 0.116 / P95 0.506，超预注册 0.1/0.3；怎么处理？ | **A. 降为 Tier B（自洽套件）或只作定性（推荐）**；B. 保留 Tier A 但如实标 FAIL，不进 G7；C. 把 real 阈值放宽到 0.15/0.6（等于看结果调阈值，不建议） |
| Q3 | scene_02/03/06 的大连通域 2π 分支差，怎么算？ | **A. 按连通域做 2π 对齐（报告“不可观测活塞”，只评形状/覆盖）（推荐）**；B. 判 U/定性，不进 G7；C. 用质量导向解包替换旧 `gt_phase.npy` 作 GT；D. 保持严格 FAIL，G7 insufficient |
| Q4 | 实测1/2 第 5 帧不是 2π 复查，`first≈fifth` 怎么算？ | **A. N/A（5th 未使用，按 fringe.m 4 步）（推荐）**；B. 保留 FAIL（5th 不复现就不信任） |
| Q5 | Tier B 方向检查是否先做 canonicalization（镜像 +x、翻符号匹配 carrier）？ | **A. 是，报告 raw/canonical 两值（推荐）**；B. 保持 raw，方向 BAD 即 FAIL |

**如果 Q1–Q5 按推荐执行**，预期：
- Train PASS；scene_02/03/06 用连通域对齐后可通过（分支不再计错）；
- 实测1 降 Tier B 或定性，不占 Tier A；
- 实测2（5th N/A + canonical）可通过；实验3 model 残差 0.132 仍 FAIL；实验4 PASS；
- Tier B 失败 1 个 → **N_eff=10**；G7 门槛 handled ≥ **8/10**、n_P≥5、P_pass≥ceil(0.8·n_P)。
- Tier A 剩 scene_01–06（分支对齐后）+ Train（5 个场景）可进 G7；G7 的 8/10 仍需要真实推理的 P_pass 来填，不能预支。

**如果 Q3 选 B/D 或 Q2 选 B**：G7 会直接 insufficient 或只剩很少场景；这不是“再调一下”能解决的，必须如实写 limitation。

---

## 3. 复现

```powershell
python code\w9_gt_audit.py
```

脚本里所有阈值在 `TH` 字典，均为看 G7 结果前写死；本 memo 只列待定口径，不改阈值。
