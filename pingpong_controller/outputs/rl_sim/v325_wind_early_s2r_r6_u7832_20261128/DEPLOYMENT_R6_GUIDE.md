# V325-wind-early-s2r-r6-u7832 Real-Robot Deployment Guide (`4.5 m/s` + `±15°`)

本目录包含 `V325-wind-early-s2r-r6-u7832`（`best_pareto_w28_u00007832.pkl`）的完整归档模型、576×60s 闭环评测结果、与训练端逐位对齐（`< 1e-6`）的纯 NumPy 推理模块 [`r6_numpy_inference.py`](./r6_numpy_inference.py) 及标准契约 [`deployment_contract.json`](./deployment_contract.json)。

## 与旧 `u7447` (`single5_windfeature120_high_v2`) 的仅 2 处契约差异

> **重要**：请以**独立 profile**（`fplc_gpu0_v325_wind_early_s2r_r6_u7832`）接入机上部署栈，**切勿修改原有 `u7447` 的 `1.6 m/s` / `±5.0°` 契约**，确保两版模型可一键切换对比。

| 参数项 | 旧版 `u7447` (`single5_windfeature120_high_v2`) | 新版 `u7832` (`single5_highwind120_4p5ms_v1`) |
| :--- | :---: | :---: |
| 风速观测上限 & 归一化分母 `obs_wind_max_m_s` | `1.6 m/s` | **`4.5 m/s`**（`obs[114:116] = est_wind_xy / 4.5`） |
| 强风死区特征 `obs[118:120]` 解码风速 | `1.6 * ||obs[114:116]||` | **`4.5 * ||obs[114:116]||`**（死区仍为 `1.25 m/s`，宽度 `0.35 m/s`） |
| 拍面倾角投影上限 `relativeRollPitchLimitDeg` | `[5.0, 5.0]` (`fixedPose=True`) | **`[15.0, 15.0]`** (`fixedPose=True`) |
| 实机硬急停倾角 `realRelativeRollPitchLimitDeg` | `[7.0, 7.0]` | **`[17.0, 17.0]`**（或按实机安全冗余设定，但**必须 `>= 15.0°`**，否则强风倾拍会触发 `fplc_hard_envelope` 急停） |
| 网络拓扑 & 动作空间 | `120 -> 282 -> 282 -> 282 -> 7` (tanh) | **完全相同** |
| 工作空间盒 `positionLowerM / positionUpperM` | `[-0.348,-0.70,1.00] .. [+0.348,-0.28,1.15]` | **完全相同** |

## 机上部署栈最小接入步骤（`/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/...`）

1. **同步模型仓库**：在机上 `simcode12/pingpong_moz1-wind-early` 执行 `git pull origin wind-early`，确认 `pingpong_controller/outputs/rl_sim/v325_wind_early_s2r_r6_u7832_20261128/model.pkl` 存在且 SHA-256 与 `MANIFEST.json` 一致。
2. **参数化风速观测上限 `obs_wind_max_m_s`（默认仍为 `1.6`，保持 `u7447` 零改动）**：
   - 在 `fplc_wind_numpy.py`（或 `fplc_single5_numpy.py`）的 `observed_force_features(...)` 和 `wind_features_encode(...)` 中增加可选参数 `max_wind_m_s: float = 1.6`，将原硬编码的两处 `1.6`（向量限幅 `min(1.0, max_wind_m_s / norm)` 与归一化 `estimate / max_wind_m_s`，以及死区解码 `speed = max_wind_m_s * norm(obs[114:116])`）替换为 `max_wind_m_s`（可直接对照本目录 `r6_numpy_inference.py`）。
   - 在 `fplc_single5_numpy.py` / `fplc_policy.py` 中：若契约的 `actorSchema == "single5_highwind120_4p5ms_v1"` 或 `checkpoint["obs_wind_max_m_s"]` 存在，即传入 `max_wind_m_s = 4.5`（否则保持 `1.6`）。
3. **注册模型契约**：将本目录 `deployment_contract.json` 中的 `fplc_gpu0_v325_wind_early_s2r_r6_u7832` 条目合入 `fplc_simcode12_model_contracts.json`（并在 `fplc_deployment.py` / `RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh` 菜单里添加对应 profile 选项）。
4. **上机前离线数值自检**：用 `r6_numpy_inference.py` 与机上 `fplc_policy.py` 对同一组随机 112-D 观测跑前向比对，确认 `max(|obs120_diff|) < 1e-6` 且 `max(|action_diff|) < 1e-6`，且 `u7447` profile 回归差异为 `0.0`。

## 60s 闭环验收摘要与适用边界

- **优势场景（已全面验证 100% 严苛 60s）**：
  - 实机零风 `RealExact_0w`：**32/32 = 100.0%**（`cRMS = 9.3 mm`，`inBand = 94.7%`）
  - 实机 `+X` 强风 `RealExact_PosX_1p6 / 2p4 / 2p8`：**全部 32/32 = 100.0%**（`cRMS = 7.8–8.4 mm`，`rY` 稳守 `[-50, -43] cm` 不撞 `-28 cm` 墙，`tilt95 = 6.4°–7.1°`）
  - 纯净 `P_warm` 外推至 `CleanPW_PosX_3p2`：**32/32 = 100.0%**
- **当前阶段已知边界（正由 `s2r_r6_v2` 专项消除）**：
  - `+X/-Y` 斜风（`-45°..-30°` @ `2.8 m/s`）存在偶发拍高擦顶 `rZ = 1.150 m`（`CleanPW_ConePosX_2p8 = 75.0%`）；
  - 实机 `-X` 2.8 m/s 存在软拍低反弹重触（`RealExact_NegX_2p8 = 68.8%`）；
  - `RealHard_0w` 极端扰动组合下为 `93.8%`（2 例擦 `rZ = 1.000 m` 下限）。
- 建议优先在**无风 + `+X` 风扇（实测 `1.6–2.8 m/s` 等效区间）**上做实机对比验证。
