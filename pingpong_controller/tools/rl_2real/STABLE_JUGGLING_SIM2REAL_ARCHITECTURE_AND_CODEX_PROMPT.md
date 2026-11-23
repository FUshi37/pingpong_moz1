# 稳定颠球 Sim2Real 架构、改造路线与 Codex 实现 Prompt

日期：2026-08-29（Asia/Shanghai）

状态：**系统设计与后续实现合同；本文件本身不授权启动训练、控制实物机器人或修改实物仓库。**

适用范围：GPU0 measured-QVEL、persistent bounded-`q_ref`、recovered-RMP 仿真代理、真实机器人内置 RMP，以及球状态、球拍碰撞和稳定颠球闭环。

本文有两个目的：

1. 用容易理解的方式说明为什么颠球 Sim2Real 不能靠“完全复现实物仿真”解决，以及推荐架构需要改什么。
2. 给后续 Codex 一组可直接使用、按阶段实施、带兼容性和验收约束的 prompt。

本文中的“V3 command-driven mirror”特指实物仓库 2026-08-29 的数字镜像分析版本，不是训练 profile `goal_d455_measured_qvel_rmp_vertical_v3`。

---

# 第一部分：架构、判断、改动和方法说明

## 1. 结论先行

稳定颠球的目标不应是让仿真和实物长期生成完全相同的逐 tick 轨迹，而应是：

> 允许每一次实物碰撞存在有限误差，但系统能够读取真实反馈，并在后续一至数次击球内把误差收缩回稳定工作区。

颠球是带离散碰撞事件的混合动力系统。微小的球拍位置、法向速度、角度、碰撞偏心和球旋转差异都可能显著改变碰后轨迹。完全精确地辨识所有接触参数既不可行，也没有必要。工程上可行的解法是同时做好五件事：

1. **接口等效**：仿真和实物必须使用完全相同的 actor 输入、状态更新、`q_ref` 语义、时间戳和单一 RMP 边界。
2. **闭环纠偏**：每次相机更新和每次碰撞后都基于真实状态重新决策，而不是复现固定周期动作。
3. **结构化鲁棒性**：依据真实数据建立相关的动力学/观测分布，不盲目扩大所有独立 DR 区间。
4. **在线适应**：从近期关节和碰撞历史估计未观测的执行器、球、旋转和接触隐变量。
5. **低维实物残差**：先用小幅、可解释、限幅、可复位的逐拍修正适配实物，不直接在机器人上更新完整 PPO actor。

推荐的最终形态是“可靠状态估计 + measured-feedback actor + 历史适应 + 逐拍残差 + 单一机器人 RMP”，而不是“一个越来越复杂、试图精确复制实物的 MuJoCo XML”。

## 2. 当前证据和可以下的结论

### 2.1 V3 command-driven mirror 证明了什么

在一段 18.33 s drag-ball 记录中，V3 使用真实 actor 已发布的同一 `q_ref` 驱动独立仿真 RMP/执行器，球被停放且碰撞关闭。post-warmup 指标从原始独立 actor mirror 到 command-driven V3 的变化为：

| 指标 | 原始独立 mirror | V3 common-command mirror | 变化 |
| --- | ---: | ---: | ---: |
| joint q RMS | 0.048014 rad | 0.002324 rad | -95.2% |
| joint dq RMS | 1.155064 rad/s | 0.050874 rad/s | -95.6% |
| racket position RMS | 0.027358 m | 0.001507 m | -94.5% |
| racket velocity RMS | 0.692860 m/s | 0.031077 m/s | -95.5% |

但是，同一 `q_ref` 下 V2 到 V3 的参数收益只有约 1%。因此：

- 约 95% 的改善主要来自消除两个 actor 的闭环分岔；
- 不能把约 95% 解释成 RMP 参数或仿真物理提高了约 95%；
- 该结果是单条、同日、无自由球碰撞的离线回放；
- 必须在 `record_new3/4/5` 的相同命令、相同初始化、相同 wall-clock 网格和相同后端上做成对复核。

V3 的正确定位是**执行器/RMP 辨识台和 shadow diagnostic**，不是新的实物控制器，也不是整体 Sim2Real 已解决的证明。

证据：

- `/home/yangzhe/Project/pingpong_playreal/pingpong-play/src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings/probe_recording_20260829_180205/analysis/v91_rmp_tuning_drag_ball_v3/SIM2REAL_1TO1_MIRROR_V3_ANALYSIS_ZH.md`

### 2.2 当前碰撞模型的能力边界

`record_new3/4/5` 共筛出 116 次有效碰撞。真实法向恢复系数总体中位数约 0.665，但 `record_new5` 的 8 次留出碰撞中位数约 0.985，说明单一恢复系数不足以描述跨球、跨工况和残余测量变化。

当前证据还说明：

- 90 Hz 相机不能辨识约 3 ms 的真实接触时间；
- 没有球自旋观测时，胶皮摩擦与入射自旋不可分辨；
- 球质量来自 4.0 g 实物先验，不是由轨迹独立辨识；
- CPU MuJoCo 与 MJX 在高速、偏心、有摩擦的接触上并非逐步等价；
- 当前参数能够改善平均法向反弹，但不能声明完全 1:1。

因此未来应优先拟合“碰前状态到碰后状态”的**结果映射和不确定度**，而不是继续把所有误差压进一个 `solref damping`。

证据：

- `/home/yangzhe/Project/pingpong_playreal/pingpong-play/analysis_results/ball_racket_contact_identification/SIM2REAL_BALL_RACKET_CONTACT_ANALYSIS_ZH.md`

### 2.3 静态起球与动态碰撞必须分开

初始起球分析发现：相机初态给出的球心到拍面法向距离只有 8.4 mm，小于 20 mm 球半径；物理放拍约为 22.0 mm，默认 reset 约为 30.7 mm。actor FK 与真实碰撞拍面 z 还存在约 10.7 mm 平均差异。

冲击数据拟合得到的低 damping 接触用于静态放拍时会产生非物理储能并把球推到异常高度。这不否定动态碰撞拟合，而是证明以下合同必须分开：

- `static/low-speed support and launch`；
- `dynamic impact and rebound`。

第一拍最好由独立、确定性、可重复的 launch 模式完成，或者从已经进入可观测自由飞行的球状态切入 steady juggling。不能把高速冲击参数直接外推到球静置在拍面上的初始化。

证据：

- `/home/yangzhe/Project/pingpong_playreal/pingpong-play/src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings/probe_recording_20260829_141442/analysis/v91_initial_launch_causal/SIM2REAL_INITIAL_LAUNCH_CAUSAL_ANALYSIS_ZH.md`

### 2.4 更宽 DR 不自动产生更强鲁棒性

V70 增加高度相关奖励后修复了 apex，但 `conv_len/full` 仍失败；V71 继续增加耦合奖励仍失败。V72 只把六个 in-view observation-calibration DR 字段从 25% 边界限制到 12.5% 中点，就恢复了 hits、`conv_len`、full、view 和 motion quality 的联合通过。

这说明：

- reward 能解决 credit 对齐问题，但不能代替分布设计；
- 独立、过宽、未经测量的 DR 会把训练推入不必要或互相冲突的组合；
- DR 的目标是覆盖可信实物后验和恢复边界，不是最大化区间宽度。

证据：

- `pingpong_controller/outputs/rl_sim/v70_height_survival_reward_repair_20260826/REWARD_DESIGN_AND_PREFLIGHT.md`

### 2.5 当前综合判断

如果 `record_new3/4/5` 的 common-command replay 进一步确认 RMP/执行器输出已经足够接近，那么后续优先级应是：

1. 球状态、时间戳、坐标和碰撞拍面合同；
2. 动态接触结果和未观测旋转；
3. 第一拍初始化与 steady-juggling 切换；
4. 策略对碰撞后偏离状态的恢复吸引域；
5. 最后才是更精细的名义 RMP/PD 参数微调。

如果跨 `record_new3/4/5` 的 common-command replay 仍出现系统性 joint/racket 缺口，则必须先完成 RMP/低层闭环辨识，不能用更宽 DR 或 PPO 把未校准名义模型掩盖掉。

## 3. 稳定性目标：从轨迹一致改为碰撞到碰撞的误差收缩

令第 `k` 次有效碰撞后的任务状态为：

```text
z_k = [post-impact ball velocity,
       estimated spin or latent contact state,
       absolute apex and lift,
       lateral drift,
       next-contact position/timing,
       hit offset]
```

目标状态为 `z*`，误差为经过物理容差归一化后的 `e_k = z_k - z*`。稳定控制的核心验收不是仿真和实物逐 tick 相等，而是：

```text
E[||e_(k+1)|| | e_k] <= rho * ||e_k|| + epsilon
```

其中在声明的恢复工作区内应有 `rho < 1`。实际报告中不能只给一个全局平均 `rho`，还要给：

- 按误差方向和大小分箱的下一拍误差；
- 单拍、两拍和多拍恢复概率；
- `P(N >= k)` 生存曲线，而不只是平均 hits；
- 不同球、不同日期和不同初态的置信区间；
- 失败模式和 safety intervention 分布。

该指标不要求每一拍单调变好。由于相位、限幅和接触离散性，单次误差可能短暂变大；关键是条件分布、有限步恢复和长期生存表现具有收缩趋势。

## 4. 必须保留的当前实物控制边界

当前 measured-QVEL/persistent bounded-`q_ref` 实物链路必须保持：

```text
encoder q/dq + timestamped ball state + encoder FK
  -> exact versioned actor observation
  -> deterministic actor mean / clipped 7-D QVEL action
  -> persistent bounded q_ref at 200 Hz
  -> existing safety limiter
  -> MechUnitCmd.jnt_pos
  -> robot existing RMP / low-level controller
```

训练链路使用 recovered RMP 和 XML PD 只是实物已有 RMP/低层控制器的仿真代理：

```text
sim measured q/dq
  -> versioned actor
  -> persistent bounded q_ref
  -> recovered 1 kHz RMP
  -> delayed RMP q/qd/qdd
  -> XML PD
  -> MJX state
```

必须继续遵守：

- 实物端只能有机器人已有的一个 RMP；
- 不得部署 recovered RMP、其仿真 output delay 或 XML PD；
- 不得叠加 actuator model、FOPDT、inverse compensation、第二个 planner 或 post-RMP hold；
- 不得把模拟器 `command_q` 当成 actor 输出；
- 现有 checkpoint 的 observation、action 和 `q_ref` 状态语义不可静默改变；
- 任何历史、latent、residual、predictor 或新 age 语义都要求新的单调版本 profile 和 deployment adapter；
- stale/non-finite/invalid 状态必须 fail closed。

当前详细合同：

- `pingpong_controller/tools/rl_2real/GPU0_QVEL_REAL_RMP_GUIDE.md`
- `pingpong_controller/tools/rl_2real/MEASURED_QVEL_RMP_V2_DEPLOYMENT_CONTRACT.md`

## 5. 推荐的三时间尺度闭环架构

```text
                     ┌──────────────────────────────────────┐
90 Hz camera frames ─► timestamp-consistent ball estimator │
                     │ current-time pos/vel/cov/validity    │
                     └──────────────────┬───────────────────┘
                                        │
200 Hz encoder q/dq/FK ─────────────────┼──────────────┐
                                        ▼              │
                              ┌──────────────────┐      │
recent command/state history ─► temporal adapter │      │
recent impact outcomes ───────► fast/contact z   │      │
                              └────────┬─────────┘      │
                                       ▼                │
                              ┌──────────────────┐      │
                              │ measured actor   │      │
                              │ + bounded residual│     │
                              └────────┬─────────┘      │
                                       ▼                │
                              persistent bounded q_ref  │
                                       ▼                │
                               safety + robot RMP       │
                                       ▼                │
                                     robot ─────────────┘

After each valid impact:
pre/post ball + racket state -> impact outcome -> latent/residual update

Read-only shadow:
the same published q_ref -> command-driven sim mirror -> diagnostics only
```

三个时间尺度分别承担不同职责：

| 时间尺度 | 频率/事件 | 主要职责 |
| --- | --- | --- |
| 关节反馈环 | 200 Hz actor，机器人内部更高频 | 使用真实 `q/dq` 更新 bounded-`q_ref`，抑制执行器状态误差积累 |
| 飞行规划环 | 90 Hz 新球状态、200 Hz held/predicted state | 更新时间一致的球状态，持续重算下一次接触，而不是坚持旧轨迹 |
| 碰撞适应环 | 每次有效碰撞 | 从实际 apex、XY、出球速度、偏心和接触时序更新隐变量与下一拍小幅修正 |

## 6. 各模块的输入、输出和方法

### 6.1 Common-command RMP/执行器辨识台

作用：隔离“同一命令下实物与仿真 plant 的差异”，不让第二个 actor 把微小状态差异反馈成完全不同的命令。

输入：

- 真实发布的 `publishedSafePosition/q_ref[7]`；
- `wallTimeS`、发布、encoder 和 source timestamp；
- 真实 `q[7]`、`dq[7]`；
- 统一 FK 定义下的 racket pose/twist；
- 明确的 warm-up、hold 和有效样本掩码。

输出：

- 仿真 `q/dq/qdd` 和 racket pose/twist/acceleration；
- per-joint、per-component 和时间分箱误差；
- 名义参数候选及跨记录不确定度；
- 完整的源文件哈希、配置、初始化和网格合同。

非目标：

- 不证明球碰撞正确；
- 不向实物控制回写仿真状态；
- 不用原始独立 actor mirror 的 95% 改善替代 V2→V3 参数增益；
- 不跨不同记录、后端和有效样本合同直接比较聚合 RMS。

### 6.2 时间一致的球状态估计器

输入：

- 原始检测位置、原始检测时间戳、到达时间；
- 相机标定和 world/base 变换；
- 可选的弹道/drag/Magnus 模型；
- 检测有效性、视野状态和最近碰撞事件。

输出建议：

```text
BallEstimate:
  position_m[3]
  velocity_m_s[3]
  represented_state_time_s
  covariance_or_scale
  raw_source_age_s
  prediction_horizon_s
  valid
  raw_detected
  in_view
  reacquiring
```

关键语义：

- actor `ball_obs_age` 是“actor 内状态所代表时刻”的年龄；
- 已预测到当前 query time 的状态，其 represented age 应接近零；
- raw detection age、prediction horizon、confidence、in-view 和 missing 必须独立；
- 不得给当前时刻预测值附上旧曝光时间并再次累计延迟；
- 当前 V85/V91 类 57-D checkpoint 使用其冻结的 ball-state 合同，不能直接插入新滤波器；新估计器必须通过新 profile 训练和部署。

方法上可以先使用带飞行/碰撞模式的 EKF/UKF 或稳健弹道拟合。若旋转不可观测，输出不应伪造精确 spin，而应提供不确定度或交给历史 latent 推断。

### 6.3 碰撞结果模型，而不是单一微观参数

建议的数据接口：

```text
ImpactInput:
  pre_ball_position_m[3]
  pre_ball_velocity_m_s[3]
  estimated_spin_rad_s[3] or spin_unknown
  racket_position_m[3]
  racket_normal[3]
  racket_linear_velocity_m_s[3]
  racket_angular_velocity_rad_s[3]
  impact_offset_m[2]
  ball_id / session_id

ImpactOutcome:
  post_ball_velocity_m_s[3]
  post_spin_rad_s[3] if observable
  apex_abs_m
  lift_m
  next_contact_or_landing_xy_m[2]
  fit_quality / covariance / validity
```

模型优先级：

1. 保留已验证的名义物理模型；
2. 对碰后速度或冲量增加小的 residual mean；
3. 同时拟合 residual covariance 或模型 ensemble；
4. 在仿真中按 outcome posterior 采样；
5. 只有留出记录通过后才进入训练 profile。

应至少按入射法向速度、切向速度、偏心距离、拍面角度、球 ID/日期分层验证。`record_new3/4` 可作拟合，`record_new5` 保持留出；如果留出仍不能解释，优先补充球面标记/高速相机旋转数据，而不是扩大所有接触参数。

### 6.4 结构化域随机化

随机化应表示真实世界中“可能共同出现”的参数组合：

- RMP/PD 增益可以使用全局 episode 因子叠加固定 per-joint 标定向量；
- output delay 可以保留 per-joint 差异；
- 一个球的 mass/inertia/contact latent 在同一 episode 内应保持一致；
- 不同碰撞可以叠加较小的 event-level residual；
- observation frame bias 应 episode-coherent，刷新噪声才是 per-frame；
- 漏帧应使用真实 burst/hold 分布；
- 静态 launch 参数和动态 impact 参数不能共用一个不加模式区分的随机变量。

候选分布应来自真实数据 posterior、低维参数族或模型 ensemble。随机化顺序为：

1. 名义一致性；
2. 单一物理族的小范围桥接；
3. unchanged-domain polish；
4. 下一物理族；
5. 最后才组合尾部。

禁止：

- 直接用独立 global min/max 生成所有组合；
- 为覆盖损坏、掉球后、stale 或 safety intervention 数据而扩大 DR；
- 为通过 coverage 丢弃有效困难记录；
- 用“所有 candidate 数值稳定”替代 pointwise motion coverage；
- 在同一个实验同时改变 DR、reward、PPO、observation 和 reset。

### 6.5 历史适应模块

当前 feed-forward 57-D actor 能根据已经出现的真实偏差做反应，但不能可靠辨识慢变或隐藏动力学。下一代 profile 可以增加两个不同时间尺度的历史表示：

- fast plant history：最近约 50–100 ms 的 `q_ref/q/dq/action/timestamp/hold`；
- contact history：最近 1–3 次有效碰撞的碰前/碰后球状态、apex、偏心和时序。

推荐先在仿真中采用 privileged teacher/student：

- teacher 看见真实采样的 RMP/PD、ball/contact/observation latent；
- adaptation encoder 只看部署可用历史；
- actor 使用估计 latent 和当前观测；
- critic 可继续看完整 privileged state；
- student 的 actor 输入、hidden-state reset、序列长度和部署状态机必须明确版本化。

必须做的 ablation：

1. 当前 57-D feed-forward baseline；
2. 仅 plant history；
3. 仅 contact history；
4. plant + contact history；
5. oracle latent teacher 上界。

只有历史模型在相同参数量或报告参数量差异的前提下，显著改善 fresh-seed recovery/survival，才可认为收益来自适应能力，而不是网络更大。

### 6.6 逐拍低维残差

最先上实物的适应不应是完整 actor 在线 PPO，而应是事件驱动、限幅、可解释的 residual：

```text
previous bounce outcome error
  -> bounded residual state
  -> small correction to the versioned actor action/QVEL intent
  -> the same persistent bounded q_ref
  -> the same safety limiter and robot RMP
```

建议的物理对应关系：

| 实测误差 | 下一拍候选修正 |
| --- | --- |
| apex/lift 偏低 | 小幅增加接触时有效法向拍速 |
| apex/lift 偏高 | 小幅减小有效法向拍速 |
| 球持续向 X/Y 漂移 | 小幅修正拦截位置或有效拍面倾角 |
| 接触持续偏心 | 修正下一拍拦截中心 |
| 同方向误差连续多拍 | 更新 episode-level bias，而不是每 tick 高频追噪声 |
| 状态不确定度过高 | 残差衰减到零或进入安全状态 |

第一版可采用手工线性/增益调度残差，用真实单拍数据确定符号、延迟和限幅。后续才能用离线监督、contextual bandit、Bayesian optimization 或小型 residual policy 学习其参数。

该 residual 必须：

- 默认关闭；
- 有独立 profile 和 checkpoint/配置校验；
- 每个输出有 SI 单位、硬限幅和 rate limit；
- episode reset 时原子清零；
- stale/invalid ball 或 safety intervention 时立即清零；
- 记录 raw actor action、residual、combined action、`q_ref` 和最终 safe target；
- 不新增第二个 RMP、PD 或 execution delay。

### 6.7 Launch、steady、recovery 和 fail-safe 模式

第一拍与稳定颠球的物理状态不同，建议显式划分：

1. `WAIT_VALID`：等待 q/dq、球状态、时间戳和工作区全部有效。
2. `LAUNCH`：从可重复的 settled/on-racket 合同或自由飞行初态完成第一拍。
3. `STEADY`：进入已训练的稳定颠球工作区。
4. `RECOVERY`：球仍可达但超出 nominal orbit 时，允许更保守的恢复动作。
5. `FAIL_SAFE`：球、关节、时间戳、预测器或安全合同失败，停止 policy command 并进入已有监督安全路径。

模式切换必须由可观测且记录的条件触发；不得通过隐藏的计时或人工动作让仿真和实物产生不同状态机。

### 6.8 Read-only shadow digital twin

command-driven mirror 可以常驻作为只读诊断：

- 输入真实发布的同一 `q_ref`；
- 使用仿真 RMP/PD/ball/contact ensemble；
- 输出 plant residual、模型置信度和预测分歧；
- 不直接改变实物 action、`q_ref` 或 safety state。

在 shadow 数据积累并通过离线反事实验证后，某些低维 latent 才可以作为新版本 actor 的输入。不能让尚未验证的 mirror 输出在线闭环控制真实机器人。

## 7. 推荐的数据合同

后续实现应优先统一以下结构化日志。字段可以适配现有代码命名，但语义、单位和时间戳不可省略。

### 7.1 ControlTickRecord（每个 200 Hz tick）

```text
wall_time_s
policy_time_s
encoder_time_s
publish_time_s
ball_represented_time_s
ball_raw_source_time_s
q_rad[7], dq_rad_s[7]
raw_action[7], clipped_action[7]
residual_action[7]
q_ref_rad[7], safe_target_rad[7]
racket_position_m[3]
racket_orientation
racket_linear_velocity_m_s[3]
racket_angular_velocity_rad_s[3]
ball_position_m[3], ball_velocity_m_s[3]
ball_age_s, prediction_horizon_s
ball_valid, raw_detected, in_view, reacquiring
publish_hold_ticks
safety_intervention_flags
profile/checkpoint/config identity
```

### 7.2 ImpactRecord（每个有效碰撞）

```text
impact_id, episode_id, ball_id, session_id
impact_time_s and timestamp quality
pre/post ball state and fit covariance
racket state at physical contact edge
impact offset and contact normal
estimated/present/unknown spin status
subsequent apex, lift, lateral drift and next-contact state
validity/exclusion reason
```

### 7.3 EpisodeRecord

```text
reset mode and initial geometry
all sampled episode-level DR/latent values in simulation
number and timing of valid contacts
time-to-failure and failure type
full/truncated/safety outcome
P(N >= k) contribution
maximum limits and safety interventions
```

所有报告必须记录 source hashes、代码/XML 状态、选样和排除原因。CSV/JSON/Parquet 等结构化数据是证据主体，图片只用于解释。

## 8. 分阶段改造路线

每一阶段应是独立、可回滚、默认关闭的变更。不要一次实现完整架构。

### Phase 0：证据和接口统一，不改变控制行为

目标：建立可靠的数据、时间和评估合同。

改动范围：

- 统一 common-command replay 的输入和输出 schema；
- 在 `record_new3/4/5` 上复现相同命令的 wall-clock replay；
- 加入 per-joint/per-racket/worst-component 和有效区间报告；
- 统一 ImpactRecord 和初态几何记录；
- 建立 collision-outcome、survival curve 和 recovery metric 的离线计算器；
- 所有新字段只记录，不进入现有 actor。

验收：

- 历史 profile/controller 的 fixed-seed 输出逐值不变；
- 相同输入重复运行结果确定；
- 所有 64/65 记录报告 parse/data failure，不静默排除；
- old baseline 与 candidate 使用相同后端、初始化、网格和样本掩码；
- 产物有命令、hash、config 和排除清单。

这是下一次代码修改最推荐的起点。

### Phase 1：仿真侧碰撞 outcome residual 和结构化 DR

目标：用真实碰前/碰后结果约束仿真分布，不改变实物 runtime。

改动范围：

- 新增 opt-in collision outcome residual/ensemble；
- 保留名义 MuJoCo/MJX 接触作为默认；
- 增加 episode-level 与 event-level 分层采样；
- 增加 `record_new3/4` fit、`record_new5` held-out validator；
- 静态 launch 和动态 impact 分开；
- 新 profile 和 `TRAINING_CURRICULA.md` 注册。

验收：

- legacy XML/profile fixed-seed regression 通过；
- held-out 3D/normal/tangential velocity、apex 和 lateral outcome 不回归；
- 输出 posterior/calibration curve，而不只给均值 RMSE；
- MJX 和 CPU 后端差异单独报告；
- 先做 frozen-policy paired replay，再决定是否进行 bounded GPU training。

### Phase 2：历史适应策略

目标：让策略从部署可用历史推断 plant/contact latent，并扩大恢复吸引域。

改动范围：

- 新的 observation/profile/checkpoint 维度；
- history buffer/encoder/recurrent hidden-state reset；
- privileged teacher/critic latent；
- impact history 和 fast plant history ablation；
- 专门的 post-impact perturbation/recovery curriculum；
- 新的 NumPy/实物 adapter 设计，但此阶段不自动部署。

验收：

- 57-D baseline 和所有旧 checkpoint 完全保留；
- hidden-state reset、mask、batch/JIT 行为有单元测试；
- oracle、feed-forward、history variants 在相同 fresh seeds 下配对比较；
- recovery、survival、apex/view/motion/safety 联合通过；
- 至少两 seed 或一主 seed 加独立确认；
- 只有 simulation candidate，不能宣称 deployable。

### Phase 3：实物状态估计和 exact runtime parity

目标：在不发命令的 shadow 条件下复现新策略的输入和状态机。

前提：用户必须明确把 `/home/yangzhe/Project/pingpong_playreal/pingpong-play` 纳入可修改范围。

改动范围：

- versioned ball estimator/runtime adapter；
- exact observation/action/hidden-state parity；
- timestamp、prediction、validity、reacquisition 日志；
- no-publish shadow 和 recorded trace replay；
- 保留所有现有 real profile。

验收：

- fixed observation 和 recorded trace 的 NumPy/JAX parity；
- actor、history、`q_ref`、mode state 逐 tick 对齐；
- stale/future/non-finite/invalid/reacquisition 全部 fail closed；
- no-publish shadow 通过后，才允许监督无球/有限工作区测试。

### Phase 4：逐拍 residual 和分级实物验证

目标：用小样本、低维、有物理解释的修正适配真实碰撞结果。

改动范围：

- 默认关闭的 event-driven residual state；
- 独立限幅、rate limit、reset 和 safety clear；
- shadow counterfactual；
- 单拍到多拍的人工监督放权。

验收顺序：

1. 离线记录反事实；
2. no-publish shadow；
3. 无球、低幅、有限工作区；
4. 标准化单次 launch/impact；
5. 2-hit；
6. 3–5-hit；
7. 10-hit；
8. 多球、多日的长时间生存。

任何频繁 safety limiter 干预、时间戳异常、持续 action/residual 饱和或未登记 failure mode 都是合同失败，不能当作正常鲁棒性行为。

## 9. 预注册验收矩阵

| 层级 | 必须回答的问题 | 核心指标 | 不能替代的证据 |
| --- | --- | --- | --- |
| RMP/plant replay | 相同 `q_ref` 是否产生相近 q/dq/racket？ | per-joint RMS/quantile、worst component、时间分箱、有效样本 coverage | 独立 actor mirror 的聚合改善 |
| ball estimator | actor 收到的球状态是否时间一致且可信？ | represented age、position/velocity continuity、covariance calibration、drop/reacquisition failure | 单帧视觉图片 |
| dynamic impact | 碰后分布是否覆盖实物？ | 3D/normal/tangential velocity、apex、XY、held-out calibration | 单一 restitution 或训练集 RMSE |
| initialization | 第一拍是否从相同物理合同开始？ | ball-plane gap、solver warm state、静态漂移、launch outcome | 数值上偶然接近的穿透初态 |
| robust policy | 偏差是否能被后续击球收缩？ | finite-step recovery、`P(N>=k)`、apex/view/motion/safety | 平均 reward 或一条视频 |
| real residual | 小修正是否真实改善且不破坏安全？ | paired outcome change、saturation、safety intervention、multi-day confidence | 仿真收益或在线单次成功 |

训练候选仍必须报告既有指标：hits、episode length、hit1/hit3/hit12、truncation/full、view、apex/lift/contact height、ball/racket lateral motion、racket angular speed、qvel/qacc、KL、explained variance 和完整 failure distribution。

## 10. 关键决策树

```text
record_new3/4/5 common-command replay 是否通过？
  ├─ 否：先修 nominal RMP/low-level parameterization；不启动 PPO，不扩大所有 DR。
  └─ 是：冻结该执行器基线，转向 ball/vision/contact。

dynamic impact held-out 是否通过？
  ├─ 否：补充 spin/高速相机/跨球数据；不把训练集参数当物理真值。
  └─ 是：进入 opt-in outcome residual + structured DR。

history policy 是否在 fresh seeds 改善 recovery/survival？
  ├─ 否：保留 57-D baseline；检查历史语义、credit 和参数量 ablation。
  └─ 是：建立 versioned runtime parity，仍不直接发实物命令。

shadow residual 是否在真实记录反事实中改善？
  ├─ 否：不进入实物命令链。
  └─ 是：按单拍→多拍分级监督验证。
```

## 11. 明确不推荐的方向

- 继续追求仿真和实物无限长轨迹逐 tick 一致。
- 把 V3 约 95% 改善解释成参数精度提高约 95%。
- 只在一条 drag-ball 记录上确定 RMP/PD 真值。
- 用一个 `solref damping` 同时拟合静态承托和高速碰撞。
- 在无旋转观测时宣称已经辨识胶皮摩擦。
- 独立扩大全部 DR 区间，或用不可能的参数组合训练“鲁棒性”。
- 在同一实验同时改变碰撞、DR、reward、PPO 和 observation。
- 给当前 57-D checkpoint 静默增加滤波、历史、latent 或新 age 语义。
- 在真实 actor 前再部署 recovered RMP、第二个 planner 或执行器补偿栈。
- 一开始就在实物上对完整 actor 做 PPO fine-tune。
- 用 reward、平均 hits 或单条成功视频作为稳定上机证据。

## 12. 当前成果和报告地址

### 主仓库

- `AGENTS.md`
- `TRAINING_CURRICULA.md`
- `pingpong_controller/tools/rl_2real/GPU0_QVEL_REAL_RMP_GUIDE.md`
- `pingpong_controller/tools/rl_2real/MEASURED_QVEL_RMP_V2_DEPLOYMENT_CONTRACT.md`
- `pingpong_controller/outputs/rl_sim/v70_height_survival_reward_repair_20260826/REWARD_DESIGN_AND_PREFLIGHT.md`

### 实物仓库分析

- V3 command-driven mirror：
  `/home/yangzhe/Project/pingpong_playreal/pingpong-play/src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings/probe_recording_20260829_180205/analysis/v91_rmp_tuning_drag_ball_v3/SIM2REAL_1TO1_MIRROR_V3_ANALYSIS_ZH.md`
- V3 参数：
  `/home/yangzhe/Project/pingpong_playreal/pingpong-play/src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings/probe_recording_20260829_180205/analysis/v91_rmp_tuning_drag_ball_v3/proposed_sim_mirror_rmp_override_v3.json`
- V3 组合方案：
  `/home/yangzhe/Project/pingpong_playreal/pingpong-play/src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings/probe_recording_20260829_180205/analysis/v91_rmp_tuning_drag_ball_v3/sim_mirror_digital_twin_v3.json`
- V3 图片目录：
  `/home/yangzhe/Project/pingpong_playreal/pingpong-play/src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings/probe_recording_20260829_180205/analysis/v91_rmp_tuning_drag_ball_v3/three_way_trajectory_comparison/`
- 碰撞辨识：
  `/home/yangzhe/Project/pingpong_playreal/pingpong-play/analysis_results/ball_racket_contact_identification/SIM2REAL_BALL_RACKET_CONTACT_ANALYSIS_ZH.md`
- 碰撞图片：
  `/home/yangzhe/Project/pingpong_playreal/pingpong-play/analysis_results/ball_racket_contact_identification/contact_fit_diagnostics.png`
- 初始起球：
  `/home/yangzhe/Project/pingpong_playreal/pingpong-play/src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings/probe_recording_20260829_141442/analysis/v91_initial_launch_causal/SIM2REAL_INITIAL_LAUNCH_CAUSAL_ANALYSIS_ZH.md`
- 初始起球图片目录：
  `/home/yangzhe/Project/pingpong_playreal/pingpong-play/src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings/probe_recording_20260829_141442/analysis/v91_initial_launch_causal/`

### 方法参考

- Dynamics Randomization：<https://arxiv.org/abs/1710.06537>
- SimOpt：<https://arxiv.org/abs/1810.05687>
- Neural Posterior Domain Randomization：<https://proceedings.mlr.press/v164/muratore22a.html>
- UP-OSI：<https://www.roboticsproceedings.org/rss13/p48.html>
- Rapid Motor Adaptation：<https://www.roboticsproceedings.org/rss17/p011.html>
- Physics Models for Sim-to-Real Robot Table Tennis：<https://arxiv.org/abs/2606.28805>

---

# 第二部分：供 Codex 进行代码修改的 Prompt

## 13. 使用方式

不要让 Codex 一次实现全部 Phase。每次复制“总 Prompt”，填写一个明确的 `IMPLEMENTATION_PHASE`，再附上相应阶段任务。推荐第一次从 `PHASE_0` 开始。

必须显式填写：

```text
IMPLEMENTATION_PHASE=PHASE_0 | PHASE_1 | PHASE_2 | PHASE_3 | PHASE_4
REAL_REPO_IN_SCOPE=NO | YES
ALLOW_GPU_EXPERIMENT=NO | YES
ALLOW_HARDWARE_COMMAND=NO | YES
```

默认值应是：

```text
IMPLEMENTATION_PHASE=PHASE_0
REAL_REPO_IN_SCOPE=NO
ALLOW_GPU_EXPERIMENT=NO
ALLOW_HARDWARE_COMMAND=NO
```

## 14. 总 Prompt（可直接复制）

```text
你正在 /home/yangzhe/Project/pingpong_controller 工作。

本次任务参数：
IMPLEMENTATION_PHASE=<填写 PHASE_0/1/2/3/4>
REAL_REPO_IN_SCOPE=<NO/YES>
ALLOW_GPU_EXPERIMENT=<NO/YES>
ALLOW_HARDWARE_COMMAND=<NO/YES>

目标：按照
pingpong_controller/tools/rl_2real/STABLE_JUGGLING_SIM2REAL_ARCHITECTURE_AND_CODEX_PROMPT.md
中对应阶段的合同，完成一个最小、版本化、默认关闭、可验证的实现。不要跨阶段扩大范围。

开始前必须完整阅读并按优先级核对：
1. AGENTS.md
2. TRAINING_CURRICULA.md
3. STABLE_JUGGLING_SIM2REAL_ARCHITECTURE_AND_CODEX_PROMPT.md
4. 本次所选 profile builder、launcher、deployment guide 和现有 tests
5. 本文列出的相关日期报告和结构化产物

硬约束：
- 保留所有历史 profile、checkpoint、observation、action、reset、launcher 和部署路径。
- 新行为必须 opt-in/default-off，并使用新的单调版本 profile；如果 profile 语义变化，同步更新 TRAINING_CURRICULA.md。
- 不得从文件名推断 checkpoint、GPU、XML、stage 或 action 语义；从 payload、launcher、hash 和代码确认。
- 不得在实物端叠加 recovered RMP、XML PD、actuator model、FOPDT、inverse compensation、第二 planner、sim output delay 或 post-RMP hold。
- 实物唯一命令边界仍是安全限制后的 MechUnitCmd.jnt_pos=q_ref，进入机器人已有 RMP。
- 当前 57-D checkpoint 的输入和 persistent bounded-q_ref 状态不可静默改变。
- observation/history/latent/residual/predictor/age 语义变化必须创建新的 profile、adapter、golden parity 和 fail-closed 校验。
- 不得覆盖已有 output、checkpoint、W&B history 或报告；所有实验使用新的目录和 identity。
- 不得删除或整理用户已有 dirty-worktree 改动；先报告重叠，再只修改本次范围。
- 先写回归测试，再改控制、observation、delay、reset、contact 或 checkpoint migration。
- 不把更高 return、单条视频、单次 replay 或全局平均 RMS 当作完成证据。
- 任何未授权 GPU、W&B-online、实物发布或真实机器人动作都禁止执行。
- REAL_REPO_IN_SCOPE=NO 时，/home/yangzhe/Project/pingpong_playreal/pingpong-play 只读。

工作流程：
1. 检查 git status、相关文件、当前 profile/launcher、XML、checkpoint contract 和现有报告；列出用户已有改动，避免覆盖。
2. 用一句话重述本阶段唯一因果假设，并在实现前写清 acceptance metrics、baseline、candidate、seed、数据 population、排除规则和产物路径。
3. 先添加 legacy regression/fail-closed tests，证明 default-off 旧路径逐值不变。
4. 实现本阶段最小功能；不要顺手修改 reward、PPO、DR、observation 或控制的其他组。
5. 运行最小单元测试、syntax/static checks、deterministic CPU smoke 和 fixed-seed paired replay。
6. 只有 ALLOW_GPU_EXPERIMENT=YES 且 CPU/preflight 通过时，才可运行已审查 launcher 的 bounded GPU trial；确认 JAX 是指定 CUDA device。
7. 只有 ALLOW_HARDWARE_COMMAND=YES、实物仓库明确在范围内、离线 parity/shadow 全部通过且用户在场监督时，才可进入分级实物测试；否则只能生成代码、测试和 no-publish/shadow 方案。
8. 保留所有负结果，生成结构化 JSON/CSV 和简明报告，记录代码/XML 状态、命令、seed、GPU UUID、checkpoint/hash、profile/stage、指标和失败分布。
9. 最终报告 changed files、接口、测试、证据、未解决风险和下一阶段；不要宣称超出证据的硬件覆盖或稳定性。

完成定义：
- 本阶段功能和测试完成；
- default-off legacy behavior 保持；
- preregistered acceptance 通过，或明确记录为负结果；
- 没有未授权训练/实物动作；
- 文档和产物足以由另一位 Codex 复现。

如果本阶段需要一个会改变 observation/action/control boundary 的用户选择，或需要把实物仓库纳入写范围，停止实现并请求用户明确授权，不自行假设。
```

## 15. PHASE_0 附加任务 Prompt：证据与接口统一

```text
本次只实现 PHASE_0，不改变训练或实物控制行为。

唯一假设：在相同 published q_ref、wall-clock 5 ms 网格、相同初始化和相同后端下，
command-driven replay 可以把 plant/RMP 误差与双 actor 闭环分岔分离，并为后续
structured DR 提供可复现的 per-record/per-component 证据。

请先检查并复用已有 replay/analysis 工具，不要复制一套语义近似的新实现。实现：
1. 一个版本化的 common-command replay 输入 schema/validator；
2. record_new3/4/5 的显式发现、parse/data-failure 清单和 source hash；
3. wallTimeS 5 ms 网格、published-target ZOH、joint-header-time 插值；
4. joint q/dq/qdd 与 racket position/orientation/linear/angular velocity/acceleration
   的 per-record、per-joint、per-component、post-warmup、held-out 指标；
5. worst-component curves、coverage artifacts 和 baseline/candidate paired deltas；
6. ImpactRecord、ControlTickRecord、EpisodeRecord 的纯数据 schema；
7. collision-to-collision recovery 和 P(N>=k) 的离线计算器；
8. deterministic repeat、invalid record、timestamp、hold、sample-mask、legacy path tests；
9. 一个 dated report，明确 V3 architecture gain 与 V2->V3 parameter gain。

不要：
- 修改 actor、reward、RMP、PD、contact、DR、reset 或 deployment runtime；
- 启动 PPO；
- 控制机器人；
- 用 sourceTimeS 代替 formal wall-clock replay；
- 静默丢弃 record_new3/4/5 的失败记录。
```

## 16. PHASE_1 附加任务 Prompt：碰撞结果模型与结构化 DR

```text
本次只实现 PHASE_1，实物 runtime 保持不变。

前提：PHASE_0 的 common-command replay、ImpactRecord 和报告已经通过。
唯一假设：在名义 MuJoCo/MJX 接触之上增加小的、分布化的碰后 outcome residual，
比继续用一个 solref damping 或独立宽 DR 更能覆盖 held-out 实物碰撞结果。

实现要求：
1. 保留当前 contact path 为默认，新增 opt-in outcome-residual mode；
2. 输入至少含碰前球速、racket pose/twist、法向、偏心和 ball/session identity；
3. 输出 post-impact velocity residual mean 与 uncertainty/ensemble；
4. 训练集固定 record_new3/4，record_new5 固定 held-out；
5. static/low-speed launch 与 dynamic impact 使用不同 mode/contract；
6. episode-level ball/contact latent 与 event-level residual 分开采样；
7. legacy XML/profile fixed-seed regression；
8. CPU MuJoCo/MJX 分别验证 normal/tangential/3D velocity、apex、XY 和 calibration；
9. 只有 held-out 与 frozen-policy paired replay 通过后，才创建新的单调 profile；
10. 如需训练，先更新 TRAINING_CURRICULA.md、写 launcher/preflight，再等待用户授权 GPU。

不要把质量、摩擦、自旋或 3 ms contact time 写成已经由 90 Hz 数据唯一辨识。
不要修改当前 released/experimental profile 的 contact 字段。
```

## 17. PHASE_2 附加任务 Prompt：历史适应与恢复训练

```text
本次只实现 PHASE_2，不修改现有 57-D actor/checkpoint 语义。

前提：PHASE_0/1 产物通过，并已有 versioned structured-DR simulator mode。
唯一假设：部署可用的 fast plant history 和 recent impact history 能推断隐藏动力学，
从而扩大 post-impact perturbation 的有限步恢复区域。

实现要求：
1. 新建单调版本 profile，旧 57-D actor 是不可变 baseline；
2. 明确定义 history fields、长度、时间单位、mask、reset 和 recurrent/encoder state；
3. teacher/critic 可以看到 realized DR，actor/adaptation 只能看到部署可用历史；
4. checkpoint validator 必须拒绝错误维度、错误 history contract 和错误 reset semantics；
5. NumPy/JAX golden parity 覆盖 actor、encoder/hidden state 和 bounded q_ref；
6. ablation：57-D baseline、plant-only、impact-only、combined、oracle latent；
7. curriculum 必须包含 post-impact perturbation、finite-step recovery 和 unchanged-domain polish；
8. acceptance 同时保护 survival、P(N>=k)、apex/lift、view、motion、qvel/qacc、KL、EV 和 failure distribution；
9. 先 deterministic CPU/frozen checkpoint，再 bounded GPU、多 seed；
10. 本阶段只能产出 simulation candidate 和 deployment design，不能控制实物。

不要用更大输入维度本身解释性能提升；报告参数量和 matched ablation。
```

## 18. PHASE_3 附加任务 Prompt：实物状态估计与 parity

```text
本次只实现 PHASE_3。
要求 REAL_REPO_IN_SCOPE=YES；若不是 YES，停止并请求授权。
ALLOW_HARDWARE_COMMAND 默认为 NO，因此先完成 no-publish/shadow。

唯一假设：实物 runtime 精确复现训练的 timestamp、BallEstimate、history、mode reset、
actor 和 persistent bounded-q_ref 语义后，才有资格评价策略 Sim2Real，而不会把 adapter gap
误判为 policy/physics gap。

实现要求：
1. 新增独立 versioned runtime profile，不覆盖 simcode5/6 或当前 57-D 路径；
2. BallEstimate 分离 represented time、raw source age、prediction horizon、validity、
   raw detection、in-view 和 reacquisition；
3. 先 checkpoint/hash/dimension/contract fail-closed，再 unpickle/运行 actor；
4. fixed observation 与 recorded trace 的 NumPy/JAX parity；
5. history、hidden state、previous action/dq、q_ref 和 mode reset 原子一致；
6. 记录 raw/clipped/residual action、q_ref、safe target、所有 timestamp 和 safety flags；
7. stale/future/non-finite/invalid/predictor timeout/reacquisition tests；
8. no-publish live shadow、recorded trace replay 和 command-driven mirror 对比；
9. 不部署 recovered RMP、XML PD、output delay 或第二 planner；
10. 未获得 ALLOW_HARDWARE_COMMAND=YES 时，不发送任何机器人运动命令。
```

## 19. PHASE_4 附加任务 Prompt：逐拍 residual 和分级实物验证

```text
本次只实现 PHASE_4。
要求 PHASE_3 parity/shadow 已通过，REAL_REPO_IN_SCOPE=YES。
只有 ALLOW_HARDWARE_COMMAND=YES 且用户现场监督时才可发送机器人命令。

唯一假设：根据上一拍实际 apex/lift、XY drift、hit offset 和 uncertainty 生成的
低维 bounded residual，可以在不改变单一 q_ref -> safety -> robot RMP 边界的前提下，
减小下一拍任务误差并扩大稳定工作区。

实现要求：
1. residual 默认关闭，独立 profile/config/checkpoint contract；
2. 第一版优先手工线性/增益调度，不在线更新完整 actor；
3. 每个 residual 输出声明 SI 单位、hard bound、rate limit、decay 和 reset；
4. invalid/stale/high-uncertainty/safety intervention 时 residual 立即清零；
5. raw actor、residual、combined action、q_ref、safe target 全量记录；
6. 先用已有真实记录做 shadow counterfactual 和符号/延迟检查；
7. 依次执行 no-publish、无球有限工作区、单次 impact、2-hit、3-5-hit、10-hit、
   多球多日长时测试；每一级单独预注册并需用户确认；
8. 评价 paired outcome、finite-step contraction、P(N>=k)、saturation、safety、
   timestamp 和 failure distribution；
9. 任一级出现频繁 limiter 干预、持续饱和或未解释 failure，停止放权并保留日志；
10. 不叠加 recovered RMP、第二 planner、PD、delay 或 inverse compensation。
```

## 20. 后续 Codex 每次交付应回答的问题

1. 本次只改变了哪个因果变量？
2. 哪些历史路径通过了逐值 regression？
3. 新接口的输入、输出、单位、shape、频率、时间戳和 reset 是什么？
4. baseline/candidate 是否使用相同 seed、记录、后端、初始化和样本掩码？
5. 预注册指标是否全部通过；未通过的负结果在哪里？
6. 是否启动 GPU、W&B-online 或实物；其授权和证据是什么？
7. 哪些结论仍只是 simulator candidate，而不是硬件证据？
8. 下一阶段最小、单变量的验证是什么？
