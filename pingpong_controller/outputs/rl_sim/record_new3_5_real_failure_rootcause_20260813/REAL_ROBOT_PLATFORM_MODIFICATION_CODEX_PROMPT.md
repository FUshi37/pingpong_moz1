# 实物机器人平台修改 Codex Prompt

将下面整段内容作为一个新任务交给机器人平台上的 Codex。该任务只修改实物部署仓库，不负责重训策略。

---

你是运行在真实机器人平台上的 Codex。请在以下仓库内完成实物颠球部署、观测一致性、接触诊断和分级试验保护的修改：

```text
/home/yangzhe/Project/pingpong_playreal/pingpong-play
```

主要启动入口：

```text
./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
```

主要代码：

```text
RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
src/pingpong_moz1/pingpong_controller/tools/interactive_policy_safety_probe.py
src/pingpong_moz1/pingpong_controller/mjx_rl_policy.py
src/pingpong_moz1/pingpong_controller/rl_policy.py
src/pingpong_moz1/pingpong_controller/policy_dq_observation_adapter.py
src/pingpong_moz1/pingpong_controller/high_ball_occlusion_predictor.py
src/pingpong_moz1/test/test_interactive_probe_sim_alignment.py
```

## 任务背景

record_new3-5 的失败不是一个单原因问题，而是至少三个根缺口和两个反馈放大器耦合：

1. 训练 reset/隐藏接触状态与实物 impact transition 不一致。旧 checkpoint 使用 `racket_launch`，计数前已有多段拍球接触；actor 不观测球自旋，训练也没有独立随机化初始自旋和归一化转动惯量。实物从未知自旋的下落球首击开始。该问题主要应在仿真训练端通过 latent-state domain randomization/鲁棒训练解决，不能在实物端伪造一个 spin observation。
2. 真实执行器、q/dq、命令状态和 previous-action/history 转移与仿真 plant 不一致。record_new3 的同球输入反事实中，`dq` 差异约在 0.13 s 出现，早于第一次接触；随后 action、命令历史、球拍速度/位置逐步分离。
3. 高球 predicted -> measured 重捕获是旧 checkpoint 未训练的 observation transition。49 个首击到第二击周期中，55.1% 使用预测，53.1% 有重捕获；重捕获误差 P50/P90 约 30/85 mm，动作跳变范数 P50/P90 约 0.135/0.923。
4. previous action、command history、acceleration 双积分以及饱和会放大前述偏差，但不是初始根因。
5. EE safety box 是安全终止端点，不是颠球失败根因。保持现有 EE box 和 fail-closed 行为，不扩大 box，也不要把 box 干预伪装成性能修复。

完整诊断依据位于：

```text
/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/record_new3_5_real_failure_rootcause_20260813/RECORD_NEW3_5_REAL_FAILURE_ROOT_CAUSE_REPORT.md
/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/record_new3_5_real_failure_rootcause_20260813/causal_chain_diagnostic.json
/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/record_new3_5_real_failure_rootcause_20260813/second_hit_impulse_decomposition.csv
```

如果机器人平台上没有这些路径，以本 prompt 的合同和验收标准为准，不要因此阻塞代码工作。

## 总目标

不要尝试通过新的实物观测滤波“修好旧 checkpoint”。请把实物平台改成：

- actor 输入、原始传感器、FK、命令状态和时间戳的关系可以被逐 tick 审计；
- 能区分真实接触候选时刻和延迟的 hit-credit/`new_hit`；
- 每次 trial 的 checkpoint、XML、profile、动作语义、有效配置均冻结并可复现；
- predicted/held/reacquired 状态切换完全可重放，异常重捕获 fail closed；
- 默认只允许单击诊断，达到接触数/时长预算后自动停止发布并保留碰撞后记录窗口；
- 不改宽 EE safety box，不自动运行机器人，不启动训练或 GPU 仿真。

## 开始工作前的约束

1. 先读取 `git status --short`、当前 commit、启动脚本和上述主要文件。保留用户已有改动，不覆盖无关内容。
2. 不运行 `RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh` 的真实发布模式，不发布 ROS 机器人命令，不启动相机 GPU 推理，不移动机械臂。
3. 只允许运行静态检查、单元测试、纯 NumPy/MuJoCo 测试和 `PROBE_CONFIG_ONLY=true` 配置检查。
4. 不删除或放宽任何安全条件；不把 `DONE_EE_*` 加进 actor observation；不默认启用未经训练的 command/contact shield。
5. 不修改 checkpoint 文件。checkpoint 的 `env_cfg` 是动作/observation 合同来源；profile 名称和 tensor shape 相同不足以证明合同一致。

## P0：冻结并校验每次实物试验合同

实现一个小而清晰的 experiment contract 层，可以新建模块，例如：

```text
src/pingpong_moz1/pingpong_controller/real_experiment_contract.py
```

要求：

1. 每次 recording 开始时生成不可变合同，至少包含：

   - 实物仓库 git commit、是否 dirty；
   - policy profile；
   - checkpoint 绝对路径和 SHA-256；
   - policy XML、robot XML 的绝对路径和 SHA-256；
   - checkpoint `env_cfg` 中所有影响 actor I/O、action integration、ball sampling、delay/history、actuator 和 observation 的字段；
   - `_policy_runtime_contract()` 的完整结果；
   - predictor/reacquisition/相机速度估计器的有效配置；
   - dq adapter、observation scale、compensation、servo planner、contact shield、workspace safety 的有效配置；
   - 控制频率、相机实际采集频率、policy ball refresh 频率；
   - 一个由上述白名单配置生成的 `experimentContractSha256`。

2. 合同写入 `metadata.json`，同时每条 `control_samples.jsonl` 写入短合同 ID。不要把整个环境变量或潜在密钥写入记录，只记录影响本试验的白名单字段。
3. recording 已开始或 `publish_commands=true` 时，禁止热切换 profile、checkpoint、observation scale、dq adapter、predictor、compensation、shield 和动作语义；请求改变时 fail closed、保持实测姿态并返回明确错误。停止 recording 并解除发布后才允许建立新合同。
4. 加入以下 fail-closed 校验：

   - checkpoint obs/action shape 不匹配；
   - qvel checkpoint 没有走 `action_command_mode=velocity` 或 `positionIntegrationCount != 1`；
   - acceleration checkpoint 被误当成 qvel；
   - simcode6 使用了 simcode5 dq adapter；
   - live publish 使用非 1.0 observation scale，除非显式设置 `ALLOW_DIAGNOSTIC_OBS_ABLATION=true` 且最大接触数为 1；
   - 旧 simcode5 live publish 默认拒绝，只有显式 `ALLOW_LEGACY_SIMCODE5_REAL=true` 且最大接触数为 1 时允许诊断。

5. 保留并扩展当前 simcode6 的 contract tests，不能仅根据类名硬编码“通过”。

## P1：消除或显式隔离 proprio observation 内部不一致

当前 simcode5 dq adapter 只改 `obs[7:14]`，但 racket velocity、FK、command error、安全和真实 plant 仍来自原始状态。不要继续把它称作 sim-real 对齐。

实现以下语义：

1. 定义一个每个 actor tick 唯一的原子 joint sample，包含：

```text
q_raw[7]
dq_raw[7]
source_stamp_s
source_dt_s
receipt_age_s
source_sequence（若 ROS 消息没有则在桥接层生成）
repeated_sample
velocity_source
```

2. corrected 67D profile 的 q、racket FK、sample-clock racket velocity、command error 和 contact phase 必须可追溯到同一个 q sample。重复 JointState 时间戳时继续 zero-order hold 上一个 source-sample racket velocity，不得在 200 Hz tick 上产生“先归零再尖峰”。
3. `dq_raw` 始终原样保留用于安全、FK/Jacobian 诊断和日志。新 profile/simcode6 的 actor dq 默认就是该原始、同时间戳 dq；禁止套用 simcode5 adapter。
4. simcode5 adapter 仅作为 legacy 单击 ablation：

   - 默认 live publish 禁用；
   - 若显式允许，metadata 必须标记 `observationCoherence="incoherent_legacy_dq_ablation"`；
   - 同时记录 `dq_raw`、`dq_actor`、`J(q)dq_raw`、`J(q)dq_actor` 和 actor 使用的 sample stamp；
   - 不要为了“凑一致”再私自滤 racket velocity 或 command error，因为这会创造另一个没有训练过的 observation contract。

5. 每个 actor tick 新增 `policyJointSample` 和 `racketStateAtActorInput`：

```text
position_base_m[3]
quat_xyzw[4]
normal_base[3]
linear_velocity_sample_diff_m_s[3]
linear_velocity_Jdq_raw_m_s[3]
linear_velocity_Jdq_actor_m_s[3]
angular_velocity_Jrot_dq_raw_rad_s[3]
fk_ok
sample_stamp_s
```

这些字段必须来自 actor inference 前冻结的 sample，不能被 inference 后到达的新 ROS callback 覆盖。

## P2：增加“物理接触候选”记录，禁止用 new_hit 代替接触时刻

实现 telemetry-only 的 `ImpactEventMonitor`，建议放在独立模块并单元测试。它不参与 actor observation，不改变命令。

要求：

1. 只使用原始相机 measurement 时间戳、原始球位置序列、相机 impact velocity-reset 诊断以及同时间戳附近的真实球拍状态；predictor 输出不得作为物理接触证据。
2. 接触候选至少同时考虑：下降到上升或速度跳变、ball-racket surface gap、横向接触距离、refractory time 和数据有效性。输出 `confidence`、每个 gate 的值和 rejection reason，不把估计事件伪装成力传感器真值。
3. 每个高置信事件分配单调 `impactEventId`。`new_hit`/hit credit 若存在，作为独立字段记录其确认延迟，绝不能覆盖 physical contact timestamp。
4. 维护至少接触前 200 ms、后 250 ms 的 ring buffer。后窗口完成后写一条 `impact_events.jsonl`，包含：

   - raw camera position samples/timestamps 和 OLS incoming/outgoing velocity；
   - impact candidate timestamp、sourceSeq、confidence/gates；
   - actor-input q/dq、球拍位置、quat、normal、线速度和角速度；
   - 球心到拍面中心偏移、估计拍面内 contact offset；
   - 球拍接触点速度 `v_center + omega_racket × offset`；
   - raw actor action、executed actor action、raw integrated command、shielded command、published command；
   - ball observation state（fresh/held/predicted/reacquiring）；
   - 若后续出现 hit credit，记录 `creditDelayS`。

5. `control_samples.jsonl` schema version 加一，并在逐 tick 数据中包含当前 `impactEventId`/phase（pre/contact/post/none）。保持旧分析脚本可读取已有字段。

## P3：将预测/重捕获变成可版本化、可重放、异常时 fail-closed 的状态机

不要再增加新的随意滤波器。整理现有 high-ball predictor、camera reacquisition smoothing 和 ball sampling contract，使其状态显式且可离线逐位重放。

1. 每 tick 输出一个明确枚举：

```text
fresh_measurement
held_measurement
predicted
reacquisition_warmup
reacquisition_blend
reacquisition_rejected
invalid
```

2. 同时记录：raw measured 6D、predictor pre-handover 6D、policy-selected 6D、measurement/policy timestamp、innovation position/velocity、gate、blend alpha、warmup sample count和状态机版本/hash。
3. 保持 `obs[20:26]` 维数和 frozen checkpoint 合同不变；不要加入一个实物独有的“预测标志”到旧 actor observation。
4. 对过大的重捕获创新实现配置化 fail-closed gate：

   - 默认只在单击诊断中使用保守阈值，初始 `MAX_REACQUISITION_INNOVATION_M=0.10`，但把它作为可由真实数据校准的安全阈值而不是性能参数；
   - 超阈值或速度 warmup 未完成时不得把测量瞬时 snap 给 actor；保持上一可信预测只用于记录，live command 路径进入 measured-pose hold/结束 trial；
   - 记录 raw measurement，不能因为 reject 丢掉诊断数据；
   - 连续多少帧、hold 行为和解除条件必须有单元测试。

5. 在重捕获 tick 做无副作用 counterfactual：用相同的 q/history 分别构造 predictor-before 和 measurement-after 两个 observation，只调用纯 actor forward，不更新 controller state，记录两者动作差 `reacquisitionActionCounterfactualNorm`。绝不能为了记录而推进 previous action 或 command history 两次。
6. 提供一个离线 replay 测试：把一段保存的 raw measurement/timestamp 输入状态机，逐 tick 重建 policy-selected 6D，结果与 JSONL 完全一致。

## P4：默认单击、再三击的试验预算和自动停止

在现有 `policyStepLimit` 基础上增加物理接触和时间预算：

```text
REAL_EXPERIMENT_MAX_IMPACTS=1
REAL_EXPERIMENT_MAX_POLICY_S=3.0
REAL_EXPERIMENT_POST_IMPACT_CAPTURE_S=0.25
ALLOW_OPEN_ENDED_REAL_POLICY=false
```

要求：

1. `RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh` 默认 max impacts 为 1。只有显式设置 `ALLOW_OPEN_ENDED_REAL_POLICY=true` 才允许 0 表示不限；legacy simcode5 即使显式允许 open-ended 也必须拒绝。
2. 计数只能使用 P2 的高置信 physical impact candidate，不使用 hit-credit。
3. 达到 impact budget 后：

   - 不再允许新的策略命令进入 publish 路径；
   - 在 `POST_IMPACT_CAPTURE_S` 内继续写相机、joint、policy 和 impact 后窗口；
   - 使用现有 measured-pose hold/command lease 失效语义安全停止，不突然跳到 reset pose；
   - 完成窗口后 freeze recording，并记录停止原因 `experiment_impact_budget_reached`。

4. 达到 policy duration budget、joint feedback stale、合同改变、重捕获 fail-closed 或记录队列严重丢包时执行相同停止流程。
5. max impacts 从 1 调到 3 必须是一次新的 experiment contract；3 调到不限也一样。UI 中不得在 live publish 期间直接修改。

## P5：补齐 plant/history 记录，不在实物端继续做 blind dq filtering

确保每个 policy tick 都能重建以下状态转移：

```text
raw q/dq/ddq + source timestamp
actor q/dq
previous raw/executed action
command error
active delayed command error
arm_cmd_qvel
raw integrated q command
post-shield q/dq command
post-compensation command
published safe position
下一原始 q/dq sample
```

要求：

1. 当前已有字段继续保留；缺失的 exact actor-time 和 source time 字段补齐。
2. 记录 `actionCommandMode`、`positionIntegrationCount`、clip/filter/slew 前后 action 和每关节 limiter 命中情况。
3. 增加一个纯离线 exporter，从 recording 生成 actuator-identification CSV；不要在本任务中自动执行任何激励轨迹。
4. 如果实现安全 actuator characterization 工具，必须独立入口、默认只生成计划、需要双重显式开关才允许 ROS publish，并且本任务绝不能实际运行它。

## P6：contact shield 只增加 shadow 模式，不把它默认当修复

现有 contact/vertical energy shield 会改变 command/history，而旧策略未按该 intervention 训练。把布尔开关扩展为：

```text
CONTACT_ACTION_SHIELD_MODE=off|shadow|apply
```

要求：

1. 默认 `off`。`shadow` 计算建议的拍面法向/横向速度/垂直能量约束及 projected command，但不改变 controller 或发布命令；完整记录 counterfactual。
2. `apply` 必须同时满足：显式 `ALLOW_UNTRAINED_CONTACT_SHIELD=true`、max impacts=1、experiment contract 记录该状态。否则 fail closed。
3. 不要宣称 shield 修复了自旋。它只是在接触前降低横向相对速度、拍面倾斜和冲量敏感性，是待验证的鲁棒化控制手段。
4. 若 apply 修改 command，清楚区分 actor executed action 和 external shield executed command；command history 已回写仍不等于 actor `previous_action` 与训练一致。metadata 必须标记这一 residual mismatch。

## P7：测试与验收

优先扩展现有：

```text
src/pingpong_moz1/test/test_interactive_probe_sim_alignment.py
```

必要时增加更小的单元测试文件。至少覆盖：

1. simcode6 必须是 velocity/单积分，且永不使用 simcode5 dq adapter。
2. acceleration checkpoint 不能误走 velocity，反之亦然。
3. 重复 JointState source stamp 时 q/FK/racket velocity 不出现 200 Hz 假尖峰。
4. `policyJointSample`、actor q/dq、racket state 使用同一个 source stamp；异步 callback 不会覆盖 trace。
5. recording metadata 的 model/XML/config hash 稳定，合同改变时 live publish 被拒绝。
6. `ImpactEventMonitor` 对合成下降-碰撞-上升轨迹只报一个事件；不依赖 `new_hit`；无近拍/无速度跳变时不报。
7. impact budget 达到后不再发布新命令，但继续记录 post-impact window。
8. large reacquisition innovation 进入 reject/hold，并保存 raw measurement；counterfactual actor forward 不改变 controller history。
9. predictor raw replay 能逐位重建 policy-selected ball 6D。
10. shield shadow 不改变命令；apply 未授权时 fail closed。
11. 现有 workspace safety、predictive stop、ball-lost failsafe 和 profile resolution tests 全部继续通过。

运行测试时限制线程，不访问 GPU，例如：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -m unittest src/pingpong_moz1/test/test_interactive_probe_sim_alignment.py
```

根据仓库实际测试入口调整命令，但不得为了测试启动真实 ROS publish。最后可运行：

```bash
PROBE_CONFIG_ONLY=true ROS_PUBLISH_COMMANDS=false \
ENABLE_CAMERA_BALL=false ENABLE_MOCAP_BALL=false ENABLE_SIM_MIRROR=false \
./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
```

确认配置打印正常，仍不能启动真实控制。

## 交付物

完成代码和测试后，在仓库内新增：

```text
REAL_ROBOT_CAUSAL_DIAGNOSTIC_CHANGES.md
```

内容包括：

- 修改文件和新模块；
- 新环境变量及安全默认值；
- observation/action/recording schema 变化；
- 单元测试命令与结果；
- 未运行机器人/未发布命令的明确声明；
- 仍需训练端解决的项目：falling-contact reset、初始自旋/惯量/contact DR、latent-state robust policy、exact predictor state-machine training、identified plant residual DR；
- 推荐的实物放权顺序，但不要自行执行：

```text
配置/record dry-run
-> publish=false 的手持球/遮挡 replay
-> max impacts=1 的单击试验
-> 单击 transition 与 exact-checkpoint 完整仿真分布对齐
-> max impacts=3
-> 只有新 checkpoint 的 blocking validate 通过后才允许开放式 6 s
```

不要使用旧 simcode5 clean validate 的固定 apex/速度阈值验收新 simcode6 checkpoint。每个待部署 checkpoint 必须使用它自己的本机完整 autonomous validate 生成 blocking 分布和 promotion gates。

工作过程中采用小步修改：先合同/记录，再 impact monitor，再试验预算，再 predictor replay，最后 shadow shield。不要把这些功能和性能调参混成一个无法归因的大改动。

---

## 这份 prompt 不要求做的事情

- 不给 frozen actor 增加 ball-spin observation；
- 不从球心平动轨迹伪造一个“精确自旋”；
- 不在实物端扩大 EE box；
- 不默认启用 contact shield、dq adapter 或新的 ball filter；
- 不启动训练、JAX、GPU sim mirror 或真实机械臂；
- 不以 record mirror 代替完整本机仿真 validation。
