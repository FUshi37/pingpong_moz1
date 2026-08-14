# 实物机器人平台修改 Codex Prompt

把下面整段交给实物仓库中的 Codex。目标是修复已经验证的发布/观测/执行接口问题，并为尚未确认收益的控制候选建立严格实验门，而不是直接在真实机器人上试错。

---

你在以下实物机器人仓库工作：

```text
/home/yangzhe/Project/pingpong_playreal/pingpong-play
```

主要入口：

```text
RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
src/pingpong_moz1/pingpong_controller/tools/interactive_policy_safety_probe.py
src/pingpong_moz1/pingpong_controller/rl_policy.py
src/pingpong_moz1/pingpong_controller/mjx_rl_policy.py
src/pingpong_moz1/pingpong_controller/policy_dq_observation_adapter.py
src/pingpong_moz1/pingpong_controller/high_ball_occlusion_predictor.py
src/pingpong_moz1/pingpong_controller/sim_mirror.py
src/pingpong_moz1/test/
```

根因报告：

```text
/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/record_new3_5_real_failure_rootcause_20260813/RECORD_NEW3_5_REAL_FAILURE_ROOT_CAUSE_REPORT.md
```

## 任务边界

本任务只修改代码、配置、离线 replay 和测试。**不得启动真实机器人 publish、相机、机械臂、训练、JAX 或 GPU。** 不得删除或放宽 joint/EE/workspace/predictive-stop/feedback-stale/lease 等现有安全条件。

只有用户后续明确授权，才可按本文最后的分级实验表在实物上执行。代码实现必须默认 `off` 或 `shadow`；任何 non-finite、过期 feedback、时间倒退、状态未初始化、hash 不匹配都 fail closed 到 measured-pose hold/现有安全路径。

先运行：

```bash
git status --short
git rev-parse HEAD
```

保留用户已有改动，不覆盖无关文件。所有新逻辑必须有单元测试和可旁路开关。

## 已验证事实：不得重新猜测

1. record_new3–5 与仿真都从近静止球放在球拍上启动；不要引入 falling-drop reset。
2. 90 Hz 球观测率已由用户实物实验和保存的 paired sim gate验证，不是失败根因；控制率仍为 200 Hz。
3. v5 checkpoint SHA256：

   ```text
   9d7e94e9ef803fcbe9385ab97485626b2529394be62c75136c69d873adffaa79
   ```

4. v5 XML SHA256：

   ```text
   7d98f2adfdbad6082be0defcec2dbd0cbbcaf1f0fc06ce45ba424b5b3257cc92
   ```

5. 32,006 个历史 policy tick 已重算：67D 顺序、actor forward、action scaling、acceleration→velocity→position 两次积分、previous-action history 都对齐；不要重写 frozen actor。
6. 已确认主因是 command→真实 `q/dq` plant 与 checkpoint plant 不同。same-ball replay 最早 `dq` 分叉 P50=`0.130 s`，其后才是 action/EE 分叉；自主 plant A/B 把 full rate 从 `0.969` 降到 `0.609`。
7. record4–5 的 dq adapter 只改 actor arm-dq，而 racket velocity 仍反映 raw physical motion，导致 observation 内部不一致。它显著改变 action；但 record3 无 adapter 也同样失败，不能把“关闭 adapter”声称为已验证根治。
8. 第二击横飞的直接机制是接触时拍面 `normal_y` 偏正和真实 impact residual 未抵消；exact spin/friction/compliance 参数尚未识别。
9. predictor reacquisition 有 30/85 mm P50/P90 error 和 0.135/0.923 action jump，但没有 paired real A/B，只能作为条件放大器。
10. EE safety 是终止保护，不是首因，必须保留。

## P0：先修复发布可复现性

当前 checkout 的启动脚本默认 profile 为 simcode6，但本地缺少脚本引用的 `simcode5/`、`simcode6/` 资产；完整测试也因缺 checkpoint/trace 出现 error。先解决“运行的到底是什么”，再碰控制逻辑。

### P0.1 Release manifest

新增版本化 manifest，例如：

```text
config/policy_release_manifest.json
```

每个 release 至少包含：

```json
{
  "release_id": "simcode5_v5_recovery",
  "policy_profile": "simcode5_pure_actuator67",
  "checkpoint_path": "...",
  "checkpoint_sha256": "9d7e...a79",
  "policy_xml_path": "...",
  "policy_xml_sha256": "7d98...c92",
  "observation_dim": 67,
  "control_rate_hz": 200.0,
  "deployment_ball_observation_rate_hz": 90.0,
  "training_ball_observation_rate_hz": 60.0,
  "action_semantics": "acceleration_integrated_once_to_dq_once_to_q",
  "reset_mode": "racket_launch"
}
```

要求：

- live 入口按 manifest 解析绝对路径并计算实际 SHA；路径存在但 hash 不同也失败；
- 缺少 release 资产时在启动前报清晰错误，不回退到另一个 profile/checkpoint；
- `PROBE_CONFIG_ONLY=true` 也执行完整合同检查，但不需要 ROS publish；
- metadata 每次记录 repo HEAD、dirty status、manifest 内容/hash、checkpoint/XML hash、实际 CLI/env override；
- 训练 60 Hz 与部署 90 Hz 同时记录，明确它是已验证 override，不把两者不同误报成错误。

### P0.2 让测试可区分“逻辑失败”和“fixture 缺失”

- checkpoint/trace fixture 缺失时应显式 skip 或由 test fixture 注入路径，不能产生 14 个重复 FileNotFound error；
- 修复 `test_simcode5_cpu_xml_restores_checkpoint_sport_pd_profile` 与 `simcode2_xml.py` 的 API 漂移；
- 添加 manifest hash mismatch、missing asset、profile mismatch、dirty repo metadata 单测；
- 不要为了让测试绿而复制未知 checkpoint 或修改期望 hash。

P0 验收：config-only 输出唯一 release 身份；错误路径/错误 hash/错误 profile 都在任何 ROS 初始化前失败。

## P1：建立单一、原子的 joint observation contract

定义不可变结构，例如：

```text
PolicyJointSample
  q_raw[7]
  dq_raw[7]
  source_stamp_s
  source_sequence
  receipt_age_s
  repeated_source_sample
  velocity_source
```

每个 actor tick 只捕获一次，随后所有字段从这一份 sample 构造：

- arm q/dq；
- FK racket position/orientation；
- Jacobian `J(q)dq` linear/angular velocity；
- command error 与 active delayed command error；
- contact phase；
- safety 检查与 matcher feedback。

不得在 inference 中途从异步 callback 读取一半新 q、一半旧 dq。重复 source stamp 时按 source sample 做 ZOH，不能按 200 Hz loop 人造零/尖峰。

### P1.1 明确 dq 模式，不暗改输入

实现显式 enum：

```text
POLICY_DQ_MODE=raw_coherent|legacy_dq_only_ablation
```

`raw_coherent`：

- actor arm-dq 使用 raw dq；
- racket velocity 使用同一 raw q/dq；
- safety、plant matcher、日志均使用 raw q/dq；
- `POLICY_DQ_OBSERVATION_SCALE=1.0`。

`legacy_dq_only_ablation`：

- 精确复现 record4–5 的 gain/tau/delay，只为 paired ablation；
- metadata 必须写 `observation_coherence=incoherent_legacy_ablation`；
- UI/CLI 明示不能称为 plant alignment；
- live 使用必须同时满足 `ALLOW_INCOHERENT_DQ_ABLATION=true` 和实验 impact 上限；
- raw dq 永远保留给 safety、FK 物理诊断和 plant fit。

不要在没有 paired real A/B 的情况下，在提交说明里声称 raw 模式或 adapter 模式已经提高实物长跑。代码默认选择应由 release manifest 明确，而不是 profile 名的隐式 case；新 release 推荐 `raw_coherent`，旧 release 的历史复现配置可保留 legacy 模式。

### P1.2 单元与 replay 验收

- 同一个 source seq 上，actor q/dq/FK/Jdq/command error 的 stamp 必须完全相同；
- raw 模式中 `racket_velocity - J(q)dq_raw` 只允许数值/已声明 sample-clock差；
- legacy 模式必须能重放 record4–5 历史 actor input/action；
- 在保存记录上重新计算 actor action，P99 RMS `<1e-5`；
- 非连续 seq、时间倒退、feedback age 超限触发现有安全 hold。

## P2：修复 `simcode5_direct` 的错误 plant 假设

`simcode5_direct` 当前把 integrated policy angle 直接发给硬件，隐含“硬件自然等于 checkpoint plant”。数据已经推翻该假设。不要简单让现有 inverse controller 追理想 `policy_qcmd`；训练 actor 实际看到的是：

```text
policy_qcmd → checkpoint delay/second-order plant → q_target,dq_target
```

正确的 model-reference 目标是：

```text
choose q_publish so that real hardware output follows q_target,dq_target
```

### P2.1 模块与模式

新增独立模块，例如：

```text
policy_plant_model_reference.py
```

接口：

```text
POLICY_PLANT_MATCHER_MODE=off|shadow|apply
POLICY_PLANT_MATCHER_REAL_CONFIG=<versioned json>
ALLOW_POLICY_PLANT_MATCHER_APPLY=false
```

- `off`：完全复现现行为；
- `shadow`：运行 target/real model 和求解器，只记录建议 command，发布原 command；
- `apply`：只有显式 allow、release hash 匹配、feedback fresh 且所有安全门正常时发布 matcher command。

默认 `shadow`。禁止把 apply 隐藏在 `COMPENSATION_PRESET` 中自动打开。

### P2.2 参数来源

target plant 从实际加载 checkpoint `env_cfg` 解析，不按 profile 名硬编码。测试用 v5 参考：

```text
target wn = [20.363849,22.655975,22.650471,21.730533,19.663364,22.815526,23.645817]
target zeta = [0.391768,0.366169,0.345738,0.346000,0.347896,0.345814,0.380713]
target gain = [0.997884,0.996612,0.992312,0.990942,0.982327,0.992695,0.983194]
target delay_ms = [45,50,45,40,35,45,55]
```

real plant 配置必须有 source recording/hash/fit version/holdout metrics。当前仅是候选中心：

```text
real wn = [21.891138,22.089575,22.650471,21.730533,20.154948,22.245138,23.054672]
real zeta = [0.333003,0.329552,0.311164,0.311400,0.313106,0.311233,0.285535]
real gain = [0.997884,0.996612,0.992312,0.990942,0.982327,0.992695,0.983194]
real delay_ms = [45,50,45,40,35,45,50]
```

holdout 仍低估 real dq/qacc，求解器必须对 model residual 保守，不能假定模型精确。

### P2.3 离散与安全语义

1. target model 复用/逐位对照训练环境 per-joint delay + second-order 离散方程、delay rounding、reset 队列；不另造 FOPDT。
2. real predictor 使用版本化 fit；未来未知 actor command 只能 hold-last，不能偷看未来。
3. 可使用小型有界 receding-horizon/LQR，目标包含 q/dq target error、command deviation、command slew；只应用第一步。
4. target delay 只在 target model 中；不得再给真实 publish 串一个 sleep/queue，防止双重延迟。
5. matcher correction、correction-rate、q/dq/ddq 均有逐关节上限，且不得宽于现有 safe publish envelope。
6. matcher 后仍经过全部现有安全限制。任何异常进入 measured-pose hold，不静默回退到未经审计的 command。
7. actor 的 previous action/history 保持原语义；另记 `policy_qcmd`、`matcher_shadow_q`、`pre_safety_q`、`published_q`，不能把 matcher correction 伪装成 actor action。

### P2.4 验收门

纯离线：

- target model 对 frozen fixture 与训练环境逐 tick一致；
- off 模式 bitwise/数值等价旧路径；
- reset、seq gap、时间倒退、non-finite、feedback stale 测试；
- 用保存 command replay，shadow prediction 必须相对 direct 降低 target q/dq error；结果按 calibration/holdout 分开，不得只报训练集；
- 最坏 correction、slew、预测 q/dq 不越现有 envelope。

这只证明实现和离线目标正确，不证明真实闭环收益。实物 apply 必须走末尾 staged gate。

## P3：保持 90 Hz，隔离 predictor handover 实验

不要改变 90 Hz 和 predictor 的基本功能。为 `_apply_policy_ball_sampling_contract` 增加明确策略：

```text
POLICY_BALL_HANDOVER_MODE=legacy_force_fresh|strict_fractional_continuous
```

`strict_fractional_continuous`：

- predictor→camera/impact reset 不能绕过 90 Hz fractional refresh；
- 在合法 refresh tick 对 position/velocity innovation 做有界、有限时长 blend；
- contact/impact guard 清理旧 velocity history，但不能制造与 position 不一致的速度；
- 记录 source timestamp、fresh/held/predicted/reacquired、scheduled refresh、forced refresh、innovation、blend、actor action jump；
- invalid/non-finite/超 gate 维持 predictor 或按现有 fail-closed 逻辑处理，不瞬间注入异常 measured state。

添加 frozen replay paired 比较：两模式收到完全相同记录输入，报告 observation jump、action jump、第二击前累计差。没有真实 paired A/B 前不把新模式称为根因修复，也不删除 legacy mode。

## P4：补齐最小接触可识别日志

现有记录无法唯一分解 spin/friction/compliance。不要伪造 ball spin observation。每个推定 contact 前后至少保留：

```text
source stamp/seq
raw q/dq
racket position/quaternion/face normal
racket linear/angular velocity
fresh measured ball position/velocity and estimator/predictor state
contact offset estimate
policy raw/clipped action and all command layers
```

若硬件已有 wrist force/torque 或高速相机接口，只实现时间同步与可选记录；无硬件时不要在代码中生成虚构值。contact detector 只标记 `inferred_contact`。

## P5：测试与交付

至少通过：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  src/pingpong_moz1/test/test_interactive_probe_sim_alignment.py \
  src/pingpong_moz1/test/test_high_ball_occlusion_predictor.py \
  <新增 release/joint-sample/plant-matcher/replay tests>
```

并执行不触发机器人/相机的 config-only 检查。若仓库没有 ROS build/install 环境，测试必须能够注入 mock/fixture，不得靠启动真实节点补齐。

交付：

- 修改文件和原因；
- release manifest 与实际 hash；
- off/shadow 的离线 replay 报告；
- 所有测试结果、skip 原因、未解决风险；
- 明确哪些是“代码已验证”、哪些仍需“实物 A/B”。

## 后续实物 staged gate（本任务不执行）

只有用户明确授权后：

1. config-only：无 ROS publish；
2. matcher shadow + raw coherent：机械臂仍走旧 command，仅记录 3 次短序列；
3. measured-pose/低幅空载动态验证：确认方向、单位、delay、correction bound；
4. 单击 ball-on-racket：每个候选/对照随机交替，预注册至少 10 对，立即停止；
5. 3-hit 上限 paired A/B；
6. 通过 plant tracking、第二击 `normal_y/vout_y`、安全指标后才允许长 bout。

每一级都保留旧路径作为对照；禁止用日期前后不配对的 bout 得出因果结论。

---
