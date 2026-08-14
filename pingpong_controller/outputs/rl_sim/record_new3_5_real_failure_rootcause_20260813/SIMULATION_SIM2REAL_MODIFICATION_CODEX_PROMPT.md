# 仿真与 Sim2Real 训练修改 Codex Prompt

把下面整段交给仿真仓库中的 Codex。目标是先把已经测得的真实 plant/contact/observation transition 作为可复现验证轴，再决定是否训练；禁止用新的 reward 调参掩盖旧模型的 sim-real gap。

---

你在以下仿真仓库工作：

```text
/home/yangzhe/Project/pingpong_controller
```

根因报告与证据目录：

```text
pingpong_controller/outputs/rl_sim/record_new3_5_real_failure_rootcause_20260813/
```

重点代码通常包括：

```text
pingpong_controller/tools/rl_sim/mjx_juggle_env.py
pingpong_controller/tools/rl_sim/train_juggle_mjx_curriculum.py
pingpong_controller/tools/rl_sim/validate_juggle_mjx.py
pingpong_controller/tools/rl_2real/mjx_policy_controller.py
```

## 任务边界与算力安全

先运行 `git status --short`，保留全部用户改动。先做 CPU-only 静态/回放/小矩阵检查；只有用户明确允许且 GPU 空闲才做 MJX smoke/training。

GPU 规则：

- 先 `nvidia-smi`，不得占用正在 99% 利用率的设备；
- 明确指定空闲 GPU，设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`；
- smoke 从 `n_envs<=16`、`n_steps<=32`、1 update 开始；
- smoke 不过不启动长训练；每个 run 有独立输出目录和 manifest；
- 不并发启动多个训练，不无限重试 OOM/compile hang；
- 本次根因审计中 GPU0 已繁忙，沙箱又无法访问 GPU1，clean-69 CPU 编译超过约 2 分钟后已主动终止。不要把“没有新 clean-69 rollout”写成成功复现。

## 已验证事实：不得反向修改

1. v5 checkpoint SHA256：

   ```text
   9d7e94e9ef803fcbe9385ab97485626b2529394be62c75136c69d873adffaa79
   ```

2. XML SHA256：

   ```text
   7d98f2adfdbad6082be0defcec2dbd0cbbcaf1f0fc06ce45ba424b5b3257cc92
   ```

3. checkpoint 与 XML 分别和 commits `4f47e6d3...`、`69b9bf344...` 中对象一致；但当前 `mjx_juggle_env.py` 相对 commit 69 已大幅变化。所有结果必须标明使用 clean-69、当前 worktree 还是另一个 snapshot。
4. 实物和仿真都从近静止球位于拍面上启动。保留 `racket_launch` 主 reset，不实现 falling drop 来“对齐实物”。
5. actor/67D/action integration 已逐 tick 验证，无需改网络 I/O 或重新解释 action。
6. 已确认主因是 actuator plant：same-ball replay 最早在 dq 分叉；当前完整环境中只替换 plant，full rate `0.969→0.609`、hit-vxy `0.101→0.155 m/s`。
7. real plant 候选在 holdout 上优于 nominal，但仍低估 real dq/qacc，因此它是 DR center/candidate，不是完美真值。
8. real first-apex P50/P90=`1.443/1.550 m`，clean reference=`1.277/1.337 m`；第二击失败 `normal_y` 和法向横向投影明显更正。real impact transition 未被旧训练覆盖。
9. 3.7 g 球质量越出旧 mass DR，但 selected-plant 上固定 3.7 g 并没有继续恶化，因此不能只扩大质量范围就宣布完成。
10. spin/friction/compliance/restitution/精确接触点没有被唯一识别。spin sweep 只证明它是充分机制之一。
11. record4–5 dq-only adapter 产生 observation 内部不一致；新策略应优先消费可测 raw q/dq，而不是学习依赖实物独有 adapter。
12. 90 Hz 已验证，不是根因；predictor/reacquisition 是需要覆盖的状态转移，而不是理由去降频。

## P0：建立不可混淆的 baseline

### P0.1 三种源码身份

建立 run manifest，至少写入：

```text
git commit
git status --short
checkpoint/XML path + sha256
mjx_juggle_env.py sha256
validator sha256
all CLI args and effective env_cfg
JAX/MuJoCo versions, backend/device
seed, env count, episode selection rule
```

结果目录名和报告必须明确：

- `clean_commit69_reproduction`：只能使用 commit 69 archive；
- `current_validator_counterfactual`：当前 worktree，同一代码内 paired A/B；
- `new_sim2real_candidate`：本任务新增模型。

禁止把 current-validator A/B 写成 clean-69 复现。

### P0.2 固定 baseline suite

在不改 policy weights 的情况下，实现统一 validator matrix。每个 env 只取第一个 episode，deterministic action，固定 seed 列表，输出逐 episode CSV 与 aggregate JSON。至少包含：

```text
A nominal checkpoint plant + checkpoint contact
B selected real plant center + checkpoint contact
C selected real plant residual-tail variants + checkpoint contact
D nominal plant + real-impact outcome/tail variants
E selected plant + impact variants
F selected plant + deployment observation state machine
G selected plant + impact + observation combined
```

先重现现有 A/B 数值方向；不能重现时停止训练，定位版本/配置差。

## P1：逐关节 actuator plant 与 residual DR

当前 selected center：

```text
wn(rad/s) = [21.8911379437,22.0895753812,22.6504705,21.730533,
             20.1549483562,22.2451380938,23.0546718187]
zeta      = [0.3330028,0.3295521,0.3111642,0.3114,
             0.3131064,0.3112326,0.28553475]
gain      = [0.99788351,0.9966119,0.99231151,0.99094239,
             0.982326685,0.992694585,0.983194325]
delay(ms) = [45,50,45,40,35,45,50]
```

要求：

1. 新增版本化 plant profile，写明 source recording、fit script/hash、calibration/holdout recording 列表和 metrics。
2. 每关节独立随机化 `wn/zeta/gain/delay`，支持相关 residual，而不是一个全局 multiplier。
3. DR 宽度从 holdout command-replay residual 和 bootstrap/recording variation 推导；如果缺 confidence interval，先实现估计脚本，不能拍脑袋给范围。
4. 训练环境、CPU/MJX validator 和部署 mirror 使用同一 plant implementation、delay rounding、reset queue semantics。
5. 增加 exact-command replay regression test：nominal 与 selected 在固定 fixture 上逐 tick稳定，selected holdout q/dq/qacc metric不劣于现有 `0.381°/0.219 rad/s/4.404 rad/s²`。
6. raw simulated q/dq 直接进入 actor observation；不要在训练端复制 record4–5 的 dq-only adapter。

## P2：用可观测 outcome 约束 impact randomization

不能从现有日志唯一确定“真实摩擦=某个数”。因此把 impact 部分分成两层：

### P2.1 物理 latent sweep

支持独立随机化：

- ball mass，覆盖实物 3.7 g；
- ball initial/angular spin；
- normalized inertia `I/(m r²)`；
- ball/racket friction；
- normal contact solref/time/damping；
- racket radius/normal/pose small calibration residual；
- 可选 compliant-layer/contact-duration proxy。

范围必须来自测量、制造规格或预注册 sensitivity sweep。未知参数可以作为 latent nuisance 取覆盖范围，但报告必须写“robustness range”，不能写“real fitted truth”。

### P2.2 outcome-matched gate

使用 real observable 作为约束，而不是只拟合参数：

```text
first contact incoming/outgoing vz
first post-contact apex distribution
second-contact incoming/outgoing vy/vz
racket face normal and linear/angular velocity at contact
contact-center offset
normal-projection/residual-y distribution
```

已知稳健目标：

- real first-apex P50/P90 约 `1.443/1.550 m`；
- real second outgoing vy P50 约 `+0.216 m/s`（detector 变体约 `+0.191…+0.229`）；
- 0–3 hit 第二击 normal projection P50 约 `+0.444 m/s`；
- 8+ hit 小样本的 outgoing vy P50 约 `-0.052 m/s`，说明恢复分支也必须被覆盖。

不要只把训练分布推向失败正 `vy`；目标是让 policy 在 real observed transition envelope 内都能恢复。对每个候选 impact profile保存 matched-state/sensitivity 结果，并明确 position-derived velocity robustness。

## P3：补齐 post-impact recovery reset/curriculum

旧 clean 闭环主要在自洽 attractor 内训练。增加以下 curriculum bucket，但保留 ball-on-racket 主任务：

1. 首击后高 apex/re-entry：从 real first-apex 分位数采样；
2. 第二击前 lateral incoming state：覆盖 real `vy` 和 relative x/y/z；
3. 拍面姿态/竖直速度相位 perturbation：覆盖 real second-contact `normal_y` 与 racket `vz`；
4. previous-action/command-history warm state：从 real/sim reachable history sample 初始化，不能随机拼接不可能组合；
5. contact residual tail：包含能把球推向正/负 y 的 latent impact 分支；
6. 恢复成功 reward/gate：优先下一次可恢复 contact、低 `|vxy|`、拍面法向回正，而不是只奖励一次击中。

所有 bucket 写入 episode metadata；validation 按 bucket 报 worst-bin/CVaR，不能只报平均值。

## P4：复现部署 observation state machine

实现 200 Hz policy/control 下的 90 Hz fractional ball refresh，至少支持：

```text
fresh
held
predicted_high_ball
reacquisition_handover
impact_reset
invalid/lost
```

要求：

- 同实物使用 timestamp/seq 语义；
- checkpoint 的 `obs[49]` age contract保持原定义，不因模拟 state machine 擅自改变网络维度；
- predicted→measured innovation 从 real replay 分布采样，并保留 position/velocity相关性；
- 分别实现 `legacy_force_fresh` 与 `strict_fractional_continuous` 以做 frozen-policy paired counterfactual；
- observation noise/dropout/prediction intervention逐项和组合评估；
- 不把真实失败动作当 imitation target，不用 record 中已失稳的 action 监督新策略。

先冻结 v5 做 observation A/B。只有候选在 clean 与 stressed rows 都不退化，才允许进入训练。

## P5：末端和接触指标必须进入 validation

每次 physical contact 记录：

```text
q/dq/ddq
racket position/quaternion/face normal
racket linear/angular velocity
contact point/offset
ball pre/post linear and angular velocity
normal/tangent impulse if simulator可得
next apex and next-contact recoverability
```

报告至少分首击、第二击、后续击，并回答：

- q/dq OOD 是否传到 EE position/orientation/linear/angular velocity；
- second contact 的 `normal_y`、racket `vz`、outgoing `vy` 是否进入 real envelope；
- angular velocity 对 contact-point translational speed 的贡献多大；
- failure 是 plant、pose、impact latent、observation handover 还是组合项。

不要再用单一 `hit_count` 掩盖第二击已经横飞的轨迹。

## P6：训练顺序

### Phase 0：zero-update frozen-policy counterfactual

完成 A–G matrix 和逐 episode paired diff。确认 plant、impact、observation各自及交互的效应。

### Phase 1：小规模 smoke

- 从 v5 权重继续训练与 from-scratch 至少比较一种；
- `n_envs<=16`、`n_steps<=32`、1 update；
- 检查 loss、KL、NaN、episode reset、bucket sampling、显存；
- smoke 只验证代码，不作为性能结论。

### Phase 2：短 paired learning probe

固定 seed/采样序列，对照 candidate：

- baseline curriculum；
- +real plant DR；
- +impact/recovery；
- +observation state machine；
- full combination。

用相同 update budget，先验证是哪一项带来 stressed-matrix 改善。若 clean 或关键 stressed row 崩溃，停止长训并回到因果定位。

### Phase 3：正式训练

只有 Phase 2 通过才启动。checkpoint promotion 不能只看训练 return；必须通过 P7 gates。

## P7：预注册验收门

每个 candidate 至少 3 个 seed、每个 row 至少 128 个 first episodes；报告 bootstrap CI 或 paired seed/env 差。

相对 frozen-v5 baseline：

1. nominal full-survival 不得下降超过 5 个百分点；
2. selected-plant row 的 full-survival 至少提升 15 个百分点，或其 CI 明确优于 baseline；
3. impact/recovery rows 的 worst-bin hit>=3 与 full-survival均改善，不能用一个 bucket 换另一个 bucket；
4. combined plant+impact+observation row 必须改善，这是 promotion 的主 gate；
5. hit-vxy、second outgoing `|vy|`、second `normal_y`、next-contact recoverability不退化；
6. q/dq/ddq、command error、action saturation、racket workspace/safety intervention不超过现有 envelope；
7. 60/90 paired gate继续保留，部署目标是 90 Hz；
8. 每个失败有明确 done/contact classification，禁止只报 mean hits。

若 15 个百分点在统计上达不到但其他指标显著改善，报告结果并停在 candidate，不得自行降低 gate 后宣布 promotion。

## P8：测试与交付

必须新增/通过：

- checkpoint/XML/hash/release identity test；
- plant delay/second-order CPU↔MJX parity test；
- exact-command calibration/holdout replay test；
- contact latent/outcome metric test；
- ball-on-racket reset与零初始平动/可控 spin test；
- 90 Hz fractional/predictor/reacquisition state-machine test；
- deterministic first-episode validator test；
- factorial matrix aggregate/report schema test。

交付内容：

```text
代码改动
运行 manifest
baseline A–G matrix
所有 frozen-policy counterfactual
smoke/短 probe（若获准使用 GPU）
逐 seed/逐 episode CSV
promotion decision
未识别的物理参数和下一步测量需求
```

最终报告必须把三种语句分开：

- “已由 real data 直接测得”；
- “已由 sim intervention 证明有因果效应”；
- “只是 robustness hypothesis，尚未识别为真实参数”。

---
