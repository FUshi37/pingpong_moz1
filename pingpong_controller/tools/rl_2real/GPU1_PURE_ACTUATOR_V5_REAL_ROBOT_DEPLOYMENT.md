# GPU1 pure-actuator v5 实物部署与提交说明

## 1. 提交对象

- 分支：`main`。
- 基线提交：`609e8c6f3f32c801b0f2b396fa6f396c6cb47019`
  (`deploy: add GPU0 measured-observation servo policy`)。
- 上一个 GPU1 部署提交：`7cf280441c0ba93cf13ebca04887037ffb9d1faf`
  (`deploy: add GPU1 resume8 inverse-MPC policy`)。
- 本次模型：
  `pingpong_controller/outputs/rl_sim/goal_d455_gpu1_pure_stable_v5_recovery_20260802/mjx_curriculum_best.pkl`。
- 模型 SHA-256：
  `9d7e94e9ef803fcbe9385ab97485626b2529394be62c75136c69d873adffaa79`。
- 文件大小：2,589,311 bytes。

本次只提交一个最佳 checkpoint，不提交 `last`、中间 checkpoint、视频、
action 曲线、W&B 目录或训练日志。

## 2. 模型选择与训练结果

最佳 checkpoint 位于第 21 阶段
`launch19_final_measured_obsres2mm_sport_nocomp_consolidation`，stage update
870、global update 2095、global step `1533116416`，best stage score 为
`38.780487182711454`。网络接口是 67D actor observation、7D action；231D
asymmetric critic 只在训练中使用。

checkpoint 所在更新的关键指标为：

| 指标 | 数值 |
|---|---:|
| mean hits | 13.1575 |
| recent mean hits (hits >= 3) | 13.5603 |
| mean episode length | 1128.1 |
| 1200-step episode rate | 0.8973 |
| hit1 / hit3 / hit12 rate | 1.0000 / 0.9658 / 0.8288 |
| camera-visible hit rate | 0.9984 |
| lower-band hit rate | 0.9678 |
| mean hit horizontal velocity | 0.1358 m/s |

最终收敛窗口为 mean hits 13.3293、hits>=3 mean 13.5203、length fraction
0.9598、full rate 0.9120，并通过课程收敛判定。本次选择训练过程中按
task score 冻结的 best checkpoint，不是简单选择最后保存的模型。实物部署前
的确定性任务回放仍应在目标软件版本和 XML 上单独执行，不能用训练窗口代替。

## 3. 与前两次部署的关键差异

本模型不是 GPU1 resume8 的 inverse-MPC 路径，也不是 GPU0 的 servo-planner
路径。固定执行契约如下：

| 项目 | 本模型 |
|---|---|
| 执行器模型 | 训练仿真中使用 second-order sport actuator model |
| 执行器 compensation | `none` |
| servo target planner | 关闭 |
| software command delay/filter | 实物端关闭 |
| policy/control rate | 200 Hz，`dt=0.005 s` |
| actor input/output | 67D / 7D |
| ball observation | 60 Hz 测量保持，位置单位 m、速度单位 m/s |
| joint state/command | 右臂 7 关节，rad / rad/s / rad |

训练中的 second-order 响应、每关节延迟和轻量 DR 用来近似真实执行器，并不
表示部署时还要在软件里再执行一遍同样的延迟/滤波。控制器必须：

1. 保留 checkpoint 所需的命令历史，以构造 delay-conditioned 67D 观测；
2. 将当前策略积分得到的名义关节位置目标直接发送给真实驱动；
3. 不启用 compensation、inverse MPC 或 servo planner；
4. 仍然经过独立的 `RightArmCommandSafetyLimiter` 和底层驱动限位。

如果把 `arm_actuator_q_ref_active`（仿真延迟后的历史目标）再次发送给实物，
就会在真实执行器自身延迟之外叠加第二次软件延迟。本次
`mjx_policy_controller.py` 的最小修复正是避免这个错误。

## 4. 本次需要提交的文件

仅包含：

1. `pingpong_controller/tools/rl_2real/mjx_policy_controller.py`：
   compensation=`none` 时输出当前名义目标，历史延迟只服务于观测；
2. `pingpong_controller/outputs/rl_sim/goal_d455_gpu1_pure_stable_v5_recovery_20260802/mjx_curriculum_best.pkl`：
   唯一提交的最佳模型；
3. 本文档。

以下运行时文件已经存在于 `main`，本次不重复修改：

- `pingpong_controller/pingpong_node.py`；
- `pingpong_controller/mjx_policy.py`；
- `pingpong_controller/safety_limiter.py`；
- `pingpong_controller/tools/rl_sim/delay_control.py`；
- `pingpong_controller/models/moz1_pd.xml`。

当前工作区的 XML、训练代码、补偿器、jerk/governor、缓存和其他实验输出均
与该 checkpoint 的最小部署无关，禁止加入本提交。

## 5. 部署前冒烟测试

先验证模型加载、维度、FK 和控制器输出：

```bash
cd /home/yangzhe/Project/pingpong_controller
PYTHONPATH=pingpong_controller/tools/rl_sim \
/home/yangzhe/miniconda3/envs/pingpong/bin/python \
  pingpong_controller/tools/rl_2real/mjx_policy_controller.py \
  --checkpoint pingpong_controller/outputs/rl_sim/goal_d455_gpu1_pure_stable_v5_recovery_20260802/mjx_curriculum_best.pkl \
  --robot-xml pingpong_controller/models/moz1_pd.xml \
  --steps 20 --ball-valid \
  --arm-q-deg 32,-58,43,98,26,-6,47 \
  --arm-dq-deg-s 0,0,0,0,0,0,0
```

检查输出至少满足：`obs_dim=67`、`act_dim=7`、delay conditioning 开启、
`actuator_compensation_mode=none`、`drive_target_tracking_planner=False`，且
全部输出有限。代码级验证还必须满足 published command 等于 latest nominal
target，而不是 delayed active target。

## 6. ROS 2 启动示例

完成工作区构建并 source 后：

```bash
cd /home/yangzhe/Project/pingpong_controller
source install/setup.bash
ros2 run pingpong_controller pingpong_node --ros-args \
  -p enable_rl_policy:=true \
  -p rl_policy_backend:=mjx \
  -p rl_model_path:=$PWD/pingpong_controller/outputs/rl_sim/goal_d455_gpu1_pure_stable_v5_recovery_20260802/mjx_curriculum_best.pkl \
  -p robot_xml_path:=$PWD/pingpong_controller/models/moz1_pd.xml \
  -p rl_require_fk:=true \
  -p control_rate_hz:=200.0 \
  -p rl_policy_dt:=0.005 \
  -p rl_action_gain:=1.0 \
  -p rl_action_scale_mult:=1.0 \
  -p record_rl_trace:=true
```

不要额外打开 compensation、command-delay filter 或 servo planner。启动日志
应显示 MJX backend、67D actor，并显示 compensation=`none`、planner=false。

## 7. 实物上机顺序与安全检查

该模型已通过仿真回放和控制器推理检查，但尚不能把这些结果等同于实物安全
验证。首次上机按以下顺序进行：

1. 机器人禁能或命令 topic 重映射时启动节点，确认关节顺序、符号、单位、
   `base_link` 球坐标和 200 Hz 周期；
2. 确认控制器读取的是真实 `q/dq`，球丢失、过期和无效状态能阻止危险输出；
3. 首次使能使用 `rl_action_gain:=0.1`，无球短时检查关节方向、命令跳变、
   safety limiter、急停和通信 watchdog；
4. 再按 0.2、0.4、0.7、1.0 分级增加，每级记录
   `/pingpong/rl_joint_cmd_state` 和 RL trace；
5. 每级检查肘部外翻/内翻、拍面倾斜、命令速度/加速度、驱动跟踪误差、温度
   和限位触发。任一项异常立即停机，不通过增益或补偿器掩盖坐标/时序错误。

模型训练时使用完整 `action_gain=1.0`。低增益只用于受控 bring-up，不代表
最终部署配置；增益变化会改变闭环行为，必须逐级验证。

## 8. 精确提交指令

由于工作区很脏，必须按文件暂存，禁止 `git add .`：

```bash
cd /home/yangzhe/Project/pingpong_controller
git switch main
git add -p pingpong_controller/tools/rl_2real/mjx_policy_controller.py
git add pingpong_controller/tools/rl_2real/GPU1_PURE_ACTUATOR_V5_REAL_ROBOT_DEPLOYMENT.md
git add pingpong_controller/outputs/rl_sim/goal_d455_gpu1_pure_stable_v5_recovery_20260802/mjx_curriculum_best.pkl
git diff --cached --stat
git diff --cached --name-only
sha256sum pingpong_controller/outputs/rl_sim/goal_d455_gpu1_pure_stable_v5_recovery_20260802/mjx_curriculum_best.pkl
git commit -m "deploy: add GPU1 pure-actuator policy"
```

执行 `git add -p` 时，只选择 `predict()` 末尾从 delayed active target 改为
latest nominal target 的 hunk，拒绝同一文件中所有 compensation 实验 hunk。
自动化操作可用等价的 index-only patch。提交后检查：

```bash
git show --stat --oneline HEAD
git show --name-only --format= HEAD
git status --short
```

如需同步远端，在再次确认提交文件只有上述三项后执行：

```bash
git push origin main
```
