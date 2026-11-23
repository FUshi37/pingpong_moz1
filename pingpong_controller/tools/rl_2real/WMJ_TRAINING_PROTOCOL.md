# WMJ 训练规范协议

| 项目 | 内容 |
| --- | --- |
| 英文名称 | World Model Juggling |
| 协议标识 | `wmj_training_protocol` |
| 协议版本 | `1.0` |
| 更新日期 | 2026-11-05 |
| 启用方式 | 每次训练独立选择，默认关闭 |

## 1. 遵循声明

在运行清单中登记；启用时保存本协议快照及 SHA-256。

```yaml
training_protocol:
  enabled: true
  id: wmj_training_protocol
  version: "1.0"
  sha256: "<协议文件 SHA-256>"
  overrides: []  # 按条款、字段和值登记本次调整
```

- `enabled: true`：遵循所绑定的协议版本及显式调整。
- `enabled: false` 或未声明：不启用本协议。
- 已启动训练固定协议快照；后续文档更新用于新运行。

## 2. 版本管理

### 2.1 版本层级

| 标识 | 内容 | 升级条件 |
| --- | --- | --- |
| `WMJ-Vxxx` | 训练主版本 | 控制架构、观测结构、核心训练目标、物理模型或课程路线发生重大变化 |
| `CONTACT-vN` | 碰撞配置 | 接触模型或标定方案发生重大变化 |
| `EXEC-vN` | 训练执行优化 | 形成新的执行、编译或内存优化方案 |
| `COURSE-vN` | 课程方案 | 阶段组织、学习路线或难度设计发生重大变化 |
| `VERIFY-vN` | 验证协议 | 验证方法、证据范围或通过判据发生重大变化 |
| `config_rev`／组件修订 `rNN` | 同方案内的小改动 | 参数微调、局部修复、日志和保存策略调整 |
| `run_id` | 一次运行批次 | 新启动、续跑、换种子或换设备 |

- 每个大修改对应一个主版本；同方案的小改动递增修订。
- 新训练名称使用 `WMJ`，机器标识和运行目录使用 `wmj` 前缀。
- 训练主版本引用各组件的版本、修订和配置哈希。
- 组件版本独立更新；训练主版本按训练方案的重大变化更新。
- 每份生效配置具有唯一标识和哈希；行为变化使用唯一的内部 profile 标识。
- 保留旧版本、旧修订和历史编号；模型别名绑定具体 checkpoint SHA-256。
- 每次启动使用独立输出目录；GPU、种子和运行批次作为独立字段登记。

### 2.2 碰撞配置登记

| 类别 | 登记内容 |
| --- | --- |
| 球与拍面 | 质量、半径、惯量、碰撞几何、接触面偏移 |
| 接触参数 | `solref`、`solimp`、`solreffriction`、friction、condim、margin、gap |
| 数值配置 | 步长、积分器、求解器、迭代数、容差、MuJoCo／MJX 版本 |
| 标定证据 | 数据哈希、有效样本、排除规则、拟合与留出结果 |
| 结果验证 | 静态／动态条件、法向与切向速度、最高点、横向偏移、CPU／MJX 对照 |

质量、接触参数、奖励的改动分别登记和对照。验证结果标明适用实验条件。

### 2.3 执行优化登记

| 类别 | 登记内容 |
| --- | --- |
| 改动 | 执行路径、编译、缓存、传输或内存管理差异 |
| 等价要求 | 逐位一致，或预先登记的逐项数值容差 |
| 配对输入 | 同 checkpoint、RNG、初态、动作及训练配置 |
| 正确性 | 状态、观测、奖励、接触、终止、参数及 Adam 状态对照 |
| 更新完整性 | 实际 minibatch／Adam 次数、样本量、优化器时间 |
| 性能 | 冷启动耗时、热态秒/update、SPS、端到端耗时、主机内存和显存 |
| 适用范围 | 软件与依赖哈希、XML、模型结构、环境规模及运行证据 |

计时等待 GPU 实际计算完成。回合长度、求解精度、采样量及更新次数的调整归入相应训练或物理配置修订。

## 3. 训练流程与启动登记

```text
目标与基线 → 版本及课程登记 → 变更验证 → 启动预检
→ 训练 → 课程判定 → 保存结果 → 按任务安排独立评测
```

| 类别 | 启动前登记与核对 |
| --- | --- |
| 目标 | 改善指标、保留指标、对照方案、停止条件 |
| 身份 | 协议声明、主版本、组件版本、修订、profile、run ID |
| 来源 | 父 checkpoint 路径及哈希、true step、update、Adam 时间、阶段 |
| 接口 | Actor／Critic／Action 维度、观测 schema、动作与控制语义 |
| 配置 | 完整命令、生效配置、PPO、回合长度、频率、XML 与数据哈希 |
| 源码 | Git commit、工作树状态、相关补丁、实际依赖快照及哈希 |
| 环境 | Python、JAX、MuJoCo、CUDA 版本与实际计算后端 |
| 资源 | 物理 GPU UUID、进程归属、tmux、主机内存及温度保护 |
| 存储 | 新输出目录、W&B 身份、checkpoint 保存与保留策略 |

行为变更先建立回归检查，再按改动范围完成 CPU smoke、配对回放、GPU probe 和多种子实验。

## 4. 每关训练要求

### 4.1 课程合同

每关填写以下合同；训练必填项在启动前确定，冻结验证项按所选模式填写。

```yaml
stage:
  id: S01
  name: "<稳定阶段名>"
  objective: "<学习目标>"
  changes_from_previous: []
  entry_checkpoint_sha256: "<来源权重 SHA-256>"
  training_episode_s: null
  min_updates: null
  training_gate:
    window_updates: 32
    consecutive_passes: 16
    min_steady_events: 100
    metrics: {}  # 每项填写单位、统计口径、阈值及必要样本量
  validation:
    mode: training_only  # training_only / frozen_sampled / frozen_paired
    duration_s: null  # 冻结验证模式填写
    cases: null
    reset_seeds: []
    action_seed: null
    conditions: []
    gates: {}
    retention_cases: []
    retry_updates: null
  advance_when: "<明确的训练与验证条件组合>"
  stop_conditions: []
```

### 4.2 训练判定

- 每关明确新增难度及保持不变的配置，优先单因素递进。
- 默认使用新建的 32-update 事件加权窗口，至少 100 个稳态事件，连续 16 次通过；调整值写入课程合同。
- 进入新关重新建立质量窗口及连续通过计数。
- Actor、Critic、Adam、RNG 和环境状态的继承／重建方式分别登记。
- 所有必需指标齐全、有限且满足样本量后计算通过状态。
- 存活、击球、高度、居中、球与拍运动、视野和安全指标逐项登记门槛。
- 训练回合时长、每环境推进时间、累计采样量和实际耗时分别记录。
- 进阶按 `advance_when` 执行；预算耗尽记录为停止。

## 5. 每关验证与进阶要求

### 5.1 验证模式

| 模式 | 验证要求 | 结果范围 |
| --- | --- | --- |
| `training_only` | 本关滚动训练质量达到合同要求 | 训练内进阶或完成资格 |
| `frozen_sampled` | 固定候选权重，采样策略完成本关验证及保留能力复测 | 本关采样策略验证 |
| `frozen_paired` | 固定候选权重，采样与确定性策略完成配对验证及保留能力复测 | 本关双模式验证 |

- 每关预先选择模式，登记验证时长、样本、条件、阈值和重试间隔。
- `training_only` 的冻结验证字段填写 `null`／空列表，输出 `NOT_REQUESTED`。
- 关键物理、控制、观测及新增 DR 变化登记配对验证与基础能力复测要求。
- 独立最终验收单独登记模式、时长、条件及通过门槛。

### 5.2 冻结验证规则

- 每次验证绑定候选 checkpoint SHA-256；验证期间权重固定。
- 对照模型使用相同测试条件和 reset keys；采样策略配对使用相同动作随机数。
- 各组登记最少样本及门槛，分别报告名义、近侧、镜像等结果。
- 首次物理或数值失败即记录该样本失败；保留原始分母和失败分布。
- 验证结束检查观测时长、样本量、指标完整性及基础能力退化。
- 结果使用 `PASS`、`FAIL`、`INCOMPLETE`；按预设间隔重试并保存历次结果。

### 5.3 进阶记录

```text
当前关合同满足 → 保存候选权重 → 完成本关规定的验证
→ 保存判定与证据 → 进入下一关
```

记录阶段 ID、候选 SHA-256、训练门槛结果、验证模式与结果、各组样本量、失败项、进阶依据及下一关变化。

训练内资格、冻结阶段验证、独立最终验收和实机资格分别保存。混合策略训练统计标记为 `TRAIN_MIXED`，固定权重评测标记为 `FROZEN`。

## 6. 终端与指标输出

### 6.1 输出频率

| 标签 | 时机 | 内容 |
| --- | --- | --- |
| `[RUN]` | 启动 | 协议开关、版本、修订、组件、来源、GPU、目录、W&B |
| `[STAGE]` | 入关 | 阶段目标、配置差异、训练与验证要求 |
| `[TRAIN]` | 每 update | 阶段、全局／关内 update、true step、Adam 增量、耗时、SPS |
| `[LEARN]` | 每 update | KL、clip fraction、entropy、explained variance 及必要的辅助损失 |
| `[QUALITY]`／`[GATE]` | 默认每 10 updates，状态变化即时 | 窗口、事件数、连续通过次数、指标与失败项 |
| `[VERIFY]` | 验证开始、定期进度、结束 | 权重、模式、时长、完成数、分组、结果 |
| `[SYS]` | 默认每 30 s | 温度、主机内存、显存及资源保护状态 |
| `[SAVE]`／`[ADVANCE]`／`[STOP]` | 事件发生 | checkpoint、证据、进阶或停止原因 |

### 6.2 输出格式

```text
[RUN] project=WMJ protocol=... V=... rev=... CONTACT=... EXEC=... COURSE=... VERIFY=...
[STAGE] S=... name=... changes=... train_gate=... validation=...
[TRAIN] S=... u=... stage_u=... true_step=... Adam=...(+...) update_s=... SPS=...
[LEARN] KL=... clip_fraction=... entropy=... explained_variance=...
[QUALITY] source=TRAIN_MIXED window=.../32 events=... hold=.../16
          center=...mm/<=... apex=...m/[...,...] band=...%/>=...
          ball_vxy_mean/rms=.../...m/s racket_vxy_rms=...m/s omega_rms=...rad/s
[GATE] status=... failed=... actual=... limit=... margin=...
[VERIFY] source=FROZEN sha=... mode=... duration_s=... completed=.../... result=...
[SYS] gpu_uuid=... temperature_c=... host_mem_gib=... vram_gib=...
[ADVANCE] from=... to=... training=... validation=... checkpoint=... evidence=...
```

- 门槛显示实际值、目标值、单位、方向和状态；记录全部失败项。
- 指标登记来源、坐标系、事件范围、均值／RMS、权重方式和样本量。
- 样本不足、缺失字段、非有限值分别输出原因；无完成回合输出 `N/A, n=0`。
- 编译、等待资源、训练和验证分别显示状态及耗时。
- SPS 按本次实际环境步数除以实际更新时间计算；击球统计包含 hit1／hit3／hit12 及完成回合数。
- 终端、结构化日志和 W&B 共用指标定义、update 标识和统计结果。
- 时间戳包含时区；日志格式修订登记 schema 版本。

## 7. 续训、保存与归档

- 同合同续训恢复已声明的完整状态；目标或配置变化登记修订和迁移方式。
- 每次续跑建立新 run，记录父 run、checkpoint 及状态继承情况。
- W&B 展示 step、checkpoint true step 和 Adam 时间分别保存；复用 W&B run 登记对应授权。
- checkpoint 使用临时文件写入、完整性检查和原子替换，登记 SHA-256。
- 分别登记最新恢复点、最佳候选、课程边界、训练达标点和最终选定模型。
- 中断点、达标点、课程边界及已引用父模型作为受保护文件；周期清理按明确保留策略执行。
- 停止时记录最后完成 update、停止原因及实际保存位置。

| 归档项 | 内容 |
| --- | --- |
| 运行清单 | 协议声明、版本、配置、命令、来源、资源与环境 |
| 源码证据 | 协议与依赖快照、补丁、配置及数据哈希 |
| 训练记录 | 结构化指标、终端日志、分组结果、失败分布 |
| 课程判定 | 入关、训练达标、验证、进阶及停止记录 |
| 模型 | 完整状态 checkpoint、种类、哈希和保留关系 |
| 结果摘要 | 目标完成情况、对照结果、负面结果和评测状态 |

关键证据持久化保存。选定模型写入模型台账，课程与 profile 合同写入训练课程登记。

## 8. 协议维护

- 条款变化更新协议版本及变更记录；文字修订通过 Git 和文件哈希追溯。
- 新规则放入对应章节，数值门槛放入具体课程合同。
- 历史 run 保留启动时的协议、配置和结果。

| 日期 | 版本 | 更新 |
| --- | --- | --- |
| 2026-11-05 | 1.0 | 统一采用 WMJ（World Model Juggling）命名，建立可选遵循声明、分层版本管理、课程训练与验证、终端输出及归档规范 |

相关文档：[仓库规则](../../../AGENTS.md) · [训练课程登记](../../../TRAINING_CURRICULA.md) · [模型训练与测评台账](FPLC_TRAINING_AND_EVALUATION_LEDGER_20261026.md)
