# record_new3–5 实物颠球失败根因报告

日期：2026-08-13  
范围：`record_new3`、`record_new4`、`record_new5` 中的 simcode5 数据；simcode6 的 2 个 bout 仅计入清单，不参与 v5 根因统计。

## 最终结论

实物不稳定的首要根因是：**真实机器人从 policy angle command 到实测 `q/dq` 的状态转移，与 v5 checkpoint 训练时的 per-joint delay + second-order plant 不一致。** 这不是依据“看起来更快”作出的判断，而是由两个独立实验确认：

1. record_new3 的同球输入 replay 中，实物与仿真最早在 `dq` 分叉，中位时间 `0.130 s`；随后才依次出现 command error（`0.200 s`）、actor action（`0.265 s`）、previous action（`0.270 s`）、拍端速度（`0.300 s`）、拍端位置（`0.310 s`）和关节位置（`0.340 s`）分叉。
2. 同一 frozen v5 checkpoint、seed 976、64 个确定性 90 Hz 自主仿真 episode，只替换 plant 中心后，6 s full-survival 从 `0.96875` 降到 `0.609375`，mean hits 从 `13.875` 降到 `10.531`，hit-vxy 从 `0.101` 增到 `0.155 m/s`。

真实失败的直接接触机制也已经定位：**到第二次碰撞时，失败轨迹的拍面法向已经偏向 `+Y`，法向冲量把球继续推向 `+Y`；真实接触的未解析切向/旋转响应通常没有把它抵消。** 这解释“为什么球飞偏”，但不是比 `dq` 更早的根因；姿态分支是前面 robot/history/ball transition 已分化后的接触结果。

还确认存在一个真实 impact transition gap：首击后实物 apex 的 P50 为 `1.443 m`，clean sim 为 `1.277 m`，且实物首击拍端 `vz` 并不更大。现有数据足以证明实物的碰撞输入—输出关系没有被 clean reference 覆盖，但**不能**在没有球自旋、接触力、胶皮变形和精确接触点测量时，把它唯一归因于自旋、摩擦、恢复系数或柔顺性中的某一个参数。

record4–5 的 `dq_adapter` 是次级 observation 问题：它只改 `obs[7:14]`，而拍端速度仍来自 raw joint sample，因而同一个 observation 内出现两套互不一致的物理速度。它会显著改变动作，且在 nominal sim 中会降低性能；但没有 adapter 的 record3 同样不稳定，所以它不是三个数据集共同的首要根因。predictor/reacquisition 是有数据支持的条件性放大器，但没有 paired real A/B，不能提升为根因。

没有发现 frozen actor 权重、67D 索引、previous-action 一拍时序、action→acceleration 映射或两次积分的通用部署实现错误。球检测/观测频率也不是根因；保留用户已经实物验证过的 90 Hz。

## 因果链，而不是现象列表

```text
球近静止放在球拍上启动（实物记录与仿真 reset 一致）
        │
        ▼
真实 command→q/dq plant 与 checkpoint plant 不同
        │  最早证据：same-ball dq gap，P50=0.130 s
        ▼
command error / actor action / previous-action history 分化
        │
        ▼
拍端速度、位置和接触时相分化
        │
        ├── 首击真实 impact transition 产生更高 apex，进入 clean sim 外状态
        │
        ▼
第二击拍面朝 +Y、竖直速度相位不对
        │
        ├── 法向投影把球推向 +Y
        └── 未解析切向/自旋/柔顺响应通常未充分抵消
        ▼
previous action + 双积分 + 部分 action saturation 继续放大
        │
        ├── record4–5 dq-only adapter 增加 observation 内部不一致
        └── 部分 bout 的 predictor→measured handover 产生附加动作跳变
        ▼
球/拍越界；EE safety 终止（终点，不是首因）
```

## 1. 数据、身份与证据边界

### 1.1 数据覆盖

共读取 65 个 recording、67 个 bout，其中 65 个为 `simcode5_pure_actuator67`：

| 数据组 | v5 bouts | 平均击数 | 中位击数 | 最大击数 | 10+ 次 |
|---|---:|---:|---:|---:|---:|
| record_new3 | 16 | 3.19 | 2.5 | 13 | 1 |
| record_new4 | 26 | 3.27 | 3 | 10 | 1 |
| record_new5 | 23 | 1.96 | 1 | 8 | 0 |

“失败标签”来自启发式 contact detector，只用于分组；它不是力传感器真值。为此又做了阈值敏感性和位置拟合复核，见第 6 节。

### 1.2 checkpoint/XML 身份是精确的

使用的 v5 checkpoint：

```text
SHA256 9d7e94e9ef803fcbe9385ab97485626b2529394be62c75136c69d873adffaa79
```

使用的 XML：

```text
SHA256 7d98f2adfdbad6082be0defcec2dbd0cbbcaf1f0fc06ce45ba424b5b3257cc92
```

这两个对象分别与 `4f47e6d3c67917acd6a9b05162a488c91b319597`、`69b9bf3442f5e9473bb291e0fa9439d0f15a53d5` 中的对象逐字节一致。

### 1.3 不能伪称“完整历史源码逐字一致”

实物 recording metadata 没有记录 git SHA。当前实物仓库 HEAD 为 `e471ddef...`，是 8 月 13 日后的汇总状态；无法从 git 唯一恢复 8 月 10–13 日每个 recording 当时的完整源码。因此：

- checkpoint、XML、runtime profile、每 tick observation/action/command 可以由 hash 和日志验证；
- 历史实物完整源码与当前 checkout 是否逐字一致，无法验证，报告不会伪造这一结论；
- 当前实物 checkout 还缺少启动脚本默认引用的 `simcode5/`、`simcode6/` 资产目录，完整测试为 `167 passed, 8 failed, 14 errors`；失败主要是缺 checkpoint/trace，另有一个 `SPORT_TASKSPACE_FIT_RIGHT_ARM_PD` 导入漂移。这是当前发布可复现问题，不是 record3–5 的历史失败证据。

### 1.4 自主仿真版本边界

完整 plant A/B 使用当前本地完整 validator；当前 `mjx_juggle_env.py` 相对 commit 69 已有大量后续变化。因为 A/B 两侧除 plant 参数外完全相同，它仍能证明 plant 替换的闭环因果效应；但数值不能冒充“clean commit 69 的独立复现”。clean-69 GPU1 运行因沙箱无 CUDA 被拒绝，CPU 1-env 编译超过约 2 分钟后为保护系统主动终止。

## 2. 启动与策略计算合同

### 2.1 球确实静止在拍上启动

record_new3 示例 metadata：

```text
mode                 racket_launch
surfaceGapM          0.009713 m
normalVelocityMps   -0.001062 m/s
tangentVelocityUv   [0.001830, -0.001572] m/s
world velocity       [0.000040, -0.002449, -0.000973] m/s
```

仿真 checkpoint 同样使用 `racket_launch`，表面间隙 `5–10 mm`，横向 jitter `4 mm`，初始平动速度约为 0。不存在“实物从拍上启动、仿真从空中下落”这个 gap。

### 2.2 67D/actor/action mapping 对齐

对 65 个 recording、32,006 个 v5 policy tick 做独立重算：

- actor clipped action 重算 RMS mean 约 `9.9e-7`，P99 约 `2.3e-6`，最大 `<3.7e-6`；
- `q`、actor 实际使用的 `dq`、ball state、relative state、previous action、command state/history 均与日志吻合；
- action→acceleration、command velocity、command position 的连续两次积分在连续 segment 内 P99 为浮点舍入量级；
- raw policy command 与最终 safe published position 的 P99 为 0/舍入量级，未发现普遍隐藏改写；
- workspace soft intervention 在 v5 bouts 中为 0。

所以 actor 权重、obs 顺序、单位、history 时序、双积分没有通用 code bug。两次积分和 previous action 是闭环记忆通道，不等于实现错误。

### 2.3 代码相同与不相同的准确答案

| 环节 | 结论 |
|---|---|
| checkpoint 参数与 NumPy actor forward | 对齐；日志动作重算已验证 |
| 67D 字段顺序、action scaling、双积分 | 对齐；逐 tick 重算已验证 |
| `q/dq` 来源 | 语义不同但预期如此：sim plant state vs 实测 JointState |
| ball state | 不相同：sim 内部 observation model；real 有相机、预测、重捕获与 90 Hz cadence |
| record4–5 arm `dq` 与 racket velocity | 不一致：arm dq 被 adapter 改写，racket velocity 仍反映 raw physical motion |
| policy command 后的 plant | 不相同且是主因：sim 有 checkpoint delay+second order；real `simcode5_direct` 直接发送 angle，假设硬件天然等价 |
| safety wrapper | 实物独有；本批记录没有持续 soft intervention，属于终止保护层 |

## 3. 从时间向前与从失败向后得到同一个根因

### 3.1 同球输入的最早分化

record_new3 有 15 个可用 same-ball bouts。sim mirror 每 tick 接收完全相同的已记录球输入，因此在首击前，ball group 的 actor counterfactual差为 0；它只测试 robot/history transition。

| 首次超过阈值 | onset P50 |
|---|---:|
| joint `dq` gap > 0.2 rad/s | **0.130 s** |
| command-error gap > 0.02 rad | 0.200 s |
| action gap > 0.2 | 0.265 s |
| previous-action gap > 0.2 | 0.270 s |
| racket-velocity gap > 0.2 m/s | 0.300 s |
| racket-position gap > 0.01 m | 0.310 s |
| joint `q` gap > 0.02 rad | 0.340 s |

这直接排除了“先由后续球丢失导致 actor 错，再导致机器人错”的时间顺序。第一处分化发生在 robot velocity transition。

首击前逐 observation group 替换的 actor mediation：

| 把实物 group 换成 same-ball sim group | action-gap MSE 被关闭比例 |
|---|---:|
| previous action | 72.2% |
| arm dq | 16.8% |
| command tracking | 12.8% |
| command velocity | 9.2% |
| ball | 0% |

这些比例不能相加。previous action 的高比例说明分化一旦发生会通过 recurrent input 自我保持；最早时间证据仍把根放在 `dq/plant`，而不是把 previous action 当成第一原因。

### 3.2 exact-command actuator replay

用相同的实际 published angle command 驱动 checkpoint plant 和逐关节标定候选；12 个 calibration recording、3 个未参与选择的 holdout：

| 模型/划分 | q RMSE | dq RMSE | robust qacc RMSE |
|---|---:|---:|---:|
| checkpoint plant / calibration | 0.538° | 0.289 rad/s | 5.470 rad/s² |
| selected real plant / calibration | **0.425°** | **0.222 rad/s** | **4.496 rad/s²** |
| selected real plant / holdout | **0.381°** | **0.219 rad/s** | **4.404 rad/s²** |

holdout 中实物 `dq` RMS 为 `1.141 rad/s`，selected model 为 `0.949 rad/s`；实物 qacc RMS 为 `17.54 rad/s²`，model 为 `14.59 rad/s²`。所以 selected plant 比 checkpoint 明显更接近，但仍低估真实高速尾部，不能称为最终真值。

当前标定候选：

```text
wn(rad/s) = [21.8911, 22.0896, 22.6505, 21.7305, 20.1549, 22.2451, 23.0547]
zeta      = [0.3330, 0.3296, 0.3112, 0.3114, 0.3131, 0.3112, 0.2855]
delay(ms) = [45, 50, 45, 40, 35, 45, 50]
```

### 3.3 自主闭环 plant 单因素实验

| condition | full | mean hits | mean length | hit-vxy |
|---|---:|---:|---:|---:|
| checkpoint nominal plant | **0.969** | **13.875** | **1182.0** | **0.101 m/s** |
| selected real plant | 0.609 | 10.531 | 932.3 | 0.155 m/s |
| selected plant + 3.7 g + solref damping 0.90 | 0.672 | 11.297 | 1003.3 | 0.144 m/s |

因此：

- plant gap 是经过自主闭环实验确认的主因；
- 它仍不是全部原因，因为 selected plant 仿真平均仍有 10.5 击，优于实物；
- 3.7 g 球不是单独充分原因，在这次组合干预中没有继续恶化。

## 4. 分布外输入与“把相同分布送进仿真”

这里用了两类 reference：自主 seed976 env0 的 1,200-step trace，以及 record3 same-real-ball mirror。前者只是 clean 代表轨迹，不是完整训练分布；后者只能隔离 robot/history，不能代替自主球物理。

real raw-dq cohort 相对 clean trace P1–P99 的主要越界：

| observation | 越界比例 | real/clean P1–P99 宽度比 | 解释 |
|---|---:|---:|---|
| ball z | 30.6% | 1.75× | same-ball mirror 同样高，主要由真实碰撞轨迹驱动 |
| arm q6 | 28.8% | 1.71× | same-ball mirror 同样扩展，属于闭环 plant/recovery 分支 |
| arm q2 | 24.9% | 2.04× | 同上 |
| relative x/y/z | 20.1–24.3% | 2.30–3.15× | 球轨迹和拍状态共同分化 |
| arm dq1 | 18.1% | 1.61× | raw real dq 高速尾部直接 OOD |
| arm dq5 | 17.6% | 1.44× | 同上 |
| command-error/history 若干维 | 10–17% | 约 2–7× | plant 分化后的策略记忆状态 |
| ball vy | 14.2% | 1.65× | 真实 impact/trajectory 分布更宽 |

base x/y 因毫米级静态标定偏移显示很高“越界比例”，但 same-ball mirror 使用相同固定 base，且 base group 替换关闭 0% action gap，故不把它误列为主因。

把真实 plant 分布放回自主仿真，性能显著下降，回答了“相同执行分布下仿真会怎样”。把真实球逐 tick 喂给 mirror 则只能回答 robot/history 何时分化，不能用来声称已经复现真实 contact dynamics。

## 5. 末端到底是位置、姿态、速度还是角速度

### 5.1 任务空间状态

| 接触状态 | raw dq RMS P50 | racket linear speed P50 | angular speed P50 | racket vz P50 |
|---|---:|---:|---:|---:|
| 实物首击，N=56 | **1.146 rad/s** | 1.024 m/s | 1.690 rad/s | 0.995 m/s |
| clean sim env0 首击 | 0.856 rad/s | **1.397 m/s** | **1.910 rad/s** | **1.372 m/s** |
| 实物第二击，N=49 | **1.036 rad/s** | 0.563 m/s | 1.027 rad/s | -0.076 m/s |
| clean sim env0 第二击 | 0.685 rad/s | **0.706 m/s** | **1.216 rad/s** | **+0.668 m/s** |

结论不能简化成“实物所有末端速度都更快”。实物关节 `dq` 更大，但由于关节组合、Jacobian 和相位不同，拍端线速度/角速度并不普遍更大。第二击 `vz` 符号不同尤其说明接触时相已经变了。

- **位置：** 首击/第二击拍心横向接触偏移 P50 约 21/26 mm，较小；没有证据支持它是首要分叉。
- **姿态：** 是第二击横飞的直接机制。0–3 击失败第二击 `normal_y` P50=`+0.144`，clean env0 为 `+0.011`。
- **线速度：** 是执行/相位 gap 的重要中介；第二击竖直速度甚至反号。
- **角速度：** raw joint dq OOD，但拍端 angular speed 不普遍高于 sim；球偏心约 10–20 mm 时，角速度转成接触点平移速度的 P50 上界约 `0.025 m/s`，不足以独自解释横飞。

### 5.2 第二击冲量分解

用 z 方向消去未知法向冲量：

```text
vout_y = vin_y + (vout_z - vin_z) * n_y / n_z + residual_y
```

| 第二击 | vin_y | 法向投影 y | residual_y | vout_y | normal_y |
|---|---:|---:|---:|---:|---:|
| clean sim env0，N=15 | -0.029 | +0.034 | -0.015 | -0.029 | +0.011 |
| 实物 0–3 击，N=30 | -0.062 | **+0.444** | -0.022 | **+0.223** | **+0.144** |
| 实物 8+ 击，N=3 | -0.158 | +0.321 | **-0.246** | -0.052 | +0.115 |

普通失败中，拍面 `+Y` 法向项很大，而负 residual 基本没有抵消；少数长 bout 恰好重新出现较大的负 residual。`residual_y` 包含未解析切向冲量、自旋、柔顺、接触点和估计误差，不能把它命名为“摩擦真值”。

### 5.3 首击已经把球送到 clean reference 外

实物首击 incoming/outgoing `vz` P50 约 `-1.596/+1.900 m/s`，拍端 `vz` P50 约 `+0.995 m/s`；clean env0 拍端更快，但实物下一 apex 仍更高：

```text
real first-apex P50/P90     1.443 / 1.550 m
clean validate P50/P90      1.277 / 1.337 m
```

这证明仅用拍端“更快”解释首击能量不成立；真实 impact transition 的结果分布未被 clean reference 覆盖。静态 MuJoCo spin sweep 只证明自旋足以改变 `vout_y` 的符号，不证明本次实物球的真实自旋值。

## 6. 接触检测与速度异常值稳健性

baseline detector 得到 181 个事件。改变 down/up/jump/gap/xy/refractory 后：

- 单参数变体有 90.6–100% 的 baseline event 在 50 ms 内匹配；
- first-apex P50 始终约 `1.421–1.450 m`；
- second outgoing `vy` P50 始终为正，约 `0.191–0.229 m/s`；
- 极端 strict-all 虽把 contact-count P50 从 3 改为 2，仍保留高 apex 和正向第二拍。

另用接触后 fresh position 做 OLS 速度，不依赖日志瞬时 velocity。49 行中有一行 logged `vy=-9.0175 m/s`，位置拟合为 `+0.288 m/s`。按预先声明的对称物理 cap `|v|<=3 m/s` 排除这一行后：

```text
n=48
Pearson r=0.6123, p=3.74e-6
Spearman rho=0.5963, p=7.74e-6
sign agreement=72.9%
median absolute difference=0.099 m/s
```

因此均值会被一个坏点污染，但第二击正向横飞和高首击 apex 对 detector 阈值及独立位置拟合都稳健。

## 7. `dq_adapter`：有作用，但位置不对

adapter gain/tau/delay 确实把 record4–5 的 actor `dq` 边缘 RMS 拉近 clean sim，并在 same-ball mirror 中让 77–80% 的 tick action 更接近 mirror action；它不是“完全没用”。问题是它只替换 arm dq：

| 一致性误差 | record4 P50 | record5 P50 |
|---|---:|---:|
| logged racket velocity vs `J(q)dq_raw` | 0.028 m/s | 0.027 m/s |
| logged racket velocity vs `J(q)dq_actor` | **0.139 m/s** | **0.124 m/s** |
| adapter 对 action 的改变量 | 0.111 | 0.118 |

自主 sim A/B：

| plant | adapter off full | adapter on full | hit-vxy off/on |
|---|---:|---:|---:|
| nominal | **0.969** | 0.547 | 0.102 / 0.189 m/s |
| selected real plant | 0.625 | 0.641 | 0.141 / 0.181 m/s |

这组 adapter 交互来自另一批保存的同批次 A/B，selected-plant off 为 `0.625`；第 3.3 节三条件 plant 实验的对应值为 `0.609`。两者使用的本地 validator snapshot/有效配置并非同一个可证明 bitwise 的 release，故只在各自 paired batch 内读 intervention 差，不跨表比较 0.625 与 0.609。

adapter 在 nominal plant 明显有害，在 selected plant 上 full 有小幅交互性回升但 lateral quality 变差；实物没有 paired A/B。正确结论是：

- 它是已确认的 observation coherence 问题；
- 它不能改变真实 joint/EE motion，不是 plant 修复；
- 不能仅凭当前记录断言“默认开”或“默认关”一定提高实物长跑；
- 新 actor 应直接训练于可测 raw `q/dq` 与真实 plant 分布；frozen v5 的任何 adapter 模式必须做一致的 observation 构造并用 paired real A/B 决定。

## 8. predictor/reacquisition 与 90 Hz

49 个首击到第二击周期中：55.1% 使用过 prediction，53.1% 发生 reacquisition；reacquisition error P50/P90=`30/85 mm`，实物 action jump P50/P90=`0.135/0.923`。代码还允许 `force_fresh_source` 在 handover/impact event 上提前刷新 fractional sampler。

这证明 handover 可产生较大瞬时扰动，但 with/without reacquisition 与日期、配置共同变化，没有 controlled A/B；约一半周期也没有该事件。因此它是**条件性放大器**，不是共同根因。修复方向是保持 90 Hz、保证 handover 连续并通过 paired replay/real A/B，而不是退回 60 Hz 或删除 predictor。

用户已经做过实物频率实验；保存的 paired sim gate 也显示 90 Hz 不劣于 60 Hz。此次 exact-v5 plant A/B 全部在 90 Hz 下仍表现良好/可区分。故球观测频率从候选根因中排除。

## 9. 候选原因判决表

| 候选 | 判决 | 实验证据 |
|---|---|---|
| actor 权重/obs 索引/action mapping/双积分 bug | 排除 | 32,006 tick 重算到浮点误差 |
| 球从哪里启动 | 排除 | metadata 与 checkpoint 都是近静止 `racket_launch` |
| 90 Hz 球观测率 | 排除 | 用户实物 A/B + 保存 paired sim gate；90 Hz exact-v5 sim 仍稳定 |
| EE safety box | 排除为根因 | soft intervention=0；它只在末端停止 |
| 真实 actuator plant | **确认主因** | same-ball 时间顺序 + holdout replay + 自主 plant A/B |
| previous action / 双积分 / saturation | 确认放大器 | actor mediation；约 23–26% tick 有任一 preclip saturation |
| dq-only adapter | 确认次级 observation gap | Jdq coherence + action counterfactual + sim A/B；record3 反证其为共同首因 |
| 第二击拍面姿态 | **确认失败机制** | normal projection 分解；失败/长 bout 对比 |
| EE 位置偏差 | 未支持为主因 | 接触横向偏移中位仅 2–3 cm；比姿态/相位证据弱 |
| EE angular speed 单独过快 | 排除为单因 | 实物 contact angular speed 并不普遍高于 sim；接触点速度上界小 |
| real impact transition | 确认存在未覆盖分布 | 首击 input/output/apex 与 clean reference 不一致 |
| 自旋/摩擦/恢复/柔顺某一个具体参数 | 未识别 | 当前传感器不能唯一分解；spin probe 仅是充分机制测试 |
| predictor reacquisition | 条件放大器，未确认根因 | action jump 已测；无 paired real A/B |
| 3.7 g 球质量 | 排除为单独充分原因 | selected-plant 组合干预未继续恶化 |

## 10. 实物与仿真分别要解决什么

### 10.1 实物平台

1. 先修发布可复现性：启动前校验 repo/checkpoint/XML hash，metadata 写入 git SHA；缺资产直接 config-only fail，不允许默默换 profile。
2. 构造原子 `PolicyJointSample`，raw q/dq、FK、racket velocity、command error 使用同一 source stamp/seq。legacy dq-only adapter 只能作为明确标记的 ablation，不能继续伪装成 plant alignment。
3. 为 frozen v5 实现可旁路 `off|shadow|apply` 的 model-reference actuator matcher：目标是让真实 `q/dq` 接近 checkpoint plant 输出，不是简单让真实 q 追理想 qcmd。先 offline/shadow，再单击安全实验；当前证据没有承诺 wrapper 必然根治。
4. 保持 90 Hz 和 predictor；增加 strict-cadence/continuous-handover 模式做 paired A/B，不能把 ball rate 再当根因。
5. 记录最小接触诊断：同一时钟的 raw q/dq、拍面 normal、拍端 twist、碰撞前后 fresh ball state；若有力/力矩或高速相机才进一步识别 impact 参数。
6. 所有修改保留现有 joint/EE/predictive stop；依次 config-only、offline replay、shadow、1-hit、3-hit、长 bout，任何阶段不通过都不扩大实验。

详见 `REAL_ROBOT_PLATFORM_MODIFICATION_CODEX_PROMPT.md`。

### 10.2 仿真/训练

1. 将逐关节真实 plant 候选及 holdout residual 纳入 DR；不要只用一个全局 gain/tau。
2. 保留 ball-on-racket reset；增加真实首击 outcome、second-impact orientation/lateral-response、post-impact recovery 状态覆盖。
3. 扩展质量到实物 3.7 g，并随机化未观测 spin 与 normalized inertia；这些是 latent robustness 变量，不是假装已测真值。
4. 用部署可测 raw q/dq 训练；复现 90 Hz fresh/held/predicted/reacquired observation state machine，而不是只给 actor 内部真值。
5. validation 必须形成 factorial matrix：nominal plant、selected plant、plant residual tail、impact tail、predictor handover，以及组合项；先冻结策略做反事实，再决定训练。
6. 报告分位数、跨 seed paired 差和失败原因，不能只报 clean mean hits。

详见 `SIMULATION_SIM2REAL_MODIFICATION_CODEX_PROMPT.md`。

## 11. 可复现证据

本次重新生成的 CPU-only 产物位于 `verified_cpu/`：

- `causal_chain_diagnostic.json`：same-ball onset、actor mediation、impact、reacquisition、actuator replay；
- `policy_computation_contract_audit.json`：67D/actor/history/积分合同；
- `observation_ood_by_dimension.csv` 与 `observation_ood_summary.json`：逐维分布；
- `contact_detector_sensitivity.json/csv`：接触阈值稳健性；
- `second_hit_position_robustness.json`：异常值和位置拟合复核；
- `raw_joint_contact_phase.csv/json`：raw dq 与拍端 6D 状态；
- `second_hit_impulse_decomposition.csv`：第二击法向与 residual；
- `record_new3_same_ball_cycle_metrics.csv`：同球输入闭环分叉。

自主仿真 A/B 位于同级目录：`controlled_full_sim_plant_ablation.csv` 及四个 `full_sim_*90*.csv`。完整执行记录、hash、测试与算力限制见 `VERIFICATION_MANIFEST.md`。
