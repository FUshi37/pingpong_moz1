# record_new3–5 失败模态可重复性与分支根因附录

日期：2026-08-13  
状态：新增附录；不修改已经认可的根因报告及实物/仿真修改说明。

## 结论

可以从失败可重复性继续追根因，而且这条路径有效。当前数据支持的结构不是“很多随机失败”，也不是“每种终止原因各有一个根因”，而是：

```text
共同上游：真实 command→q/dq plant 与 checkpoint plant 不一致
        │  record3 最早 dq 分叉 P50≈0.130 s
        ▼
首击后的机器人姿态/能量/闭环历史进入少数可重复轨迹分支
        ├── B3：高能量回球 → 第二击 +Y 拍面法向 → 法向冲量主导横飞
        ├── B2：低能量回球 → 第二击法向近中性 → 切向/未解析冲量主导横飞
        ├── B1：小样本中间/混合分支，证据不足以单独命名根因
        └── BR1：单例反向横飞，不属于可重复模态
        ▼
previous action、双积分、dq adapter、predictor handover 等条件性放大
        ▼
EE/球越界终止（终点，不是首因）
```

这进一步细化了原报告：**plant mismatch 仍是三个数据组共同的首要 sim-to-real 根因，但它不是每次失败的充分条件；它把闭环推入少数高风险分支，真实 impact transition 再决定具体如何丢球。** record3 唯一 8+ 次参考 bout 同样存在早期 plant 分叉，因此不能声称“看到 0.13 s dq gap 就必然失败”。

## 1. 怎样避免把结果重新包装成原因

分析使用 65 个 `simcode5_pure_actuator67` bouts，其中 62 个为 0–7 击失败参考，3 个为 8+ 击长 bout 参考。对有至少两次可靠复触的 44 个失败 bout，轨迹指纹严格只取：

- 第一次推断触球开始，到第二次推断触球的前一个采样点为止；
- 7D arm `q`、7D raw arm `dq`、6D ball state、6D racket state、7D policy action 和该阶段时长；
- 21 点归一化相位；各物理组等权。

明确排除第二次触球判定样本、第二击后观测、最终触球次数、终止原因和任何后验失败字段。分支聚类完成后，才用第二击输出和终止模态检验它的后果。

这项严格化很重要：先前若包含第二次触球判定点，速度反转可能泄漏结果。去掉该点后，分支数量、样本数和预测结论均保持，稳定性反而略升。

## 2. 可重复性已经被量化验证

### 2.1 严格触球前存在三个重复分支

44 条轨迹得到三个重复 core 分支和一个单例：

| 分支 | N | 日期覆盖（r3/r4/r5） | dq 配置（raw/adapter） |
|---|---:|---:|---:|
| B1 | 4 | 1 / 2 / 1 | 1 / 3 |
| B2 | 10 | 4 / 5 / 1 | 4 / 6 |
| B3 | 29 | 8 / 13 / 8 | 11 / 18 |
| BR1 | 1 | 0 / 1 / 0 | 0 / 1 |

分支内距离 P50=`0.679`，分支间距离 P50=`1.282`，相差约 `1.89×`。逐个删除一个输入组再聚类的 mean adjusted Rand index 为 `0.834`：

| 删除的输入组 | ARI |
|---|---:|
| arm q | 0.313 |
| phase duration | 0.689 |
| raw arm dq | 1.000 |
| ball state | 1.000 |
| racket state | 1.000 |
| policy action | 1.000 |

因此分支不是只由球状态、动作或 adapter 某一个字段机械制造。arm `q` 对区分分支有不可替代的信息；其余通道在当前样本中存在较强冗余。

### 2.2 触球前轨迹可以跨日期、跨 dq 配置预测后续失败标签

使用留一最近邻，并做 10,000 次标签置换：

| 最近邻限制 | N | 实际准确率 | null P50 | null P95 | 单侧 p |
|---|---:|---:|---:|---:|---:|
| 不限制 | 44 | **61.4%** | 31.8% | 47.7% | 0.0011 |
| 必须来自不同 record group | 44 | **63.6%** | 31.8% | 45.5% | 0.0004 |
| 必须来自不同 dq cohort | 44 | **70.5%** | 31.8% | 45.5% | <0.0001 |
| 同 record group | 44 | 56.8% | 31.8% | 45.5% | 0.0035 |
| 同 dq cohort | 44 | 54.5% | 31.8% | 45.5% | 0.0066 |

这排除了“只是某一天的调参批次”或“只是 dq adapter 把数据分开”这两个解释。它证明重复失败在第二击结果出现前已经形成局部动力学分支。

边界：部分事件标签（特别是 `high_first_apex`）本来就发生于该阶段，因此 61.4% 本身不能证明因果方向。更强的证据是：**完全由第二击前轨迹定义的 B2/B3，在未参与聚类的第二击后冲量分解上出现显著不同。**

### 2.3 一个重要的负结果

若不使用物理含义明确的事件标签，只对触球结果、轨迹跨度、存活时间和终止族做无监督 phenotype 聚类，重复 core 的 silhouette 只有 `0.290`；前触球轨迹预测该 phenotype 的准确率为 `52.3%`，置换 `p=0.215`，不显著。

因此不能把失败硬切成若干全局、互斥的“漂亮簇”。数据更符合：少数局部轨迹分支最后汇入相同的 `ee_y` 等终点。后续应分析分支转移，不应按终止字符串直接统计根因。

## 3. 两个主要重复分支对应不同的直接丢球机制

### 3.1 B3：高能量、第二击法向主导

B3 覆盖全部三个日期和两种 dq 配置，是最大分支（N=29）：

| 指标 | B3 P50 |
|---|---:|
| 首击后 apex | 1.496 m |
| 首击 outgoing vz | 2.697 m/s |
| 第二击拍面 `normal_y` | +0.182 |
| 第二击法向投影 y | **+0.530 m/s** |
| 第二击未解析切向/残差 y | -0.144 m/s |
| 第二击 outgoing vy | +0.216 m/s |

解释：首击后进入高能量回落；到第二击时，拍面法向已经明显朝 `+Y`。法向项把球推向 `+Y`，负 residual 虽有抵消但通常不够。这个分支与原报告的聚合结论一致，是重复性最高的“姿态—法向冲量”失败通道。

### 3.2 B2：低能量、切向/未解析冲量主导

B2 同样跨三个日期和两种 dq 配置重复出现（N=10）：

| 指标 | B2 P50 |
|---|---:|
| 首击后 apex | 1.305 m |
| 首击 outgoing vz | 0.935 m/s |
| 第二击拍面 `normal_y` | +0.009 |
| 第二击法向投影 y | +0.028 m/s |
| 第二击未解析切向/残差 y | **+0.142 m/s** |
| 第二击 outgoing vy | **+0.392 m/s** |

这里不能继续用“拍面朝 +Y”解释：拍面法向基本中性，+Y 横飞主要落在 residual 项。该项混合了切向冲量、球自旋、胶皮柔顺、真实接触点、球速估计和 FK 误差；现有记录只能定位到这组未解析机制，不能唯一命名其中某一个物理参数。

### 3.3 B2/B3 的后验独立检验

下面的量均来自第二击结果，没有用于严格前触球聚类。差值定义为 B3−B2；置信区间为 10,000 次 bootstrap，p 值为 10,000 次双侧置换并做 Holm 校正：

| 后验指标 | B2 P50 | B3 P50 | 差值 [95% CI] | Holm p |
|---|---:|---:|---:|---:|
| 拍面 `normal_y` | 0.009 | 0.182 | +0.173 [0.112, 0.305] | 0.0008 |
| 法向投影 y (m/s) | 0.028 | 0.530 | +0.502 [0.289, 0.863] | 0.0012 |
| residual y (m/s) | +0.142 | -0.144 | -0.286 [-0.414, -0.177] | 0.0228 |
| outgoing vy (m/s) | +0.392 | +0.216 | -0.175 [-0.362, 0.141] | 0.0515 |

这验证了两种不同机制最后可以产生相似的 `+Y` 丢球：B3 的法向推动更强，B2 的 residual 推动更强。若只看球最后飞向哪里，会把它们错误合并。

### 3.4 两个分支仍共享同一个更早的 robot/plant 起点

record3 的 same-ball replay onset：

| 分支 | record3 N | dq gap P50 | action gap P50 | q gap P50 |
|---|---:|---:|---:|---:|
| B2 | 4 | **0.130 s** | 0.330 s | 0.330 s |
| B3 | 8 | **0.130 s** | 0.240 s | 0.343 s |
| B1 | 1 | 0.135 s | 0.340 s | 0.345 s |

最早的共同异常仍是 `dq/plant`；B2/B3 的差异在闭环后续才形成。因而“角速度观测滤波”不能消除物理执行差异，只会改 actor 看到的其中一条速度通道；这也解释了为什么 raw 与 adapter cohort 都保留相同分支。

## 4. `no_confident_contact` 不是一个干净的物理模态

这个旧标签应改读为“**没有可靠推断到 launch 之后的再次触球**”。launch 本身未被 bout contact count 计数，所以它绝不表示球从未接触球拍。

对全部 181 个已推断复触，用 camera ball state 和 arm-q FK 建立经验触球几何包络；再只检查 11 个零计数 bout 在第一次 apex 之后的下降段。审计结果：

| 子类 | N | 可下的结论 |
|---|---:|---|
| 明显横向几何漏接 | 2 | 球与拍面切向距离超出经验 P99，可作为真实 first-return miss |
| 放松 detector 后出现触球 | 5 | 标签对阈值敏感，不能作为独立物理失败模态 |
| baseline 未计数但 fresh 实测 vz 反转 | 1 | 存在漏计的测量反转，不能叫“未触球” |
| 位于经验触球几何内但没有实测反转 | 3 | camera/FK 空间一致性、真实 EE 位姿、接触面高度或确实未碰到之间不可辨识 |

9/11 会进入已知触球状态的宽 P1–P99 几何包络，但 camera+FK 的几何相交不是物理触球真值。要拆分剩余 3 条，必须增加独立证据：腕部/拍柄冲击传感器或麦克风、高速视频，以及独立末端位姿测量；不能继续通过调 detector 阈值“证明”接触。

## 5. 现在应怎样利用失败模态做验证实验

### 5.1 实物：对每个分支做单因素干预，不以总击数作为唯一指标

1. **共同 plant 层：**按原实物修改说明记录 command、raw q/dq、独立 EE pose，并先完成固定命令/单击 calibration A/B。主指标是 `dq` 最早分叉和 command→q/dq holdout error；只有它们改善，才算修到了共同根。
2. **B3 法向层：**在相同首击 incoming state 分层内，只干预第二击 face-normal `Y`/接触相位。预注册主指标为 `normal_y`、法向投影 y、outgoing vy 和 B3 占比。如果法向项下降而 +Y 丢球不下降，则推翻“B3 法向主导”的实现假设。
3. **B2 residual 层：**不要先盲调摩擦/恢复系数。先同步测球自旋、独立 EE pose/velocity、精确接触点和冲击时刻，再判断 residual 来自真实切向物理还是状态估计误差。
4. **首次回落层：**给 11 类 bout 增加独立 contact truth；将 detector threshold 只作为诊断变量，不作为物理标签。

每次 A/B 必须随机交错运行，按 first incoming position/velocity、record day 和 dq cohort 分层；同时报告分支转移率，而不是只报告平均击数。这样才能判断修改是消灭了根因，还是把 B3 转移成 B2。

### 5.2 仿真：验收目标从“均值像”升级为“分支转移像”

在保留原仿真修改说明的前提下，新增以下验收维度：

- nominal plant 与 fitted real plant 的 B1/B2/B3 进入率；
- first-impact energy 分层后的第二击 `normal_y`、normal-projection-y、residual-y 联合分布；
- 从第一次触球状态初始化的 matched counterfactual：只替换 plant、只替换 impact kernel、两者同时替换；
- branch-conditioned survival，而不只比较总体 mean hits/full survival；
- 仿真若能复现 +Y 终点但把 B2/B3 机制比例复现错，仍判定 sim-to-real 未对齐。

必须保留的识别边界：没有实测 spin/force/compliance/contact point 之前，impact kernel 可以作为能够覆盖 residual 分布的不确定性族训练，但不能声称某个具体摩擦或恢复系数已被真实数据唯一拟合。

## 6. 当前证据等级

| 命题 | 状态 |
|---|---|
| 失败存在可重复的前触球动力学分支 | 已验证：严格 pre-second 聚类、稳定性与跨组置换 |
| 分支不是 record date 或 dq adapter 的批次假象 | 已验证：跨日期、跨 cohort 最近邻仍显著 |
| B3 是第二击法向主导通道 | 已验证到当前观测分辨率：独立后验冲量差异显著 |
| B2 是 residual 主导通道 | 已验证到“未解析 residual”层；具体 spin/摩擦/柔顺来源未识别 |
| plant mismatch 是共同首要 sim-to-real 根因 | 继承原报告的 same-ball onset + autonomous plant A/B；本附录未推翻 |
| plant mismatch 对每次失败都充分 | 已否定；少数长 bout 也有早期 gap |
| 零计数 bout 都是 first-return miss | 已否定；该标签至少混合四类情况 |
| workspace/EE safety 是根因 | 未支持；仍是终止端点，且 `workspace_stop` 与 adapter/date 高度混杂 |

## 7. 可复跑产物

分析脚本：

```text
pingpong_controller/tools/rl_sim/analyze_record_new3_5_failure_modes.py
```

CPU-only 重跑命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/yangzhe/miniconda3/envs/pingpong/bin/python \
pingpong_controller/tools/rl_sim/analyze_record_new3_5_failure_modes.py \
--output-dir pingpong_controller/outputs/rl_sim/record_new3_5_real_failure_rootcause_20260813/failure_modes_cpu
```

主要产物：

```text
failure_modes_cpu/failure_mode_analysis.json
failure_modes_cpu/failure_mode_assignments.csv
failure_modes_cpu/failure_mode_summary.csv
failure_modes_cpu/trajectory_pair_distances.csv
failure_modes_cpu/branch_outcome_contrasts.csv
failure_modes_cpu/first_return_contact_audit.csv
```

本次没有导入 JAX、没有使用 GPU、没有训练、没有运行或写入真实机器人仓库。
