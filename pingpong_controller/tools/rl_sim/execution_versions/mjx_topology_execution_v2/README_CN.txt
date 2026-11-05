命名版本：MJX训练加速V2
固定版本ID：mjx_topology_execution_v2
训练启用值：topology_v2
保存日期：2026-11-04

用途
保留已经验证的训练执行加速方案，供后续训练明确选择是否启用。
默认关闭，不根据resume检查点自动开启。以后可直接称“启用训练加速V2”或“关闭训练加速”。
该名称指执行方案，不改变训练课程、动作/观测含义、奖励、DR或检查点身份。

本目录保存3份核心源码、3份当时的训练器/启动器集成参考及SHA-256清单。
versions.json锁定manifest.json的哈希。后续优化创建新版本ID，保留此版本和源码快照。
integration/中的文件仅用于保存和对照；实际训练使用rl_sim目录的已审核启动器。
当前训练执行代码继续使用已经通过GPU检查的实现，本次归档不修改其计算逻辑。

选择方式
V322_MJX_EXECUTION_MODE=legacy，V322_MJX_LINE_SEARCH_SCHEDULE=legacy：关闭全部执行优化。
V322_MJX_EXECUTION_MODE=legacy，V322_MJX_LINE_SEARCH_SCHEDULE=chunk5_v1：使用前一版行搜索优化。
V322_MJX_EXECUTION_MODE=topology_v2，V322_MJX_LINE_SEARCH_SCHEDULE=legacy：启用本命名版本。
V2已经包含chunk5调度，两个优化开关不能叠加。
直接调用训练器时，使用--mjx-execution-mode topology_v2；关闭时为legacy。

当前V322的推荐使用方式
以下命令在pingpong_controller/tools/rl_sim/目录执行。V322_CHECKPOINT须明确设为兼容的
V322检查点，RUN_ID须替换为新的唯一身份；不要复用已有输出目录或W&B身份。
正式训练仍须在pp_gpu1会话中执行，并通过原GPU/内存/温度预检。

启用：
  V322_MJX_LINE_SEARCH_SCHEDULE=legacy \
    bash run_gpu1_fixed_plane_v322_execution_v2.sh --formal RUN_ID "$V322_CHECKPOINT"

关闭，保留相同训练预设，仅改变执行选项：
  V322_REWARD_PRESET=late_center_precision \
  V322_LOSS_PRESET=anchor1 \
  V322_GATE_PRESET=deployment_groups \
  V322_MJX_EXECUTION_MODE=legacy \
  V322_MJX_LINE_SEARCH_SCHEDULE=legacy \
    bash run_gpu1_fixed_plane_v322_long_episode.sh --formal RUN_ID "$V322_CHECKPOINT"

完整性检查（只读，不启动GPU训练）：
  /home/yangzhe/miniconda3/envs/pingpong/bin/python -B \
    verify_training_execution_version.py --version mjx_topology_execution_v2 \
    --check-current --check-installed

不带--check-current时，仅核验已保存的版本快照。当前源码未来增加其他版本后，集成参考可能
出现有意差异；“当前源码不同”和“版本快照损坏”分别报告，不能混为一谈。

保留的方案
1. 固定拓扑质量矩阵掩码缓存为常量，保留原浮点矩阵运算顺序。
2. 静态证明contact区域合法、在界内且互不重叠，再使用唯一索引更新提示。
3. 两个安全投影循环按24步展开，仍完整执行各自576次顺序投影。
4. 保留既有chunk5行搜索调度和原100/50求解上限。
失败的重复索引重排、变慢的摩擦布局提示和GPU图循环候选均不属于此版本。

已经验证的范围
V322：late_center_precision / anchor1 / deployment_groups；72/393/7维actor/critic/action。
JAX0.10.1、MuJoCo3.8.1及manifest列出的原生源码哈希；1024环境×128步，物理1ms。
两种子26次真实更新/832Adam中，每轮全部1603数值叶子逐字节一致。
最终入口另完成3次更新/96Adam，晚期和有reset输入重放一致；364项CPU测试通过。
完整采样均值10.287->6.493秒，耗时降低约36.9%；整轮10.619->6.829秒。
没有重新完成完整收敛课程或1200秒资格测试；现有证据不构成新的策略/部署认证。
原始证据：pingpong_controller/outputs/rl_sim/v322_targeted_execution_20261104_r1/。

用于后续新训练方案
同一兼容V322配置可以通过上述选项直接选择。其他训练器、课程、模型、GPU或依赖版本必须
先做兼容检查、固定种子逐字节配对和完整GPU采样测速，再接入这一命名执行版本。
不得为方便启用而放宽现有V322预设、库版本、原生源码和索引证明检查。
新训练方案保留显式的关闭选项，并在训练参数和检查点元数据记录执行版本ID。
若需要改变本版本的算法或来源文件，使用新的单调递增版本名，保留V2。
