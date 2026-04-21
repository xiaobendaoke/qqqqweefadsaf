# 第四章正文草稿

> 注：当前结果目录约定为：stage5 产物写入 `第四章/results/stage5/paper_stage5_v2/`，stage6 终稿复跑产物写入 `第四章/results/stage6/paper_stage6_v2/`。主候选仍以 `paper_stage5_v2/tuning_summary.json` 的 `selected_candidate` 为准，当前自动选中为 `freeze_energy2_240`。

## 第三章到第四章的衔接段落

第三章已经在单 UAV 场景下完成了统一任务模型、统一状态动作接口和统一指标体系的构建，并验证了缓存命中、时延、能耗、截止期与可靠性等因素可以在同一框架内被一致刻画。但单 UAV 模型无法反映多 UAV 协同场景下的任务分配、公平性与联合决策问题。基于此，第四章在不改变第三章统一建模口径的前提下，将环境扩展到多 UAV 场景，并要求在 `NUM_UAVS=1` 时尽量退化为第三章模型。这样既保证了章节之间的方法连续性，也为后续研究多 UAV 分配规则与多智能体学习策略提供了统一而可比的实验基础。

## 第四章方法段落草稿

第四章采用共享策略参数的多智能体 PPO 方法，并使用中心化 critic 估计团队状态价值。具体而言，每架 UAV 使用相同的 actor 网络，根据本地观测独立输出 joint action：连续 mobility、对固定 `task_slots x candidate_plans` 的 masked offloading selection，以及连续 cache priority 分数；critic 则接收 joint observation summaries，用于评估团队回报，从而在训练阶段利用更多全局信息稳定优势估计。需要明确的是：当前实现已经进入 candidate-based joint control，而不是旧的“只学习移动”；但它仍不是在全任务空间上无约束地产生调度方案。环境会先枚举 local / associated UAV / collaborator UAV / BS 四类候选，再由策略在固定暴露的 candidate set 中选择 `candidate_plan_id` 或 `defer`。当目标 UAV 未缓存所需服务时，当前实现只在基站与其他已缓存该服务的邻居 UAV 之间比较 fetch 时延，再把选中的 fetch 链路并入整体时延与可靠性检查。对于多跳卸载路径，链路可靠性按端到端成功概率联合计算；在链路等待建模上，当前实现采用 shared resource-class serial queue approximation：所有接入传输共享 `edge_access` 串行资源，所有中继与服务回源链路共享 `backhaul` 串行资源。在能耗分解上，`relay_fetch_energy` 统计 UAV 发起的协同中继与 peer fetch 发射能耗，`bs_fetch_tx_energy` 单列统计 `BS -> UAV` 的服务回源传输能耗，两者共同进入系统总能耗。在奖励设计上，当前实现已经将 episode 级展示指标与 step 级训练信号显式分离：训练奖励使用 step-level proxy reward，并显式纳入 `completion / cache_hit / backlog / expired / latency / energy / deadline / reliability / invalid_action / infeasible_action` 等项；其中完成项使用基于 finalized task 的 step completion ratio，能耗项采用 step-level reference scale 做归一化。该 proxy 与最终 episode 指标方向对齐，但并非完全相同的数值函数。同时，为抑制无效大幅移动，方法中保留了 movement budget 约束，使动作幅度随用户相对位置与任务紧迫度自适应变化。需要说明的是，第四章主配置不应再写成“硬编码固定候选”，而应表述为：主候选由 `paper_stage5_v2` 调参汇总规则自动选出，必要时才通过显式参数覆盖。

## 第四章实验设置草稿

实验均基于第三章与第四章共享的统一环境实现展开，冻结 `action / observation / uav_state / episode_log` schema；同时在日志中补充 `metric_schemas`，显式区分 episode 主指标与 step 训练信号。sensitive profile 不再缩短邻居观测维度，而是沿用相同 schema 并对缺失邻居槽位做零填充。PPO 调参阶段采用训练种子 `seed={42, 52, 62}` 与对应调参评估种子 `{142, 152, 162}`；终稿复跑采用与调参阶段不重叠的训练种子 `seed={72, 82, 92}` 与对应最终评估种子 `{172, 182, 192}`；评估 episode 数固定为 `64`。主实验矩阵包括三部分：其一，验证第三章与第四章在 `NUM_UAVS=1` 下的一致性；其二，在敏感配置下比较 `nearest_uav` 与 `least_loaded_uav` 两种环境侧任务分配规则；其三，在 `NUM_UAVS=2` 和 `NUM_UAVS=3` 场景下比较 joint RL 与 joint heuristic baseline，并额外保留 mobility-only RL/heuristic 作为 legacy 对照。这里第二部分应明确写成“固定 UAV/UE 布局下、基于 heuristic 控制的 assignment-rule 对比”，而不是 PPO 对 assignment 规则的泛化结论。消融实验固定为两项：在自动选出的主配置基础上增强能耗/动作正则，以及去除 movement budget，其余设置均与主配置保持一致。第三章方面，在启发式控制之外，本文补入了一个基于统一观测的 look-ahead heuristic shell，用于体现“单 UAV 优化器壳子”这一章节定位；正文中应把 `MPC shell` 写成候选动作滚动打分的 heuristic shell，而不是标准约束 MPC 求解器。当前主配置的具体超参数应直接引用 `第四章/results/stage5/paper_stage5_v2/tuning_summary.json` 中自动选出的 `freeze_energy2_240` 及其 overrides，而不是手填到正文里。

## 第四章结果分析草稿

从一致性验证看，第三章与第四章在 `NUM_UAVS=1` 下的 `completion_rate`、`average_latency`、`total_energy`、`cache_hit_rate` 以及新增能耗拆分指标继续保持逐项对齐，而 `fairness_uav_load` 在单 UAV 场景中仍保留为 `null`，而不是伪造为 `0.0`。分配规则、主方法对比与能耗分解的具体数值应直接引用 `第四章/results/stage6/paper_stage6_v2/assignment_multiseed_summary.json`、`ppo_vs_heuristic_multiseed_summary.json`、`ablation_multiseed_summary.json` 与对应表格/图片自动生成的均值和标准差。当前更稳妥的结果分析写法应强调三点：第一，多 UAV 结果必须与 `NUM_UAVS=1` 退化一致性共同报告；第二，assignment 规则实验是在固定 UAV 布局下、用 heuristic 控制完成的，结论宜限定为“固定布局环境下的关联规则差异”；第三，当前完整复跑结果下，若要讨论能耗收益，必须直接引用新复跑日志里的 `step_energy_norm`、总能耗主指标与 joint heuristic 对比结果，而不能再沿用旧 mobility-only 口径或历史根目录结果。

## 第四章消融分析草稿

消融结果表明，movement budget 与 reward shaping 应被分开讨论。终稿里应直接引用 `第四章/results/stage6/paper_stage6_v2/ablation_multiseed_summary.json` 自动生成的汇总值，并把结论限定为：movement budget 是否稳定有效，以新复跑结果为准；reward shaping 是否改善总能耗与 reject 行为，也应以新复跑日志为准，而不再沿用旧 mobility-only 结论。

## 第四章小结草稿

本章在第三章统一建模框架基础上，将单 UAV 场景扩展到多 UAV 协同场景，并在不改变统一接口与统一指标体系的前提下，引入了共享 actor 与中心化 critic 的 PPO 主方法。进一步地，本文已经将 collaborator-UAV 卸载分支与 peer UAV service fetch 纳入统一环境，使协同服务缓存进入多 UAV 时延与完成率计算链路；但更准确的表述应是“固定环境侧卸载/缓存逻辑上的多智能体移动控制”，而不是“联合学习卸载、缓存与轨迹优化”。当前更稳妥的终稿表述应当是：第四章首先提供了一个与第三章可对齐、可复跑、并且把 episode 指标与训练信号明确拆开的统一实验平台；至于 PPO 是否优于 heuristic、reward shaping 是否有效、assignment 规则是否具有泛化优势、总能耗是否有优化增益，都应在 `paper_stage6_v2` 完整复跑后再据实写入，而不再沿用历史结果目录中的旧结论。
