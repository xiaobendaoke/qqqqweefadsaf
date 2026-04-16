# 第四章正文草稿

## 第三章到第四章的衔接段落

第三章已经在单 UAV 场景下完成了统一任务模型、统一状态动作接口和统一指标体系的构建，并验证了缓存命中、时延、能耗、截止期与可靠性等因素可以在同一框架内被一致刻画。但单 UAV 模型无法反映多 UAV 协同场景下的任务分配、公平性与联合决策问题。基于此，第四章在不改变第三章统一建模口径的前提下，将环境扩展到多 UAV 场景，并要求在 `NUM_UAVS=1` 时尽量退化为第三章模型。这样既保证了章节之间的方法连续性，也为后续研究多 UAV 分配规则与多智能体学习策略提供了统一而可比的实验基础。

## 第四章方法段落草稿

第四章采用共享策略参数的多智能体 PPO 方法，并使用中心化 critic 估计团队状态价值。具体而言，每架 UAV 使用相同的 actor 网络，根据本地观测独立输出二维移动动作；critic 则接收所有 UAV 观测拼接形成的全局状态，用于评估团队回报，从而在训练阶段利用更多全局信息稳定优势估计。在环境内部决策上，本文已经将卸载逻辑统一为 `local / associated UAV / collaborator UAV / BS` 四分支，并在服务未命中时允许执行 UAV 从基站或其他已缓存该服务的邻居 UAV 拉取服务镜像，因此协同服务缓存不再只是概念性设定，而是进入了实际时延计算链路。对于多跳卸载路径，链路可靠性按端到端成功概率联合计算；而 `relay_fetch_energy` 仅统计 UAV 发起的协同中继与 peer fetch 发射能耗，`BS -> UAV` 的服务拉取只进入时延链路而不混入该能耗项。在奖励设计上，本文保持团队奖励机制不变，将任务完成率作为正向激励，并联合考虑缓存命中率、平均时延、增量能耗、截止期违约率与可靠性违约率；同时，为抑制无效大幅移动，方法中保留了 movement budget 约束，使动作幅度随用户相对位置与任务紧迫度自适应变化。需要说明的是，第四章不再引入新的机制或第二种学习算法，最终固定采用 `freeze_noshaping_240` 配置下的 `shared_ppo_centralized_critic` 作为论文主方法，即保留低探索、小步更新与 movement budget，但不再把显式能耗项和动作幅度罚项作为最终主配置中的必要条件。

## 第四章实验设置草稿

实验均基于第三章与第四章共享的统一环境实现展开，冻结 `action / observation / uav_state / episode_log` schema；sensitive profile 不再缩短邻居观测维度，而是沿用相同 schema 并对缺失邻居槽位做零填充。PPO 主配置固定为：`train_episodes=240`、`actor_lr=8.0e-5`、`critic_lr=5.0e-4`、`clip_ratio=0.10`、`entropy_coef=0.0003`、`value_coef=0.82`、`action_std_init=0.04`、`action_std_min=0.005`、`action_std_decay=0.984`、`reward_energy_weight=0.0`、`reward_action_magnitude_weight=0.0`、`use_movement_budget=True`。最终复跑采用训练种子 `seed={42, 52, 62}` 三组随机种子，并在评估阶段固定使用对应 held-out 种子 `{142, 152, 162}`；评估 episode 数固定为 `64`。主实验矩阵包括三部分：其一，验证第三章与第四章在 `NUM_UAVS=1` 下的一致性；其二，在敏感配置下比较 `nearest_uav` 与 `least_loaded_uav` 两种任务分配规则；其三，在 `NUM_UAVS=2` 和 `NUM_UAVS=3` 场景下比较 PPO 主方法与 heuristic baseline。消融实验固定为两项：在主配置基础上加回 reward shaping，以及去除 movement budget，其余设置均与主配置保持一致。第三章方面，在启发式控制之外，本文补入了一个基于统一观测的 MPC shell，用于体现“单 UAV 优化器壳子”这一章节定位。

## 第四章结果分析草稿

从一致性验证看，第三章与第四章在 `NUM_UAVS=1` 下的 `completion_rate`、`average_latency`、`total_energy`、`cache_hit_rate` 以及新增能耗拆分指标在 3 个种子上均保持 `delta=0.0000 +/- 0.0000`，而 `fairness_uav_load` 由于单 UAV 场景不适用，被保留为 `null` 而不是伪造为 `0.0`，说明第四章方法依然建立在第三章统一模型之上，而不是重新构造一套不兼容环境。分配规则对比表明，在当前高预算复跑下，`least_loaded_uav` 仍然主要体现为更高的负载均衡公平性，而非稳定的总能耗优势。以最新 3-seed 汇总为例，`NUM_UAVS=2` 时 `nearest_uav` 与 `least_loaded_uav` 的总能耗分别为 `14.3645 +/- 0.0574` 与 `14.4095 +/- 0.0632`，而 `fairness_uav_load` 则分别为 `0.9962 +/- 0.0006` 与 `0.9981 +/- 0.0002`；`NUM_UAVS=3` 时两者总能耗分别为 `14.8721 +/- 0.0748` 与 `15.0281 +/- 0.0773`，对应的 `fairness_uav_load` 分别为 `0.9851 +/- 0.0003` 与 `0.9909 +/- 0.0002`。主方法与 heuristic 的对比则表明，采用更长训练预算、低探索和 movement budget 的 `freeze_noshaping_240` 配置后，学习策略已经进一步压低了无效移动，但总能耗仍高于 heuristic 基线。具体而言，`NUM_UAVS=2` 时 PPO 的 `completion_rate=0.9988 +/- 0.0000`、`average_latency=0.3711 +/- 0.0036`、`total_energy=13.7229 +/- 2.3373`，heuristic 的 `total_energy=6.1702 +/- 0.3617`；`NUM_UAVS=3` 时 PPO 的 `completion_rate=0.9996 +/- 0.0004`、`average_latency=0.3531 +/- 0.0031`、`total_energy=16.9553 +/- 2.5856`，heuristic 的 `total_energy=5.1814 +/- 1.0311`。进一步分解能耗可见，PPO 仍然主要受 `uav_move_energy` 拖累：`NUM_UAVS=2` 时该项均值为 `10.6207`，而 heuristic 仅为 `3.0679`；`NUM_UAVS=3` 时 PPO 为 `13.8578`，heuristic 为 `2.0824`。因此，当前代码结果更适合表述为：高预算低探索 PPO 已经显著缩小了与 heuristic 的能耗差距，并形成了稳定可复现实验闭环，但最终瓶颈仍集中在移动能耗控制上。

## 第四章消融分析草稿

消融结果表明，当前实现中 movement budget 的作用依然非常明确，而显式 reward shaping 并未带来最终主结果上的稳定收益。最新 3-seed 汇总下，主配置 `freeze_noshaping_240` 的 `total_energy=13.7229 +/- 2.3373`；在相同训练预算与低探索设定下加回 `reward_energy_weight=2.0` 与 `reward_action_magnitude_weight=1.0` 后，能耗上升到 `14.0853 +/- 2.2260`；进一步去除 movement budget 后，能耗则显著升高到 `60.3768 +/- 17.5405`。这说明在当前长预算训练设定下，movement budget 仍然是抑制无效大范围移动的关键机制，而显式能耗项与动作幅度罚项至少在当前参数组合下并未形成稳定、单调的附加收益。与此同时，三种设置下的完成率与时延差异整体较小，因此更稳妥的论文表述应当是：movement budget 已被清晰验证有效，reward shaping 则提供了一个值得继续调参与重设计的优化方向，但暂不宜写成已经被最终结果充分证明的正向机制。

## 第四章小结草稿

本章在第三章统一建模框架基础上，将单 UAV 场景扩展到多 UAV 协同场景，并在不改变统一接口与统一指标体系的前提下，引入了共享 actor 与中心化 critic 的 PPO 主方法。进一步地，本文已经将 collaborator-UAV 卸载分支与 peer UAV service fetch 纳入统一环境，使“协同服务缓存”真正进入多 UAV 时延与完成率主线。实验结果表明，该方法在 `NUM_UAVS=1` 时能够保持与第三章一致，在多 UAV 场景下则形成了完整、可复现的多智能体训练与评估链路。采用长预算、低探索并保留 movement budget 的 `freeze_noshaping_240` 主配置后，PPO 已经进一步收敛了无效移动行为，并在保持完成率与时延基本持平的同时继续缩小了与 heuristic 的能耗差距；不过在最终多 seed 指标上，它仍未整体优于 heuristic baseline。由此可见，本文构建的第四章方法既延续了第三章的统一框架，又提供了一个经过高预算调参后更稳定的多 UAV 协同学习基线，为后续进一步改进移动控制与奖励设计提供了统一实验平台。
