# 第四章正文草稿

## 第三章到第四章的衔接段落

第三章已经在单 UAV 场景下完成了统一任务模型、统一状态动作接口和统一指标体系的构建，并验证了缓存命中、时延、能耗、截止期与可靠性等因素可以在同一框架内被一致刻画。但单 UAV 模型无法反映多 UAV 协同场景下的任务分配、公平性与联合决策问题。基于此，第四章在不改变第三章统一建模口径的前提下，将环境扩展到多 UAV 场景，并要求在 `NUM_UAVS=1` 时尽量退化为第三章模型。这样既保证了章节之间的方法连续性，也为后续研究多 UAV 分配规则与多智能体学习策略提供了统一而可比的实验基础。

## 第四章方法段落草稿

第四章采用共享策略参数的多智能体 PPO 方法，并使用中心化 critic 估计团队状态价值。具体而言，每架 UAV 使用相同的 actor 网络，根据本地观测独立输出二维移动动作；critic 则接收所有 UAV 观测拼接形成的全局状态，用于评估团队回报，从而在训练阶段利用更多全局信息稳定优势估计。在环境内部决策上，本文已经将卸载逻辑统一为 `local / associated UAV / collaborator UAV / BS` 四分支，并在服务未命中时允许执行 UAV 从基站或其他已缓存该服务的邻居 UAV 拉取服务镜像，因此协同服务缓存不再只是概念性设定，而是进入了实际时延计算链路。对于多跳卸载路径，链路可靠性按端到端成功概率联合计算；而 `relay_fetch_energy` 仅统计 UAV 发起的协同中继与 peer fetch 发射能耗，`BS -> UAV` 的服务拉取只进入时延链路而不混入该能耗项。在奖励设计上，本文保持团队奖励机制不变，将任务完成率作为正向激励，并联合考虑缓存命中率、平均时延、增量能耗、截止期违约率、可靠性违约率与动作幅度惩罚，构成统一的 shaped team reward。为抑制无效大幅移动，方法中进一步保留了 movement budget 约束，使动作幅度随用户相对位置与任务紧迫度自适应变化。需要说明的是，第四章不再引入新的机制或第二种学习算法，最终固定采用 `energy_e30` 配置下的 `shared_ppo_centralized_critic` 作为论文主方法。

## 第四章实验设置草稿

实验均基于第三章与第四章共享的统一环境实现展开，冻结 `action / observation / uav_state / episode_log` schema；sensitive profile 不再缩短邻居观测维度，而是沿用相同 schema 并对缺失邻居槽位做零填充。PPO 主配置固定为：`train_episodes=30`、`actor_lr=2.5e-4`、`critic_lr=8.0e-4`、`clip_ratio=0.18`、`entropy_coef=0.008`、`value_coef=0.6`、`reward_energy_weight=1.5`、`reward_action_magnitude_weight=0.2`、`use_movement_budget=True`。最终复跑采用训练种子 `seed={42, 52, 62}` 三组随机种子，并在评估阶段固定使用对应 held-out 种子 `{142, 152, 162}`；评估 episode 数固定为 `4`。主实验矩阵包括三部分：其一，验证第三章与第四章在 `NUM_UAVS=1` 下的一致性；其二，在敏感配置下比较 `nearest_uav` 与 `least_loaded_uav` 两种任务分配规则；其三，在 `NUM_UAVS=2` 和 `NUM_UAVS=3` 场景下比较 PPO 主方法与 heuristic baseline。消融实验固定为两项：去除 energy-shaped reward，以及去除 movement budget，其余设置均与主配置保持一致。第三章方面，在启发式控制之外，本文补入了一个基于统一观测的 MPC shell，用于体现“单 UAV 优化器壳子”这一章节定位。

## 第四章结果分析草稿

从一致性验证看，第三章与第四章在 `NUM_UAVS=1` 下的 `completion_rate`、`average_latency`、`total_energy`、`cache_hit_rate` 以及新增能耗拆分指标在 3 个种子上均保持 `delta=0.0000 +/- 0.0000`，而 `fairness_uav_load` 由于单 UAV 场景不适用，被保留为 `null` 而不是伪造为 `0.0`，说明第四章方法依然建立在第三章统一模型之上，而不是重新构造一套不兼容环境。分配规则对比表明，在修正共享环境 v2 之后，敏感配置下 `least_loaded_uav` 与 `nearest_uav` 已重新拉开差异。以最新 3-seed 汇总为例，`NUM_UAVS=2` 时 `least_loaded_uav` 的 `total_energy=130.1174 +/- 0.5214`，低于 `nearest_uav` 的 `134.1836 +/- 10.0652`；`NUM_UAVS=3` 时两者分别为 `149.7891 +/- 4.5289` 与 `182.0858 +/- 1.9352`。主方法与 heuristic 的对比则表明，PPO 在修正后的最终结果中并未继续保持此前草稿里的能耗优势。根据当前验证记录，`NUM_UAVS=2` 时 PPO 的 `completion_rate=0.9846 +/- 0.0137`、`average_latency=0.4082 +/- 0.0540`、`total_energy=28.6199 +/- 7.2119`，而 heuristic 的 `total_energy=16.5777 +/- 16.3382`；`NUM_UAVS=3` 时 PPO 的 `completion_rate=0.9931 +/- 0.0120`、`average_latency=0.4314 +/- 0.0427`、`total_energy=46.2983 +/- 15.4309`，heuristic 的 `total_energy=40.6808 +/- 28.0994`。因此，修正后的代码结果更适合表述为：当前 PPO 主方法已经形成稳定可复现实验闭环，并验证了 movement budget 等机制的作用，但在现有训练预算下尚未整体优于 heuristic 基线。

## 第四章消融分析草稿

消融结果表明，当前实现中 movement budget 的作用依然非常明确，但 energy-shaped reward 的收益需要更谨慎表述。最新 3-seed 汇总下，主配置的 `total_energy=28.6199 +/- 7.2119`；去除 energy-shaped reward 后，能耗为 `27.7935 +/- 11.4441`，并未出现稳定恶化；进一步去除 movement budget 后，能耗则显著升高到 `111.0782 +/- 7.9667`。这说明在当前训练预算与奖励权重下，energy-shaped reward 至少还没有形成稳定、单调的能耗优势，而 movement budget 对抑制无效大范围移动仍然是最关键的机制。与此同时，三种设置下的完成率与时延差异整体有限，因此更稳妥的论文表述应当是：movement budget 已被清晰验证有效，energy-shaped reward 则提供了一种可继续调参和改进的优化方向，但暂不宜写成已经被最终结果充分证明的正向机制。

## 第四章小结草稿

本章在第三章统一建模框架基础上，将单 UAV 场景扩展到多 UAV 协同场景，并在不改变统一接口与统一指标体系的前提下，引入了共享 actor 与中心化 critic 的 PPO 主方法。进一步地，本文已经将 collaborator-UAV 卸载分支与 peer UAV service fetch 纳入统一环境，使“协同服务缓存”真正进入多 UAV 时延与完成率主线。实验结果表明，该方法在 `NUM_UAVS=1` 时能够保持与第三章一致，在多 UAV 场景下则形成了完整、可复现的多智能体训练与评估链路。虽然当前 PPO 主方法在最终多 seed 指标上尚未整体优于 heuristic baseline，但消融实验已经清晰验证 movement budget 的必要性，并表明 energy-shaped reward 仍有继续调参和改进空间。由此可见，本文构建的第四章方法既延续了第三章的统一框架，又提供了一个可继续改进的多 UAV 协同学习基线，为后续更复杂的多智能体 UAV-MEC 研究提供了统一实验平台。
