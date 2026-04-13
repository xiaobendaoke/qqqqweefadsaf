# 第四章正文草稿

## 第三章到第四章的衔接段落

第三章已经在单 UAV 场景下完成了统一任务模型、统一状态动作接口和统一指标体系的构建，并验证了缓存命中、时延、能耗、截止期与可靠性等因素可以在同一框架内被一致刻画。但单 UAV 模型无法反映多 UAV 协同场景下的任务分配、公平性与联合决策问题。基于此，第四章在不改变第三章统一建模口径的前提下，将环境扩展到多 UAV 场景，并要求在 `NUM_UAVS=1` 时尽量退化为第三章模型。这样既保证了章节之间的方法连续性，也为后续研究多 UAV 分配规则与多智能体学习策略提供了统一而可比的实验基础。

## 第四章方法段落草稿

第四章采用共享策略参数的多智能体 PPO 方法，并使用中心化 critic 估计团队状态价值。具体而言，每架 UAV 使用相同的 actor 网络，根据本地观测独立输出二维移动动作；critic 则接收所有 UAV 观测拼接形成的全局状态，用于评估团队回报，从而在训练阶段利用更多全局信息稳定优势估计。在环境内部决策上，本文已经将卸载逻辑统一为 `local / associated UAV / collaborator UAV / BS` 四分支，并在服务未命中时允许执行 UAV 从基站或其他已缓存该服务的邻居 UAV 拉取服务镜像，因此协同服务缓存不再只是概念性设定，而是进入了实际时延计算链路。在奖励设计上，本文保持团队奖励机制不变，将任务完成率作为正向激励，并联合考虑缓存命中率、平均时延、增量能耗、截止期违约率、可靠性违约率与动作幅度惩罚，构成统一的 shaped team reward。为抑制无效大幅移动，方法中进一步保留了 movement budget 约束，使动作幅度随用户相对位置与任务紧迫度自适应变化。需要说明的是，第四章不再引入新的机制或第二种学习算法，最终固定采用 `energy_e30` 配置下的 `shared_ppo_centralized_critic` 作为论文主方法。

## 第四章实验设置草稿

实验均基于第三章与第四章共享的统一环境实现展开，冻结 `action / observation / uav_state / episode_log` schema。PPO 主配置固定为：`train_episodes=30`、`actor_lr=2.5e-4`、`critic_lr=8.0e-4`、`clip_ratio=0.18`、`entropy_coef=0.008`、`value_coef=0.6`、`reward_energy_weight=1.5`、`reward_action_magnitude_weight=0.2`、`use_movement_budget=True`。最终复跑采用 `seed={42, 52, 62}` 三组随机种子，评估 episode 数固定为 `4`。主实验矩阵包括三部分：其一，验证第三章与第四章在 `NUM_UAVS=1` 下的一致性；其二，在敏感配置下比较 `nearest_uav` 与 `least_loaded_uav` 两种任务分配规则；其三，在 `NUM_UAVS=2` 和 `NUM_UAVS=3` 场景下比较 PPO 主方法与 heuristic baseline。消融实验固定为两项：去除 energy-shaped reward，以及去除 movement budget，其余设置均与主配置保持一致。第三章方面，在启发式控制之外，本文补入了一个基于统一观测的 MPC shell，用于体现“单 UAV 优化器壳子”这一章节定位。

## 第四章结果分析草稿

从一致性验证看，第三章与第四章在 `NUM_UAVS=1` 下的 `completion_rate`、`average_latency`、`total_energy`、`cache_hit_rate` 等指标在 3 个种子上均保持 `delta=0.0000 +/- 0.0000`，说明第四章方法建立在第三章统一模型之上，而不是重新构造一套不兼容环境。分配规则对比表明，在引入 collaborator-UAV 与 peer-fetch 后，敏感配置下两种 assignment 规则的整体表现更加接近，但 `least_loaded_uav` 依然略优于 `nearest_uav`。例如在 `NUM_UAVS=2` 时，两者的 `fairness_uav_load` 都接近 `0.9978`；在 `NUM_UAVS=3` 时，两者都达到约 `0.9809`。从主方法与 heuristic 的对比结果看，二者在三组种子上的任务完成率均保持 `1.0000 +/- 0.0000`，说明主方法没有牺牲任务完成能力。时延方面，PPO 与 heuristic 基本持平，`NUM_UAVS=2` 时分别为 `0.2701 +/- 0.0089` 与 `0.2701 +/- 0.0089`，`NUM_UAVS=3` 时分别为 `0.2710 +/- 0.0031` 与 `0.2710 +/- 0.0031`。能耗方面，PPO 在三组种子上的均值分别为 `68.7092 +/- 11.7735` 和 `105.7111 +/- 36.3220`，对应 heuristic 为 `81.4671 +/- 47.4514` 和 `123.0361 +/- 71.0983`。这表明在补入协同卸载与 peer fetch 后，PPO 在多 seed 均值上继续保持优于 heuristic 的能耗水平，且 `NUM_UAVS=2` 时波动较此前更小。

## 第四章消融分析草稿

消融结果说明 energy-shaped reward 与 movement budget 均对主方法有效。主配置下的 `total_energy` 为 `68.7092 +/- 11.7735`。去除 energy-shaped reward 后，能耗上升至 `84.3502 +/- 44.7163`，且均值已略高于 heuristic 的 `81.4671 +/- 47.4514`，说明显式能耗惩罚对于引导策略学习低能耗协同移动是必要的。进一步去除 movement budget 后，能耗增至 `73.4761 +/- 13.3339`，虽然仍低于 heuristic，但相较主配置已有明显恶化，表明动作约束对抑制无效大幅移动依然关键。两项消融下，任务完成率仍维持在 `1.0000 +/- 0.0000`，而平均时延变化很小，说明本章提出的两项设计主要作用于能耗控制而不是任务完成能力本身。综合来看，energy-shaped reward 提供了优化方向，而 movement budget 进一步将这一方向落实为更稳定的动作幅度控制。

## 第四章小结草稿

本章在第三章统一建模框架基础上，将单 UAV 场景扩展到多 UAV 协同场景，并在不改变统一接口与统一指标体系的前提下，引入了共享 actor 与中心化 critic 的 PPO 主方法。进一步地，本文已经将 collaborator-UAV 卸载分支与 peer UAV service fetch 纳入统一环境，使“协同服务缓存”真正进入多 UAV 时延与完成率主线。实验结果表明，该方法在 `NUM_UAVS=1` 时能够保持与第三章一致，在多 UAV 场景下能够与 heuristic baseline 保持相当的时延与完成率，并在多 seed 均值上实现更低的能耗表现。消融实验进一步验证了 energy-shaped reward 与 movement budget 对降低能耗具有直接作用。由此可见，本文构建的第四章方法既延续了第三章的统一框架，又具备面向多 UAV 协同决策的扩展能力，为后续更复杂的多智能体 UAV-MEC 研究提供了统一基线。
