# 第四章

本目录是基于统一任务模型与统一指标体系构建的多 UAV 扩展版实现。第四章当前会优先从 `results/stage5/paper_stage5_v2/tuning_summary.json` 自动读取主配置候选；若未提供 stage-5 v2 调参结果，则回退到 `freeze_energy2_240`。当前 candidate-based joint system 已经落地：RL actor 同时输出 `mobility / offloading / caching` 三部分动作，但 offloading 仍是在固定 `task_slots x candidate_plans` 空间上做 masked selection，而不是在全任务空间上无约束地端到端生成调度方案。

## 目录说明

- `chapter4/`: 多 UAV 环境、启发式策略、PPO 训练与评估实现
- `run_experiment.py`: heuristic baseline 实验入口
- `run_train_marl.py`: 单次 PPO 训练入口
- `run_eval_marl.py`: 单次 PPO 评估入口
- `run_paper_experiments.py`: 第五阶段论文实验矩阵与消融入口
- `run_finalize_paper.py`: 第六阶段终稿复跑、均值/标准差统计与结果包入口
- `results/legacy/`: legacy mobility-only baseline、legacy RL 与相关 smoke / episode
- `results/joint/`: joint heuristic、joint RL 与相关训练 / 评估结果
- `results/stage5/`: stage-5 调参与主配置选择
- `results/stage6/`: stage-6 终稿复跑、验证刷新与结果包输出根目录
- `VERIFICATION.md`: 已执行命令、关键日志与结果路径
- `requirements.txt`: 本章依赖说明

## 环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r 第四章/requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})"
```

默认 `device=auto`。安装 CUDA 版 `torch` 且 `torch.cuda.is_available()` 为 `True` 时，第四章 MARL 训练与评估会自动落到 `cuda:0`；也可以通过 `--device auto|cpu|cuda|cuda:0` 显式指定设备。

注意：PPO 网络前向、反向与 batch 更新会使用 GPU，但统一环境仿真、任务调度和启发式 baseline 仍然是 Python 侧 CPU 逻辑，因此整体耗时不一定会与 GPU 利用率等比例变化。

## 一键复现实验

先验证 `NUM_UAVS=1` 退化一致性：

```powershell
.\.venv\Scripts\python.exe 第三章/run_experiment.py --episodes 1 --compare-ch4 --seed 42
.\.venv\Scripts\python.exe 第四章/run_train_marl.py --seed 42 --train-episodes 240 --num-uavs 2 --assignment-rule nearest_uav --device auto
```

再复现 stage5 三 seed 调参与主配置选择：

```powershell
.\.venv\Scripts\python.exe 第四章/run_paper_experiments.py --tuning-seeds 42 52 62 --eval-episodes 64 --device auto
```

再复现最终论文结果包：

```powershell
.\.venv\Scripts\python.exe 第四章/run_finalize_paper.py --seeds 72 82 92 --eval-episodes 64 --output-dir-name paper_stage6_v2 --device auto
```

说明:

- stage5 调参默认使用训练种子 `42/52/62` 与对应调参评估种子 `142/152/162`；当前 `results/stage5/paper_stage5_v2/tuning_summary.json` 已按真实三 seed 聚合。
- stage6 最终报告默认使用与 stage5 不重叠的训练种子 `72/82/92`，并在评估阶段固定采用 `172/182/192`。
- `observation / uav_state / episode_log` schema 在主实验与 sensitive profile 下保持一致；缺失邻居槽位使用零填充。

## 主配置选择

- `run_paper_experiments.py` 会按调参汇总规则自动选择主候选；也可以用 `--selected-candidate <name>` 显式覆盖。
- `run_finalize_paper.py` 会优先读取 `results/stage5/paper_stage5_v2/tuning_summary.json` 中的 `selected_candidate`；若该文件不存在，则退回内置保底配置。
- 当前训练日志、评估日志与 episode log 已写入 `metric_schemas`，用来区分 episode 主指标与按生成/积压/终局事件对齐的 step 训练信号。
- 当前最新自动选中候选：`freeze_energy2_240`。

## 当前默认保底/自动主配置

- `train_episodes=240`
- `actor_lr=0.00008`
- `critic_lr=0.0005`
- `clip_ratio=0.10`
- `entropy_coef=0.0003`
- `value_coef=0.82`
- `action_std_init=0.04`
- `action_std_min=0.005`
- `action_std_decay=0.984`
- `reward_completion_weight=1.0`
- `reward_cache_hit_weight=0.10`
- `reward_latency_weight=0.25`
- `reward_energy_weight=0.25`
- `reward_backlog_weight=0.25`
- `reward_expired_weight=0.0`
- `reward_deadline_weight=0.50`
- `reward_reliability_weight=0.50`
- `reward_action_magnitude_weight=0.0`
- `reward_invalid_action_weight=0.05`
- `reward_infeasible_action_weight=0.10`
- `use_movement_budget=True`

## 当前环境侧卸载逻辑

- `local`
- `associated UAV`
- `collaborator UAV`
- `BS`

- joint actor 当前会输出 `mobility / offloading / caching` 三部分动作。
- offloading 仍采用 candidate-based 执行：策略只在固定 task slot 暴露出的候选 plan 集合中选择 `candidate_plan_id` 或 `defer`。
- 在 joint 主路径中，环境会直接执行策略选中的 plan，并在无效或不可行动作上走 `reject/defer`；只有 legacy baseline 路径仍保留旧的自动关联/卸载/缓存规则。
- 当目标 UAV 未缓存所需服务时，环境当前只在 `BS` 与其他已缓存该服务的 UAV 之间比较 fetch 时延，并选择更快的 fetch 来源；这不是联合能耗/可靠性优化。

## 共享链路近似

- 所有 `UE -> UAV` 与 `UE -> BS` 接入传输共享 `edge_access` 串行资源。
- 所有 `UAV -> UAV` 中继、`BS -> UAV` 服务回源与 `UAV -> UAV` peer fetch 共享 `backhaul` 串行资源。
- 当前实现是 shared resource-class serial queue approximation，不建模更细粒度的空间复用或干扰图。

## 指标口径说明

- `cache_hit_rate` 只统计 UAV 执行链路上的服务缓存命中；`local` 与 `BS` 分支不计入缓存命中。
- episode 到达终局时，剩余未完成任务会被统一转为 `expired` 并纳入最终 episode 指标，避免 summary 漏掉终局 pending 任务。
- `step_signals` 与 episode 级展示指标已彻底分离；训练奖励消费按 finalized / pending / generated 事件构造的 step proxy 信号，而不是直接复用 episode 指标名。
- 训练奖励当前显式包含 `completion / cache_hit / backlog / expired / latency / energy / deadline / reliability / action_magnitude` 九项；README、论文和结果说明都应按这一口径书写。
- `relay_fetch_energy` 只统计 UAV 发起的协同中继与 peer fetch 发射能耗；`BS -> UAV` 的服务回源传输能耗单列为 `bs_fetch_tx_energy`，并计入系统总能耗。
- 协同中继与 peer fetch 的发射能耗会回写到对应发送 UAV 的剩余能量状态中，因此多 UAV 观测里的能量状态与实际能耗链路保持一致。
- 当前训练奖励已经显式纳入能耗项，且能耗归一化按 step-level reference scale 计算；更稳妥的写法应是“能耗已进入训练目标，但其相对贡献仍需以复跑结果为准”。

## 结果目录说明

- `results/stage5/paper_stage5_v2/`: 当前调参与主配置选择目录
- `results/stage6/paper_stage6_v2/`: `run_finalize_paper.py` 与 `run_refresh_verification.py` 的输出目录
  - `tables/`: 终稿表格 markdown/csv
  - `figures/`: 终稿训练曲线与对比柱状图
  - `*_summary.json`: 均值/标准差汇总
  - `reproducibility_package.json`: 一键复现实验包说明
- `results/legacy/`: legacy mobility-only baseline 与 legacy RL 的 checkpoint / 训练日志 / 评估日志
- `results/joint/`: joint heuristic 与 joint RL 的 checkpoint / 训练日志 / 评估日志

## 论文材料

- 正文草稿: `第四章/docs/chapter4_writing_draft.md`
- 验证记录: `第四章/VERIFICATION.md`
