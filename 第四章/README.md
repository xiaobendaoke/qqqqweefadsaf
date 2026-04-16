# 第四章

本目录是基于统一任务模型与统一指标体系构建的多 UAV 扩展版实现。当前默认主方法为 `energy_e30` 配置下的 `shared_ppo_centralized_critic`。在统一环境内核上，第四章已经补齐 collaborator-UAV 分支与 peer UAV service fetch，因此卸载逻辑不再只停留在 `local / associated UAV / BS` 三分支。

## 目录说明

- `chapter4/`: 多 UAV 环境、启发式策略、PPO 训练与评估实现
- `run_experiment.py`: heuristic baseline 实验入口
- `run_train_marl.py`: 单次 PPO 训练入口
- `run_eval_marl.py`: 单次 PPO 评估入口
- `run_paper_experiments.py`: 第五阶段论文实验矩阵与消融入口
- `run_finalize_paper.py`: 第六阶段终稿复跑、均值/标准差统计与结果包入口
- `results/`: 实验结果、模型 checkpoint、表格与图表
- `VERIFICATION.md`: 已执行命令、关键日志与结果路径
- `requirements.txt`: 本章依赖说明

## 环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r 第四章/requirements.txt
```

## 一键复现实验

先验证 `NUM_UAVS=1` 退化一致性：

```powershell
.\.venv\Scripts\python.exe 第三章/run_experiment.py --episodes 1 --compare-ch4 --seed 42
```

再复现最终论文结果包：

```powershell
.\.venv\Scripts\python.exe 第四章/run_finalize_paper.py --seeds 42 52 62 --eval-episodes 4
```

说明:

- stage6 会使用训练种子 `42/52/62`，并在评估阶段固定采用对应 held-out 种子 `142/152/162`。
- `observation / uav_state / episode_log` schema 在主实验与 sensitive profile 下保持一致；缺失邻居槽位使用零填充。

## 固定主配置

- `train_episodes=30`
- `actor_lr=0.00025`
- `critic_lr=0.0008`
- `clip_ratio=0.18`
- `entropy_coef=0.008`
- `value_coef=0.6`
- `reward_energy_weight=1.5`
- `reward_action_magnitude_weight=0.2`
- `use_movement_budget=True`

## 当前统一卸载逻辑

- `local`
- `associated UAV`
- `collaborator UAV`
- `BS`

当目标 UAV 未缓存所需服务时，环境会在 `BS` 与其他已缓存该服务的 UAV 之间比较 fetch 时延，并选择更优的 service fetch 来源。

## 指标口径说明

- `cache_hit_rate` 只统计 UAV 执行链路上的服务缓存命中；`local` 与 `BS` 分支不计入缓存命中。
- episode 到达终局时，剩余未完成任务会被统一转为 `expired` 并纳入最终 episode 指标，避免 summary 漏掉终局 pending 任务。
- `relay_fetch_energy` 只统计 UAV 发起的协同中继与 peer fetch 发射能耗；`BS -> UAV` 的 fetch 仅计入时延，不并入该能耗项。
- 协同中继与 peer fetch 的发射能耗会回写到对应发送 UAV 的剩余能量状态中，因此多 UAV 观测里的能量状态与实际能耗链路保持一致。

## 结果目录说明

- `results/paper_stage5/`: 第五阶段单组主实验、调参与最小消融结果
- `results/paper_stage6/`: 第六阶段 3-seed 复跑后的终稿结果包
  - `tables/`: 终稿表格 markdown/csv
  - `figures/`: 终稿训练曲线与对比柱状图
  - `*_summary.json`: 均值/标准差汇总
  - `reproducibility_package.json`: 一键复现实验包说明
- `results/marl_shared_ppo_*.pt`: 各次训练生成的 checkpoint
- `results/marl_train_shared_ppo_*.json`: 训练日志
- `results/marl_eval_shared_ppo_*.json`: 评估与 heuristic 对比结果

## 论文材料

- 正文草稿: `第四章/docs/chapter4_writing_draft.md`
- 验证记录: `第四章/VERIFICATION.md`
