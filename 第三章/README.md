# 第三章

本目录是统一任务模型下的单 UAV 增强版实现。当前第三章继续复用 `common/uav_mec` 共享核心，并同时保留两条单 UAV 控制路径：

- `heuristic`：最小可运行对照策略
- `mpc`：基于统一 observation 的 look-ahead heuristic shell（命名沿用 `MPC`，但不求解标准约束 MPC）
- `fixed_point`：固定点基线
- `fixed_patrol`：固定巡航基线

## 目录说明

- `chapter3/env.py`: 单 UAV 环境包装
- `chapter3/policies/mobility_heuristic.py`: 启发式移动策略
- `chapter3/policies/mpc_shell.py`: look-ahead heuristic shell 策略入口
- `chapter3/solvers/ch3_mpc_optimizer.py`: 第三章候选动作滚动评分优化器
- `chapter3/experiments/trajectory.py`: 轨迹日志与 PNG 导出
- `run_smoke.py`: 烟雾测试入口
- `run_experiment.py`: 第三章实验与 `compare-ch4` 入口
- `run_refresh_verification.py`: 刷新当前验证 JSON 的入口
- `results/`: 结果文件
- `VERIFICATION.md`: 运行命令与关键日志

## 常用命令

```powershell
.\.venv\Scripts\python.exe 第三章/run_smoke.py --mode episode --seed 42
.\.venv\Scripts\python.exe 第三章/run_experiment.py --episodes 1 --policy heuristic --seed 42
.\.venv\Scripts\python.exe 第三章/run_experiment.py --episodes 1 --policy mpc --seed 42
.\\.venv\\Scripts\\python.exe 第三章/run_experiment.py --episodes 1 --policy fixed_point --steps-per-episode 20 --seed 42
.\\.venv\\Scripts\\python.exe 第三章/run_experiment.py --episodes 1 --policy fixed_patrol --steps-per-episode 20 --seed 42
.\.venv\Scripts\python.exe 第三章/run_experiment.py --episodes 1 --compare-ch4 --seed 42
.\.venv\Scripts\python.exe 第三章/run_refresh_verification.py
```

普通实验默认会额外输出单 UAV 轨迹结果到 `第三章/results/trajectories/`：

- `trajectory_<profile>_<policy>_seed<seed>_ep<idx>.json`
- `trajectory_<profile>_<policy>_seed<seed>_ep<idx>.png`

需要更长的可视化轨迹时，可额外指定：

```powershell
.\.venv\Scripts\python.exe 第三章/run_experiment.py --episodes 1 --policy mpc --steps-per-episode 20 --seed 42
```

## 当前状态

- 单 UAV 统一模型与第四章共享同一任务/状态/指标主干
- `compare-ch4` 持续通过，`第四章(NUM_UAVS=1)` 仍可退化为第三章
- 第三章已补入基于候选动作滚动评分的 look-ahead heuristic shell，不再只有 heuristic 单一路径
- 论文里若引用 `MPC shell`，应将其表述为候选动作滚动打分的 heuristic shell，而不是标准优化求解式 MPC
- 第三章普通实验会导出 UAV/UE 位置轨迹，便于论文中补充单 UAV 路径图
- 当前验证真值应优先以 `第三章/results/verification_refresh.json` 与 `第三章/VERIFICATION.md` 为准

## 指标口径说明

- `cache_hit_rate` 现在只统计 UAV 执行分支上的服务缓存命中；`local` 与 `BS` 分支不再记为缓存命中。
- episode 结束时若仍有未完成任务，这些任务会被统一转为 `expired` 并计入最终指标，避免 `completion_rate / average_latency / violation_rate` 漏统终局 pending 任务。
- `step_signals` 与 episode 总结指标已分离；轨迹与过程图只读取显式逐步信号，不再复用 episode 指标名。
- `BS -> UAV` 的服务回源传输能耗单列为 `bs_fetch_tx_energy`，并纳入总能耗与能耗分解图。
- 若本机缺少 `matplotlib`，轨迹导出仍会稳定写出 `json`，但 `png` 会被跳过。
