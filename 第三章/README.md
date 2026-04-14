# 第三章

本目录是统一任务模型下的单 UAV 增强版实现。当前第三章继续复用 `common/uav_mec` 共享核心，并同时保留两条单 UAV 控制路径：

- `heuristic`：最小可运行对照策略
- `mpc`：基于统一 observation 的 receding-horizon MPC shell
- `fixed_point`：固定点基线
- `fixed_patrol`：固定巡航基线

## 目录说明

- `chapter3/env.py`: 单 UAV 环境包装
- `chapter3/policies/mobility_heuristic.py`: 启发式移动策略
- `chapter3/policies/mpc_shell.py`: MPC shell 策略入口
- `chapter3/solvers/ch3_mpc_optimizer.py`: 第三章 MPC shell 候选动作优化器
- `chapter3/experiments/trajectory.py`: 轨迹日志与 PNG 导出
- `run_smoke.py`: 烟雾测试入口
- `run_experiment.py`: 第三章实验与 `compare-ch4` 入口
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
- 第三章已补入 MPC shell，不再只有 heuristic 单一路径
- 第三章普通实验会导出 UAV/UE 位置轨迹，便于论文中补充单 UAV 路径图
