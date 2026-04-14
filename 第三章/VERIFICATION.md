# 第三章验证记录

## 第一阶段

### 已运行命令

```powershell
python 第三章/run_smoke.py --mode import_only
python 第三章/run_smoke.py --mode env_step
python 第三章/run_smoke.py --mode episode
python 第三章/run_smoke.py --mode task_contract
python 第三章/run_smoke.py --mode comms_contract
python 第三章/run_smoke.py --mode scheduler_contract
```

### 关键结果

- `import_only`: 通过
- `env_step`: 通过
  - `completion_rate=1.0`
  - `average_latency=0.29126824754856045`
  - `cache_hit_rate=1.0`
- `episode`: 通过
  - `completion_rate=1.0`
  - `average_latency=0.2924221163699376`
  - `total_energy=55.9523399752047`
- `task_contract`: 通过
  - `observation_dim=17`
- `comms_contract`: 通过
  - `success_probability=0.990863216100025`
- `scheduler_contract`: 通过
  - `decision_target=bs`
  - `decision_total_latency=0.3032819491836381`

### 结果文件

- `第三章/results/smoke_import_only.json`
- `第三章/results/smoke_env_step.json`
- `第三章/results/smoke_episode.json`
- `第三章/results/smoke_task_contract.json`
- `第三章/results/smoke_comms_contract.json`
- `第三章/results/smoke_scheduler_contract.json`

## 第一阶段 1.5 修正

### 新增运行命令

```powershell
python 第三章/run_experiment.py --episodes 1 --profile default --seed 42
python 第三章/run_experiment.py --episodes 1 --profile hard --seed 42
python 第三章/run_experiment.py --episodes 1 --compare-ch4 --seed 42
```

### 关键结果

- `default`:
  - `completion_rate=1.0`
  - `cache_hit_rate=0.9166666666666666`
- `hard`:
  - `completion_rate=0.0`
  - `cache_hit_rate=0.7291666666666666`
  - `deadline_violation_rate=0.9895833333333334`
  - `reliability_violation_rate=0.7395833333333334`
- `compare-ch4`:
  - 第三章与第四章 `NUM_UAVS=1` 指标逐项一致

### 新增结果文件

- `第三章/results/experiment_short.json`
- `第三章/results/experiment_hard.json`
- `第三章/results/experiment_compare_ch3_ch4.json`

## 第二阶段

### 复核命令

```powershell
python 第三章/run_experiment.py --episodes 1 --compare-ch4 --seed 42
```

### 复核结果

- `第四章(NUM_UAVS=1)` 与 `第三章` 指标仍一致
- 各项对照 `delta=0.0`

## 第二阶段 2.5

### 复核命令

```powershell
python 第三章/run_experiment.py --episodes 1 --compare-ch4 --seed 42
```

### 复核结果

- `compare-ch4` 跨章节导入在当前工作目录下稳定运行
- `第四章(NUM_UAVS=1)` 与 `第三章` 仍保持一致

## 第三阶段前置整理

### 复核命令

```powershell
python 第三章/run_experiment.py --episodes 1 --compare-ch4 --seed 42
```

### 复核结果

- `compare-ch4` 继续直接成功运行
- 第三章与第四章 `NUM_UAVS=1` 的主指标保持逐项一致
- `comparison` 中除 `fairness_uav_load=null` 外，其余指标 `delta=0.0`

### 结果文件

- `第三章/results/experiment_compare_ch3_ch4.json`

## 第六阶段补完：MPC shell

### 新增运行命令

```powershell
.\.venv\Scripts\python.exe 第三章/run_experiment.py --episodes 1 --policy mpc --seed 42
.\.venv\Scripts\python.exe 第三章/run_experiment.py --episodes 1 --compare-ch4 --seed 42
```

### 关键结果

- `mpc`:
  - `completion_rate=1.0`
  - `average_latency=0.28992447022728784`
  - `total_energy=216.22837538036669`
  - `cache_hit_rate=0.9166666666666666`
- `compare-ch4`:
  - 第三章与第四章 `NUM_UAVS=1` 逐项对照继续成立
  - `completion_rate / average_latency / total_energy delta=0.0`

### 新增结果文件

- `第三章/results/experiment_short_mpc.json`

## 第六阶段补完：单 UAV 轨迹日志与路径图

### 新增运行命令

```powershell
.\.venv\Scripts\python.exe 第三章/run_experiment.py --episodes 1 --policy heuristic --seed 42
.\.venv\Scripts\python.exe 第三章/run_experiment.py --episodes 1 --policy mpc --seed 42
.\.venv\Scripts\python.exe 第三章/run_experiment.py --episodes 1 --compare-ch4 --seed 42
```

### 关键结果

- `heuristic`:
  - `completion_rate=1.0`
  - `average_latency=0.2898769422422626`
  - `total_energy=56.18071535557137`
  - 已生成单 UAV 轨迹 `json/png`
- `mpc`:
  - `completion_rate=1.0`
  - `average_latency=0.28992447022728784`
  - `total_energy=216.22837538036669`
  - 已生成单 UAV 轨迹 `json/png`
- `compare-ch4`:
  - 第三章与第四章 `NUM_UAVS=1` 逐项对照继续成立
  - `completion_rate / average_latency / total_energy delta=0.0`

### 新增结果文件

- `第三章/results/trajectories/trajectory_default_heuristic_seed42_ep0.json`
- `第三章/results/trajectories/trajectory_default_heuristic_seed42_ep0.png`
- `第三章/results/trajectories/trajectory_default_mpc_seed42_ep0.json`
- `第三章/results/trajectories/trajectory_default_mpc_seed42_ep0.png`

## 第六阶段整理：长轨迹与论文基线

### 新增运行命令

```powershell
.\.venv\Scripts\python.exe 第三章/run_experiment.py --episodes 1 --policy heuristic --steps-per-episode 20 --seed 42
.\.venv\Scripts\python.exe 第三章/run_experiment.py --episodes 1 --policy mpc --steps-per-episode 20 --seed 42
.\.venv\Scripts\python.exe 第三章/run_experiment.py --episodes 1 --policy fixed_point --steps-per-episode 20 --seed 42
.\.venv\Scripts\python.exe 第三章/run_experiment.py --episodes 1 --policy fixed_patrol --steps-per-episode 20 --seed 42
.\.venv\Scripts\python.exe 第三章/run_experiment.py --episodes 1 --compare-ch4 --seed 42
```

### 关键结果

- `heuristic (20 steps)`:
  - `completion_rate=1.0`
  - `average_latency=0.276176731772956`
  - `total_energy=105.73552009918245`
- `mpc (20 steps)`:
  - `completion_rate=1.0`
  - `average_latency=0.27634236549410374`
  - `total_energy=459.9450803374705`
  - 路径由直角折线改为连续候选 + 惯性平滑后的弯曲轨迹
- `fixed_point (20 steps)`:
  - `completion_rate=1.0`
  - `average_latency=0.2761164399848625`
  - `total_energy=0.5843118416494011`
- `fixed_patrol (20 steps)`:
  - `completion_rate=1.0`
  - `average_latency=0.27609754970239925`
  - `total_energy=540.4162595424357`
- `compare-ch4`:
  - 第三章与第四章 `NUM_UAVS=1` 逐项对照继续成立
  - `completion_rate / average_latency / total_energy delta=0.0`

### 新增结果文件

- `第三章/results/experiment_default_heuristic_s20.json`
- `第三章/results/experiment_default_mpc_s20.json`
- `第三章/results/experiment_default_fixed_point_s20.json`
- `第三章/results/experiment_default_fixed_patrol_s20.json`
- `第三章/results/trajectories/trajectory_default_s20_heuristic_seed42_ep0.png`
- `第三章/results/trajectories/trajectory_default_s20_mpc_seed42_ep0.png`
- `第三章/results/trajectories/trajectory_default_s20_fixed_point_seed42_ep0.png`
- `第三章/results/trajectories/trajectory_default_s20_fixed_patrol_seed42_ep0.png`
