# 第四章验证记录

## 第一阶段

### 已运行命令

```powershell
python 第四章/run_smoke.py --mode env_step --num-uavs 1
python 第四章/run_smoke.py --mode episode --num-uavs 1
python 第四章/run_smoke.py --mode env_step --num-uavs 2
```

### 关键结果

- `env_step --num-uavs 1`: 通过
  - `completion_rate=1.0`
  - `average_latency=0.29126824754856045`
  - `fairness_uav_load=1.0`
- `episode --num-uavs 1`: 通过
  - `completion_rate=1.0`
  - `average_latency=0.2924221163699376`
  - `fairness_uav_load=1.0`
- `env_step --num-uavs 2`: 按预期返回未实现提示
  - `message=Chapter 4 phase 1 only supports NUM_UAVS=1.`

### 结果文件

- `第四章/results/smoke_env_step.json`
- `第四章/results/smoke_episode.json`

## 第一阶段 1.5 修正

### 新增运行命令

```powershell
python 第四章/run_smoke.py --mode env_step --num-uavs 1
python 第四章/run_experiment.py --episodes 1 --profile default --seed 42 --num-uavs 1
python 第四章/run_experiment.py --episodes 1 --profile hard --seed 42 --num-uavs 1
```

### 关键结果

- `env_step --num-uavs 1`:
  - `fairness_uav_load=null`
- `default experiment`:
  - `completion_rate=1.0`
  - `cache_hit_rate=0.9166666666666666`
- `hard experiment`:
  - `completion_rate=0.0`
  - `cache_hit_rate=0.7291666666666666`
  - `deadline_violation_rate=0.9895833333333334`
  - `reliability_violation_rate=0.7395833333333334`

### 新增结果文件

- `第四章/results/experiment_short.json`
- `第四章/results/experiment_hard.json`

## 第二阶段

### 新增运行命令

```powershell
python 第四章/run_experiment.py --episodes 1 --profile default --seed 42 --num-uavs 1 --assignment-rule nearest_uav
python 第四章/run_experiment.py --episodes 1 --profile default --seed 42 --num-uavs 2 --assignment-rule nearest_uav
python 第四章/run_experiment.py --episodes 1 --profile default --seed 42 --num-uavs 3 --assignment-rule nearest_uav
python 第四章/run_experiment.py --episodes 1 --profile default --seed 42 --num-uavs 3 --assignment-rule least_loaded_uav
```

### 关键结果

- `NUM_UAVS=1`
  - `completion_rate=1.0`
  - `fairness_uav_load=null`
- `NUM_UAVS=2`
  - `completion_rate=1.0`
  - `fairness_uav_load=1.0`
- `NUM_UAVS=3`
  - `completion_rate=1.0`
  - `fairness_uav_load=0.3333333333333333`
- `least_loaded_uav`
  - 命令可运行
  - 当前 seed 下结果与 `nearest_uav` 相同

### 新增结果文件

- `第四章/results/experiment_short_u1_nearest_uav.json`
- `第四章/results/experiment_short_u2_nearest_uav.json`
- `第四章/results/experiment_short_u3_nearest_uav.json`
- `第四章/results/experiment_short_u3_least_loaded_uav.json`

## 第二阶段 2.5

### 新增运行命令

```powershell
python 第四章/run_smoke.py --mode observation --num-uavs 1
python 第四章/run_smoke.py --mode observation --num-uavs 2
python 第四章/run_smoke.py --mode observation --num-uavs 3
python 第四章/run_experiment.py --episodes 1 --profile sensitive --seed 42 --num-uavs 2 --assignment-rule nearest_uav
python 第四章/run_experiment.py --episodes 1 --profile sensitive --seed 42 --num-uavs 2 --assignment-rule least_loaded_uav
python 第四章/run_experiment.py --episodes 1 --profile sensitive --seed 42 --num-uavs 3 --assignment-rule nearest_uav
python 第四章/run_experiment.py --episodes 1 --profile sensitive --seed 42 --num-uavs 3 --assignment-rule least_loaded_uav
```

### 关键结果

- observation smoke:
  - `NUM_UAVS=1/2/3` 均可返回 observation schema、per-UAV state 与 observation sample
  - 当前 observation length 固定为 `46`
- `sensitive, NUM_UAVS=2`
  - `nearest_uav`: `fairness_uav_load=0.9411764705882353`
  - `least_loaded_uav`: `fairness_uav_load=0.9993429697766097`
- `sensitive, NUM_UAVS=3`
  - `nearest_uav`: `fairness_uav_load=0.6274509803921569`
  - `least_loaded_uav`: `fairness_uav_load=0.9988116458704694`

### 新增结果文件

- `第四章/results/smoke_observation_u1.json`
- `第四章/results/smoke_observation_u2.json`
- `第四章/results/smoke_observation_u3.json`
- `第四章/results/experiment_sensitive_u2_nearest_uav.json`
- `第四章/results/experiment_sensitive_u2_least_loaded_uav.json`
- `第四章/results/experiment_sensitive_u3_nearest_uav.json`
- `第四章/results/experiment_sensitive_u3_least_loaded_uav.json`

## 第三阶段前置整理

### 新增运行命令

```powershell
python 第四章/run_multi_agent_episode.py --seed 42 --num-uavs 1 --assignment-rule nearest_uav
python 第四章/run_multi_agent_episode.py --seed 42 --num-uavs 2 --assignment-rule nearest_uav
python 第四章/run_multi_agent_episode.py --seed 42 --num-uavs 3 --assignment-rule nearest_uav
```

### 关键结果

- `NUM_UAVS=1/2/3` 均成功返回标准化多 agent 接口
- `action_schema` 固定为 canonical shape `[num_uavs, 2]`
- `observation_schema.total_length=46`
- `uav_state_schema.total_length=31`
- `episode_log` 同时包含 `global_metrics`、`per_uav_metrics`、`assignment_rule`、`num_uavs`、`seed`

### 新增结果文件

- `第四章/results/multi_agent_episode_u1_nearest_uav.json`
- `第四章/results/multi_agent_episode_u2_nearest_uav.json`
- `第四章/results/multi_agent_episode_u3_nearest_uav.json`

## 第四阶段 最小 MARL 接入

### 新增运行命令

```powershell
python 第四章/run_train_marl.py --seed 42 --train-episodes 12 --num-uavs 2 --assignment-rule nearest_uav
python 第四章/run_eval_marl.py --seed 142 --eval-episodes 4 --num-uavs 2 --assignment-rule nearest_uav --model-path 第四章/results/marl_shared_ac_u2_nearest_uav.json
python 第三章/run_experiment.py --episodes 1 --compare-ch4 --seed 42
```

### 关键结果

- 最小 MARL 训练闭环已跑通
  - 算法：`shared_centralized_actor_critic`
  - `observation_batch_shape=["T", 2, 46]`
  - `central_state_shape=["T", 92]`
  - `action_batch_shape=["T", 2, 2]`
- 训练日志样例
  - `episode=0`
  - `team_return=6.898523812642534`
  - `completion_rate=1.0`
  - `actor_loss=-0.6582713615160721`
  - `critic_loss=8.157060019969908`
- 评估结果
  - `MARL completion_rate=1.0`
  - `MARL average_latency=0.25998963405751235`
  - `MARL fairness_uav_load=0.7`
  - `heuristic completion_rate=1.0`
  - `heuristic average_latency=0.25996999112363817`
  - `heuristic fairness_uav_load=0.6`
- `compare-ch4` 复核通过
  - 第三章与第四章 `NUM_UAVS=1` 指标仍逐项一致

### 新增结果文件

- `第四章/results/marl_shared_ac_u2_nearest_uav.json`
- `第四章/results/marl_train_u2_nearest_uav.json`
- `第四章/results/marl_eval_u2_nearest_uav.json`
