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

## 第四阶段 4.5 纯 Python MARL 能耗约束改良

### 新增运行命令

```powershell
python 第三章/run_experiment.py --episodes 1 --compare-ch4 --seed 42
python 第四章/run_train_marl.py --seed 42 --train-episodes 12 --num-uavs 2 --assignment-rule nearest_uav
python 第四章/run_eval_marl.py --seed 142 --eval-episodes 4 --num-uavs 2 --assignment-rule nearest_uav --model-path 第四章/results/marl_shared_ac_u2_nearest_uav.json
```

### 关键结果

- `compare-ch4`:
  - 第三章与第四章 `NUM_UAVS=1` 主指标仍逐项一致
  - `total_energy delta=0.0`
- MARL 训练侧改动:
  - 动作后处理增加“最近用户距离 + slack 紧迫度”驱动的 movement budget
  - training signal 升级为 `shaped_team_reward_v2`
  - 显式包含 `completion_rate / cache_hit_rate / average_latency / delta_total_energy / deadline_violation_rate / reliability_violation_rate / action_magnitude`
- MARL 训练日志样例:
  - `episode=0`
  - `team_return=5.750323508351074`
  - `mean_step_energy=11.968951686670389`
  - `mean_step_action_magnitude=0.1658389550572857`
- MARL vs heuristic 评估:
  - `MARL completion_rate=1.0`
  - `MARL average_latency=0.2599826757271505`
  - `MARL total_energy=159.3156091199015`
  - `heuristic total_energy=27.865925714269967`
  - 相比上一版 MARL 评估结果 `181.66260509156172`，当前 `total_energy` 下降约 `12.30%`
  - `fairness_uav_load=0.7`

### 新增结果文件

- `第四章/results/marl_shared_ac_u2_nearest_uav.json`
- `第四章/results/marl_train_u2_nearest_uav.json`
- `第四章/results/marl_eval_u2_nearest_uav.json`

## 第四阶段 2 真正 PPO 接入

### 新增环境命令

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy torch --index-url https://download.pytorch.org/whl/cpu
```

### 新增运行命令

```powershell
.\.venv\Scripts\python.exe 第三章/run_experiment.py --episodes 1 --compare-ch4 --seed 42
.\.venv\Scripts\python.exe 第四章/run_train_marl.py --seed 42 --train-episodes 12 --num-uavs 2 --assignment-rule nearest_uav
.\.venv\Scripts\python.exe 第四章/run_eval_marl.py --seed 142 --eval-episodes 4 --num-uavs 2 --assignment-rule nearest_uav --model-path 第四章/results/marl_shared_ppo_u2_nearest_uav.pt
.\.venv\Scripts\python.exe 第四章/run_train_marl.py --seed 42 --train-episodes 1 --num-uavs 1 --assignment-rule nearest_uav
.\.venv\Scripts\python.exe 第四章/run_train_marl.py --seed 42 --train-episodes 1 --num-uavs 3 --assignment-rule nearest_uav
```

### 关键结果

- `compare-ch4`:
  - 第三章与第四章 `NUM_UAVS=1` 主指标仍逐项一致
  - `total_energy delta=0.0`
- 新算法:
  - `shared_ppo_centralized_critic`
  - `torch_version=2.11.0+cpu`
  - `numpy_version=2.4.3`
  - 共享 actor 基于单 agent observation
  - centralized critic 基于拼接后的全局 state
  - 团队奖励继续使用 `shaped_team_reward_v2`
- PPO 最小训练日志样例:
  - `episode=0`
  - `team_return=4.194825172424316`
  - `completion_rate=1.0`
  - `average_latency=0.2892902322758323`
  - `total_energy=178.71281825686273`
  - `actor_loss=-0.055647116132604424`
  - `critic_loss=2.6283752620220184`
  - `entropy=0.06519803404808044`
- PPO 评估:
  - `completion_rate=1.0`
  - `average_latency=0.2599643812592326`
  - `total_energy=58.58949143888045`
  - `fairness_uav_load=0.6`
- shape 兼容性补充检查:
  - `NUM_UAVS=1` 训练成功，`central_state_shape=["T", 46]`
  - `NUM_UAVS=3` 训练成功，`central_state_shape=["T", 138]`
- 与上一版纯 Python shared actor-critic 对比:
  - 旧版 `total_energy=159.3156091199015`
  - 新版 `total_energy=58.58949143888045`
  - `total_energy` 下降约 `63.22%`
  - `average_latency` 从 `0.2599826757271505` 降到 `0.2599643812592326`
- 与 heuristic baseline 对比:
  - heuristic `completion_rate=1.0`
  - heuristic `average_latency=0.25996999112363817`
  - heuristic `total_energy=27.865925714269967`
  - PPO 相比 heuristic 仍高能耗，但差距从纯 Python 版的 `131.44968340563153` 收敛到 `30.723565724610484`

### 新增结果文件

- `第四章/results/marl_shared_ppo_u2_nearest_uav.pt`
- `第四章/results/marl_train_shared_ppo_u2_nearest_uav.json`
- `第四章/results/marl_eval_shared_ppo_u2_nearest_uav.json`
- `第四章/results/marl_shared_ppo_u1_nearest_uav.pt`
- `第四章/results/marl_train_shared_ppo_u1_nearest_uav.json`
- `第四章/results/marl_shared_ppo_u3_nearest_uav.pt`
- `第四章/results/marl_train_shared_ppo_u3_nearest_uav.json`

## 第五阶段 论文实验成型与消融

### 新增环境命令

```powershell
.\.venv\Scripts\python.exe -m pip install matplotlib --index-url https://pypi.org/simple
```

### 新增运行命令

```powershell
.\.venv\Scripts\python.exe 第三章/run_experiment.py --episodes 1 --compare-ch4 --seed 42
.\.venv\Scripts\python.exe 第四章/run_paper_experiments.py --seed 42 --eval-seed 142 --eval-episodes 4
```

### 最终 PPO 主配置

- 自动选中的候选: `energy_e30`
- `train_episodes=30`
- `actor_lr=0.00025`
- `critic_lr=0.0008`
- `clip_ratio=0.18`
- `entropy_coef=0.008`
- `value_coef=0.6`
- `reward_energy_weight=1.5`
- `reward_action_magnitude_weight=0.2`
- `use_movement_budget=True`

### 关键结果

- `compare-ch4`:
  - 第三章与第四章 `NUM_UAVS=1` 主指标继续一致
  - `completion_rate / average_latency / total_energy` 的 `delta` 仍为 `0.0`
- 主配置训练日志样例:
  - `episode=0`
  - `team_return=3.3262407779693604`
  - `completion_rate=1.0`
  - `average_latency=0.2892902322758323`
  - `total_energy=178.71281825686273`
  - `actor_loss=-0.04599898870947072`
  - `critic_loss=2.0121232867240906`
  - `entropy=0.06516480445861816`
  - `action_std_scale=0.2486697877943516`
- 主实验 `PPO vs heuristic`:
  - `u2 + nearest_uav`:
    - PPO `completion_rate=1.0`
    - PPO `average_latency=0.25996951130234985`
    - PPO `total_energy=50.27020852408417`
    - heuristic `total_energy=27.865925714269967`
    - `delta_total_energy=22.4042828098142`
  - `u3 + nearest_uav`:
    - PPO `completion_rate=1.0`
    - PPO `average_latency=0.26831519456870234`
    - PPO `total_energy=71.75571435026387`
    - heuristic `total_energy=42.57059405190918`
    - `delta_total_energy=29.18512029835469`
- 调参结果:
  - `base_e12 total_energy=58.58949143888045`
  - `energy_e30 total_energy=50.27020852408417`
  - `energy_e60 total_energy=91.86481817652535`
  - `stable_e30 total_energy=63.73852040513789`
  - 按 `completion_rate` 优先、再按 `total_energy` 最低选择，最终固定 `energy_e30`
- assignment heuristic 对比使用 `sensitive` profile:
  - `u2`:
    - `nearest_uav total_energy=32.94511440276091`
    - `least_loaded_uav total_energy=32.793658122393566`
    - `least_loaded_uav fairness_uav_load=0.9995228523080433`
  - `u3`:
    - `nearest_uav total_energy=76.26380151222608`
    - `least_loaded_uav total_energy=76.45117456803078`
    - `least_loaded_uav fairness_uav_load=0.999169751169418`
- 最小消融:
  - `no_energy_shaped_reward`:
    - `total_energy=62.063485676623266`
    - 相比主配置多 `11.793277152539093`
  - `no_movement_budget`:
    - `total_energy=74.0749144830099`
    - 相比主配置多 `23.804705958925735`

### 新增结果文件

- `第四章/results/marl_shared_ppo_u2_nearest_uav_tune_base_e12.pt`
- `第四章/results/marl_train_shared_ppo_u2_nearest_uav_tune_base_e12.json`
- `第四章/results/marl_eval_shared_ppo_u2_nearest_uav_tune_base_e12.json`
- `第四章/results/marl_shared_ppo_u2_nearest_uav_tune_energy_e30.pt`
- `第四章/results/marl_train_shared_ppo_u2_nearest_uav_tune_energy_e30.json`
- `第四章/results/marl_eval_shared_ppo_u2_nearest_uav_tune_energy_e30.json`
- `第四章/results/marl_shared_ppo_u2_nearest_uav_tune_energy_e60.pt`
- `第四章/results/marl_train_shared_ppo_u2_nearest_uav_tune_energy_e60.json`
- `第四章/results/marl_eval_shared_ppo_u2_nearest_uav_tune_energy_e60.json`
- `第四章/results/marl_shared_ppo_u2_nearest_uav_tune_stable_e30.pt`
- `第四章/results/marl_train_shared_ppo_u2_nearest_uav_tune_stable_e30.json`
- `第四章/results/marl_eval_shared_ppo_u2_nearest_uav_tune_stable_e30.json`
- `第四章/results/marl_shared_ppo_u2_nearest_uav_paper_main_u2.pt`
- `第四章/results/marl_train_shared_ppo_u2_nearest_uav_paper_main_u2.json`
- `第四章/results/marl_eval_shared_ppo_u2_nearest_uav_paper_main_u2.json`
- `第四章/results/marl_shared_ppo_u3_nearest_uav_paper_main_u3.pt`
- `第四章/results/marl_train_shared_ppo_u3_nearest_uav_paper_main_u3.json`
- `第四章/results/marl_eval_shared_ppo_u3_nearest_uav_paper_main_u3.json`
- `第四章/results/marl_shared_ppo_u2_nearest_uav_ablation_no_energy.pt`
- `第四章/results/marl_train_shared_ppo_u2_nearest_uav_ablation_no_energy.json`
- `第四章/results/marl_eval_shared_ppo_u2_nearest_uav_ablation_no_energy.json`
- `第四章/results/marl_shared_ppo_u2_nearest_uav_ablation_no_budget.pt`
- `第四章/results/marl_train_shared_ppo_u2_nearest_uav_ablation_no_budget.json`
- `第四章/results/marl_eval_shared_ppo_u2_nearest_uav_ablation_no_budget.json`
- `第四章/results/paper_stage5/tuning_summary.csv`
- `第四章/results/paper_stage5/tuning_summary.json`
- `第四章/results/paper_stage5/main_experiment_matrix.csv`
- `第四章/results/paper_stage5/main_experiment_matrix.json`
- `第四章/results/paper_stage5/ablation_summary.csv`
- `第四章/results/paper_stage5/ablation_summary.json`
- `第四章/results/paper_stage5/assignment_rule_matrix.csv`
- `第四章/results/paper_stage5/assignment_rule_matrix.json`
- `第四章/results/paper_stage5/config_notes.json`
- `第四章/results/paper_stage5/paper_summary.md`
- `第四章/results/paper_stage5/chapter3_vs_chapter4_num_uavs1.json`
- `第四章/results/paper_stage5/paper_experiments_summary.json`
- `第四章/results/paper_stage5/ppo_training_curves.png`
- `第四章/results/paper_stage5/assignment_rule_comparison.png`
- `第四章/results/paper_stage5/ppo_vs_heuristic.png`

## 第六阶段 论文写作定稿与结果固化

### 固定主配置

- 最终主方法固定为 `energy_e30`
- `train_episodes=30`
- `actor_lr=0.00025`
- `critic_lr=0.0008`
- `clip_ratio=0.18`
- `entropy_coef=0.008`
- `value_coef=0.6`
- `reward_energy_weight=1.5`
- `reward_action_magnitude_weight=0.2`
- `use_movement_budget=True`
- 复跑种子: `42 / 52 / 62`
- `eval_episodes=4`

### 运行命令

```powershell
.\.venv\Scripts\python.exe 第三章/run_experiment.py --episodes 1 --compare-ch4 --seed 42
.\.venv\Scripts\python.exe 第四章/run_finalize_paper.py --seeds 42 52 62 --eval-episodes 4
```

### 最终复跑结果

- `compare-ch4` 多 seed 汇总:
  - `completion_rate delta_mean=0.0, delta_std=0.0`
  - `average_latency delta_mean=0.0, delta_std=0.0`
  - `total_energy delta_mean=0.0, delta_std=0.0`
- `PPO vs heuristic`:
  - `u2 + nearest_uav`
    - PPO `completion_rate=1.0000 +/- 0.0000`
    - PPO `average_latency=0.2702 +/- 0.0090`
    - PPO `total_energy=72.1017 +/- 21.2612`
    - heuristic `total_energy=81.4560 +/- 47.4455`
    - `delta_total_energy=-9.3543 +/- 33.7623`
  - `u3 + nearest_uav`
    - PPO `completion_rate=1.0000 +/- 0.0000`
    - PPO `average_latency=0.2713 +/- 0.0031`
    - PPO `total_energy=117.2762 +/- 51.0676`
    - heuristic `total_energy=123.0361 +/- 71.0983`
    - `delta_total_energy=-5.7599 +/- 55.5372`
- assignment 对比:
  - `u2 + nearest_uav`
    - `total_energy=33.4950 +/- 0.5457`
    - `fairness_uav_load=0.9758 +/- 0.0142`
  - `u2 + least_loaded_uav`
    - `total_energy=33.4323 +/- 0.6196`
    - `fairness_uav_load=0.9996 +/- 0.0001`
  - `u3 + nearest_uav`
    - `total_energy=76.8105 +/- 0.4738`
    - `fairness_uav_load=0.7427 +/- 0.0126`
  - `u3 + least_loaded_uav`
    - `total_energy=77.1246 +/- 0.5873`
    - `fairness_uav_load=0.9991 +/- 0.0002`
- 消融:
  - `main total_energy=72.1017 +/- 21.2612`
  - `no_energy_shaped_reward total_energy=77.7777 +/- 35.1335`
  - `no_movement_budget total_energy=82.3234 +/- 7.2455`

### 最终结果包

- `第四章/results/paper_stage6/compare_ch4_multiseed_raw.json`
- `第四章/results/paper_stage6/compare_ch4_multiseed_summary.json`
- `第四章/results/paper_stage6/assignment_multiseed_raw.json`
- `第四章/results/paper_stage6/assignment_multiseed_summary.json`
- `第四章/results/paper_stage6/ppo_vs_heuristic_multiseed_raw.json`
- `第四章/results/paper_stage6/ppo_vs_heuristic_multiseed_summary.json`
- `第四章/results/paper_stage6/ablation_multiseed_raw.json`
- `第四章/results/paper_stage6/ablation_multiseed_summary.json`
- `第四章/results/paper_stage6/reproducibility_package.json`
- `第四章/results/paper_stage6/tables/table_main_results.md`
- `第四章/results/paper_stage6/tables/table_main_results.csv`
- `第四章/results/paper_stage6/tables/table_assignment_comparison.md`
- `第四章/results/paper_stage6/tables/table_assignment_comparison.csv`
- `第四章/results/paper_stage6/tables/table_ppo_vs_heuristic.md`
- `第四章/results/paper_stage6/tables/table_ppo_vs_heuristic.csv`
- `第四章/results/paper_stage6/tables/table_ablation.md`
- `第四章/results/paper_stage6/tables/table_ablation.csv`
- `第四章/results/paper_stage6/figures/final_training_curve_mean_std.png`
- `第四章/results/paper_stage6/figures/final_comparison_bars.png`
- `第四章/docs/chapter4_writing_draft.md`
- `第四章/README.md`

## 第六阶段补完：collaborator-UAV 与 peer fetch

### 新增验证命令

```powershell
.\.venv\Scripts\python.exe 第四章/run_experiment.py --episodes 1 --profile sensitive --num-uavs 2 --assignment-rule nearest_uav
.\.venv\Scripts\python.exe 第四章/run_finalize_paper.py --seeds 42 52 62 --eval-episodes 4
```

### 定向验证结果

- collaborator branch:
  - 受控构造下 `decision.target='collaborator'`
  - `decision.assigned_uav_id=1`
  - `decision.reason='collaborator_execution'`
- peer fetch:
  - 受控构造下 `decision.fetch_source='uav:0'`
  - 说明执行 UAV 可从邻居 UAV 服务缓存拉取服务镜像
- 最小多 UAV 运行:
  - `python 第四章/run_experiment.py --episodes 1 --profile sensitive --num-uavs 2 --assignment-rule nearest_uav`
  - 运行成功
  - `completion_rate=1.0`
  - `average_latency=0.7992923222656232`
  - `total_energy=32.596614093346766`

### 更新后的最终复跑结果

- `compare-ch4` 多 seed 汇总仍保持:
  - `completion_rate delta_mean=0.0, delta_std=0.0`
  - `average_latency delta_mean=0.0, delta_std=0.0`
  - `total_energy delta_mean=0.0, delta_std=0.0`
- `PPO vs heuristic`:
  - `u2 + nearest_uav`
    - PPO `completion_rate=1.0000 +/- 0.0000`
    - PPO `average_latency=0.2701 +/- 0.0089`
    - PPO `total_energy=68.7092 +/- 11.7735`
    - heuristic `total_energy=81.4671 +/- 47.4514`
    - `delta_total_energy=-12.7579 +/- 40.7482`
  - `u3 + nearest_uav`
    - PPO `completion_rate=1.0000 +/- 0.0000`
    - PPO `average_latency=0.2710 +/- 0.0031`
    - PPO `total_energy=105.7111 +/- 36.3220`
    - heuristic `total_energy=123.0361 +/- 71.0983`
    - `delta_total_energy=-17.3250 +/- 68.8353`
- assignment 对比:
  - `u2 + nearest_uav`
    - `total_energy=33.4224 +/- 0.5985`
    - `fairness_uav_load=0.9978 +/- 0.0011`
  - `u2 + least_loaded_uav`
    - `total_energy=33.4224 +/- 0.5985`
    - `fairness_uav_load=0.9978 +/- 0.0011`
  - `u3 + nearest_uav`
    - `total_energy=77.3177 +/- 0.6238`
    - `fairness_uav_load=0.9809 +/- 0.0046`
  - `u3 + least_loaded_uav`
    - `total_energy=77.3177 +/- 0.6238`
    - `fairness_uav_load=0.9809 +/- 0.0046`
- 消融:
  - `main total_energy=68.7092 +/- 11.7735`
  - `no_energy_shaped_reward total_energy=84.3502 +/- 44.7163`
  - `no_movement_budget total_energy=73.4761 +/- 13.3339`

## 第六阶段整理：论文终稿图表重排

### 运行命令

```powershell
.\.venv\Scripts\python.exe 第四章/run_finalize_paper.py --seeds 42 52 62 --eval-episodes 4
```

### 结果说明

- `paper_stage6` 结果已基于当前代码重新复跑
- 终稿图已拆分为多张独立图，避免原先“训练曲线 + 混合柱状图”信息过载
- 当前可直接用于正文或附录的图包括：
  - `final_training_return_curve.png`
  - `final_training_energy_curve.png`
  - `final_ppo_vs_heuristic.png`
  - `final_assignment_comparison.png`
  - `final_ablation_energy.png`
  - `final_comparison_bars.png`

### 更新图表文件

- `第四章/results/paper_stage6/figures/final_training_return_curve.png`
- `第四章/results/paper_stage6/figures/final_training_energy_curve.png`
- `第四章/results/paper_stage6/figures/final_ppo_vs_heuristic.png`
- `第四章/results/paper_stage6/figures/final_assignment_comparison.png`
- `第四章/results/paper_stage6/figures/final_ablation_energy.png`
- `第四章/results/paper_stage6/figures/final_comparison_bars.png`

## 第七阶段统一系统可信度修复：共享环境 v2 与 PPO 正确性修复

### 运行命令

```powershell
.\.venv\Scripts\python.exe 第三章/run_experiment.py --episodes 1 --compare-ch4 --seed 42
.\.venv\Scripts\python.exe 第四章/run_train_marl.py --seed 42 --train-episodes 2 --num-uavs 2 --assignment-rule nearest_uav
.\.venv\Scripts\python.exe 第四章/run_eval_marl.py --seed 142 --eval-episodes 1 --num-uavs 2 --assignment-rule nearest_uav --model-path 第四章/results/marl_shared_ppo_u2_nearest_uav.pt
.\.venv\Scripts\python.exe 第四章/run_finalize_paper.py --seeds 42 52 62 --eval-episodes 4
```

### 关键结果

- `schema`:
  - `observation_schema = observation.v2`
  - `uav_state_schema = uav_state.v2`
  - `episode_log_schema = episode_log.v2`
- `compare-ch4`:
  - 在覆盖约束、backlog、compute queue、system energy 与 cache score 全部接入后继续通过
  - 新增的 `uav_move_energy / uav_compute_energy / ue_local_energy / ue_uplink_energy / bs_compute_energy / relay_fetch_energy` 多 seed `delta` 仍为 `0.0 +/- 0.0`
- `PPO 训练 smoke`:
  - `python 第四章/run_train_marl.py --seed 42 --train-episodes 2 --num-uavs 2 --assignment-rule nearest_uav`
  - 训练成功
  - `observation_batch_shape=["T", 2, 56]`
  - actor 已切换为 `squashed Gaussian + log_prob correction + minibatch PPO`
- `PPO 评估 smoke`:
  - `completion_rate=1.0`
  - `average_latency=0.3384`
  - `total_energy=15.1212`
- `assignment rule` 在敏感场景下重新拉开差异:
  - `u2 nearest_uav total_energy = 134.1836 +/- 10.0652`
  - `u2 least_loaded_uav total_energy = 130.1174 +/- 0.5214`
  - `u3 nearest_uav total_energy = 182.0858 +/- 1.9352`
  - `u3 least_loaded_uav total_energy = 149.7891 +/- 4.5289`
- `PPO vs heuristic`:
  - `u2 + nearest_uav`
    - PPO `completion_rate=0.9846 +/- 0.0137`
    - PPO `average_latency=0.4082 +/- 0.0540`
    - PPO `total_energy=28.6199 +/- 7.2119`
    - heuristic `total_energy=16.5777 +/- 16.3382`
  - `u3 + nearest_uav`
    - PPO `completion_rate=0.9931 +/- 0.0120`
    - PPO `average_latency=0.4314 +/- 0.0427`
    - PPO `total_energy=46.2983 +/- 15.4309`
    - heuristic `total_energy=40.6808 +/- 28.0994`
- `消融`:
  - `main total_energy=28.6199 +/- 7.2119`
  - `no_energy_shaped_reward total_energy=27.7935 +/- 11.4441`
  - `no_movement_budget total_energy=111.0782 +/- 7.9667`
  - `compare_ch4` 中 `fairness_uav_load` 在单 UAV 对齐实验里保持 `delta=null`，不再被写成 `0.0`

### 新增/更新结果文件

- `第四章/results/marl_train_shared_ppo_u2_nearest_uav.json`
- `第四章/results/marl_eval_shared_ppo_u2_nearest_uav.json`
- `第四章/results/paper_stage6/compare_ch4_multiseed_summary.json`
- `第四章/results/paper_stage6/assignment_multiseed_summary.json`
- `第四章/results/paper_stage6/ppo_vs_heuristic_multiseed_summary.json`
- `第四章/results/paper_stage6/ablation_multiseed_summary.json`
- `第四章/results/paper_stage6/tables/table_main_results.md`
- `第四章/results/paper_stage6/tables/table_assignment_comparison.md`
- `第四章/results/paper_stage6/tables/table_ppo_vs_heuristic.md`
- `第四章/results/paper_stage6/tables/table_ablation.md`
- `第四章/results/paper_stage6/figures/final_training_return_curve.png`
- `第四章/results/paper_stage6/figures/final_training_energy_curve.png`
- `第四章/results/paper_stage6/figures/final_ppo_vs_heuristic.png`
- `第四章/results/paper_stage6/figures/final_assignment_comparison.png`
- `第四章/results/paper_stage6/figures/final_ablation_energy.png`
