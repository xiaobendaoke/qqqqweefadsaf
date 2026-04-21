# 第四章验证记录

> 当前结果目录约定为：stage5 产物写入 `第四章/results/stage5/paper_stage5_v2/`，stage6 产物写入 `第四章/results/stage6/paper_stage6_v2/`，验证刷新写入 `第四章/results/stage6/paper_stage6_v2/verification_refresh.json`。

## 当前验证与复跑命令

```powershell
.\.venv\Scripts\python.exe 第四章/run_refresh_verification.py
.\.venv\Scripts\python.exe 第四章/run_paper_experiments.py --tuning-seeds 42 52 62 --eval-episodes 64 --device auto
.\.venv\Scripts\python.exe 第四章/run_finalize_paper.py --seeds 72 82 92 --eval-episodes 64 --device auto
```

当前已固定的关键输出位置：

- `第四章/results/stage6/paper_stage6_v2/verification_refresh.json`
- `第四章/results/stage5/paper_stage5_v2/tuning_summary.json`
- `第四章/results/stage5/paper_stage5_v2/main_experiment_matrix.json`
- `第四章/results/stage5/paper_stage5_v2/ablation_summary.json`
- `第四章/results/stage5/paper_stage5_v2/assignment_rule_matrix.json`
- `第四章/results/stage5/paper_stage5_v2/paper_summary.md`
- `第四章/results/stage6/paper_stage6_v2/`
  - `run_finalize_paper.py` 复跑后会在此目录生成 `reproducibility_package.json`、多 seed summary、tables` 与 `figures`
  - `run_refresh_verification.py` 当前会在此目录生成 `verification_refresh.json`

## 当前 smoke 与基础验证

`run_refresh_verification.py` 当前覆盖：

- `smoke.u1`
- `smoke.u2`
- `experiments.default_u2`
- `experiments.hard_u2`
- `experiments.sensitive_u2_nearest`
- `experiments.sensitive_u2_least_loaded`
- `experiments.sensitive_u3_nearest`
- `experiments.sensitive_u3_least_loaded`
- `compare_ch4`

关键结果：

- `u1 env_step`
  - `completion_rate=1.0`
- `u1 observation_length=56`
- `u2 observation_length=56`
- `default_u2 total_energy=2.5787417575365508`
- `hard_u2 completion_rate=0.11627906976744186`
- `compare_ch4 total_energy delta=0.0`

## Stage5 当前结果

当前 stage5 已按真实三 seed 调参运行，使用：

- `tuning_train_seeds=[42, 52, 62]`
- `tuning_eval_seeds=[142, 152, 162]`
- `seed_split_policy=stage5_multi_seed_tuning_only`

当前自动选中候选：

- `selected_candidate=freeze_energy2_240`

调参汇总中的候选对比摘要：

- `freeze_energy2_240`
  - `completion_rate=0.9988 +/- 0.0000`
  - `total_energy=14.4258 +/- 2.3820`
  - `average_latency=0.3719 +/- 0.0029`
- `cons_lowlr`
  - `completion_rate=0.9988 +/- 0.0000`
  - `total_energy=16.8387 +/- 7.3400`
  - `average_latency=0.3720 +/- 0.0039`

当前 `paper_stage5_v2` 已不再是单 seed 调参结果。

## Stage6 当前结果

当前 stage6 已按与 stage5 不重叠的最终种子重新复跑：

- `selected_main_candidate=freeze_energy2_240`
- `seed_split_policy=stage5_tuning_vs_stage6_final_disjoint`
- `tuning_train_seeds=[42, 52, 62]`
- `tuning_eval_seeds=[142, 152, 162]`
- `final_train_seeds=[72, 82, 92]`
- `final_eval_seeds=[172, 182, 192]`

`PPO vs heuristic` 当前多 seed 汇总：

- `u2 + nearest_uav`
  - PPO `completion_rate=0.9995 +/- 0.0005`
  - PPO `average_latency=0.3629 +/- 0.0012`
  - PPO `total_energy=12.5034 +/- 1.1116`
  - heuristic `total_energy=5.7030 +/- 0.6403`
  - `delta_total_energy=6.8004 +/- 0.7908`
- `u3 + nearest_uav`
  - PPO `completion_rate=0.9998 +/- 0.0003`
  - PPO `average_latency=0.3568 +/- 0.0019`
  - PPO `total_energy=18.8824 +/- 2.1100`
  - heuristic `total_energy=6.6259 +/- 1.0564`
  - `delta_total_energy=12.2566 +/- 3.0060`

## Assignment 与消融当前结果

assignment 当前多 seed 汇总：

- `u2 least_loaded_uav`
  - `total_energy=12.1858 +/- 0.0314`
  - `fairness_uav_load=0.9975 +/- 0.0002`
- `u2 nearest_uav`
  - `total_energy=12.3372 +/- 0.0311`
  - `fairness_uav_load=0.9960 +/- 0.0003`
- `u3 least_loaded_uav`
  - `total_energy=12.3932 +/- 0.0328`
  - `fairness_uav_load=0.9631 +/- 0.0014`
- `u3 nearest_uav`
  - `total_energy=12.5217 +/- 0.0335`
  - `fairness_uav_load=0.9681 +/- 0.0004`

当前消融汇总：

- `main total_energy=12.5034 +/- 1.1116`
- `with_reward_shaping total_energy=12.5632 +/- 0.8332`
- `no_movement_budget total_energy=41.7085 +/- 12.7220`

当前更稳妥的结论是：

- `movement budget` 对控制无效移动仍然关键
- 当前 reward shaping 组合没有带来稳定的总能耗收益
- 当前完整复跑结果下，PPO 的总能耗仍高于 heuristic

## Compare-Ch4 当前结果

当前 `paper_stage6_v2/compare_ch4_multiseed_summary.json` 继续表明：

- `completion_rate delta = 0.0000 +/- 0.0000`
- `average_latency delta = 0.0000 +/- 0.0000`
- `total_energy delta = 0.0000 +/- 0.0000`
- `cache_hit_rate delta = 0.0000 +/- 0.0000`
- `fairness_uav_load delta = null`
- 新增能耗拆分指标也保持逐项对齐

## 引用规则

- 论文、README、附录若需要引用第四章当前结果，应优先引用 `paper_stage5_v2` / `paper_stage6_v2` 下的自动生成 JSON、表格与图
- 若本文件与这些结果目录不一致，应以结果目录为准并重新刷新本文件
