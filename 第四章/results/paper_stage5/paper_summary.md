# 第五阶段实验汇总

## 最终 PPO 主配置

- output_tag: `paper_main_u2`
- train_episodes: `30`
- actor_lr / critic_lr: `0.00025` / `0.0008`
- clip_ratio: `0.18`
- entropy_coef: `0.008`
- value_coef: `0.6`
- reward_energy_weight: `1.5`
- reward_action_magnitude_weight: `0.2`
- use_movement_budget: `True`

## A. Chapter3 vs Chapter4(NUM_UAVS=1)

| metric | delta |
| --- | ---: |
| average_latency | 0.0 |
| cache_hit_rate | 0.0 |
| completion_rate | 0.0 |
| deadline_violation_rate | 0.0 |
| fairness_uav_load | None |
| fairness_user_completion | 0.0 |
| reliability_violation_rate | 0.0 |
| total_energy | 0.0 |

## B. nearest_uav vs least_loaded_uav

| setting | completion_rate | average_latency | total_energy | fairness_uav_load |
| --- | ---: | ---: | ---: | ---: |
| u2 nearest_uav | 1.0 | 0.7913068339024849 | 32.94511440276091 | 0.9922404903184621 |
| u2 least_loaded_uav | 1.0 | 0.7953474494987626 | 32.793658122393566 | 0.9995228523080433 |
| u3 nearest_uav | 1.0 | 0.7679370024591761 | 76.26380151222608 | 0.7519260328776769 |
| u3 least_loaded_uav | 1.0 | 0.7392624618340504 | 76.45117456803078 | 0.999169751169418 |

## C. PPO vs heuristic

| setting | PPO completion | PPO latency | PPO energy | heuristic energy | delta energy |
| --- | ---: | ---: | ---: | ---: | ---: |
| u2 nearest_uav | 1.0 | 0.25996951130234985 | 50.27020852408417 | 27.865925714269967 | 22.4042828098142 |
| u3 nearest_uav | 1.0 | 0.26831519456870234 | 71.75571435026387 | 42.57059405190918 | 29.18512029835469 |

## D. 最小消融

| variant | completion_rate | average_latency | total_energy | delta energy vs heuristic |
| --- | ---: | ---: | ---: | ---: |
| no_energy_shaped_reward | 1.0 | 0.25997478905086757 | 62.063485676623266 | 34.197559962353296 |
| no_movement_budget | 1.0 | 0.25997329396776453 | 74.0749144830099 | 46.20898876873994 |
