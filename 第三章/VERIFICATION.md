# 第三章验证记录

> 当前真值以 `第三章/results/verification_refresh.json` 为准。本文件只保留当前代码版本下的可复跑验证摘要，不再把历史手填数字当作当前论文引用来源。文中的 `MPC shell` 指基于候选动作滚动评分的 look-ahead heuristic shell，而非标准优化求解式 MPC。

## 当前验证命令

```powershell
.\.venv\Scripts\python.exe 第三章/run_refresh_verification.py
```

该命令会统一刷新：

- smoke 验证
- 单 UAV baseline 实验
- `compare-ch4` 退化一致性验证

当前刷新结果文件：

- `第三章/results/verification_refresh.json`

## Smoke 当前结果

当前 smoke 子项：

- `import_only`
- `task_contract`
- `comms_contract`
- `scheduler_contract`
- `env_step`
- `episode`

关键结果：

- `task_contract`
  - `schema_version=observation.v2`
  - `observation_dim=56`
- `comms_contract`
  - `success_probability=0.9996346529295138`
- `scheduler_contract`
  - `decision_target=bs`
  - `decision_total_latency=0.17511975507332653`
- `env_step`
  - `completion_rate=1.0`
  - `average_latency=0.3321061383970643`
- `episode`
  - `completion_rate=1.0`
  - `average_latency=0.4002087360101324`
  - `total_energy=2.9462380689981567`

## 实验当前结果

`run_refresh_verification.py` 当前覆盖的单 UAV 实验包括：

- `heuristic`
- `mpc`
- `fixed_point`
- `fixed_patrol`

对应当前结果摘要：

- `heuristic`
  - `total_energy=2.9462380689981567`
- `mpc`
  - `total_energy=2.9462380689981567`
- `fixed_point`
  - `total_energy=7.29604341109928`
- `fixed_patrol`
  - `total_energy=547.1281479765431`

若需要更细的 episode 日志、轨迹导出和能耗分解，请直接查看 `verification_refresh.json` 中的 `experiments` 字段。

## Compare-Ch4 当前结果

当前 `compare-ch4` 由 `verification_refresh.json.compare_ch4` 记录，验证的是：

- 第三章 heuristic shared 口径
- 第四章 `NUM_UAVS=1` 退化口径

当前结论：

- `completion_rate delta = 0.0`
- `average_latency delta = 0.0`
- `total_energy delta = 0.0`
- `cache_hit_rate delta = 0.0`
- `fairness_uav_load delta = null`
- 新增能耗拆分指标也保持逐项一致

当前对齐样例：

- `chapter3 total_energy = 2.9462380689981567`
- `chapter4 total_energy = 2.9462380689981567`

## 引用规则

- 论文、README、附录若需要引用第三章当前验证数字，应优先引用 `第三章/results/verification_refresh.json`
- 若本文件与 JSON 不一致，应以 JSON 为准并重新刷新本文件
