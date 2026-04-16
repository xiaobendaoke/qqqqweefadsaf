"""共享短实验运行与指标对比模块。

该模块为第三章和第四章提供统一的“运行若干 episode 并聚合结果”的辅助流程，
适用于 smoke、退化一致性比较和启发式基线实验等轻量实验场景。
"""

from __future__ import annotations

from typing import Any, Callable


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def run_short_experiment(
    *,
    env_factory: Callable[[dict[str, Any]], Any],
    policy_fn: Callable[..., list[list[float]]],
    overrides: dict[str, Any] | None,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    """运行轻量实验并汇总 episode 指标与日志。

    参数：
        env_factory: 环境构造函数，接收覆盖配置并返回环境实例。
        policy_fn: 策略函数，输入观测和环境实例，输出动作列表。
        overrides: 传递给环境构造函数的覆盖配置。
        episodes: 需要运行的 episode 数量。
        seed: 实验起始随机种子；实际每个 episode 会在此基础上递增。

    返回：
        包含平均指标、每回合摘要、日志 schema 和完整 episode 日志的结果字典。
    """
    episode_summaries: list[dict[str, Any]] = []
    episode_logs: list[dict[str, Any]] = []
    aggregate_metrics: list[dict[str, float | None]] = []
    log_schema: dict[str, Any] | None = None
    for episode_idx in range(episodes):
        env = env_factory(overrides or {})
        episode_seed = seed + episode_idx
        reset_result = env.reset(seed=episode_seed)
        log_schema = env.get_episode_log_schema()
        observations = reset_result["observations"]
        last_step = None
        while True:
            actions = policy_fn(observations, env)
            last_step = env.step(actions)
            observations = last_step["observations"]
            if last_step["terminated"] or last_step["truncated"]:
                break
        summary = env.export_episode_summary()
        episode_log = env.export_episode_log(episode_index=episode_idx, seed=episode_seed)
        episode_summaries.append(
            {
                "episode": episode_idx,
                "seed": episode_seed,
                "metrics": summary["metrics"],
                "last_info": last_step["info"] if last_step else {},
            }
        )
        episode_logs.append(episode_log)
        aggregate_metrics.append(summary["metrics"])

    metric_keys = aggregate_metrics[0].keys() if aggregate_metrics else []
    averaged_metrics: dict[str, float | None] = {}
    for key in metric_keys:
        numeric_values = [metrics[key] for metrics in aggregate_metrics if metrics[key] is not None]
        averaged_metrics[key] = _mean([float(value) for value in numeric_values]) if numeric_values else None

    return {
        "episodes": episodes,
        "seed": seed,
        "overrides": overrides or {},
        "averaged_metrics": averaged_metrics,
        "episode_summaries": episode_summaries,
        "episode_log_schema": log_schema,
        "episode_logs": episode_logs,
    }


def compare_metric_dicts(left: dict[str, float | None], right: dict[str, float | None]) -> dict[str, dict[str, float | None]]:
    """对齐比较两组指标字典，并给出逐项差值。"""
    comparison: dict[str, dict[str, float | None]] = {}
    all_keys = sorted(set(left.keys()) | set(right.keys()))
    for key in all_keys:
        left_value = left.get(key)
        right_value = right.get(key)
        delta = None
        if left_value is not None and right_value is not None:
            delta = float(left_value) - float(right_value)
        comparison[key] = {"left": left_value, "right": right_value, "delta": delta}
    return comparison
