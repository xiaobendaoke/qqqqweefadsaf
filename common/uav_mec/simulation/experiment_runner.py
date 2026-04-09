from __future__ import annotations

from typing import Any, Callable


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def run_short_experiment(
    *,
    env_factory: Callable[[dict[str, Any]], Any],
    policy_fn: Callable[[list[list[float]]], list[list[float]]],
    overrides: dict[str, Any] | None,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
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
            try:
                actions = policy_fn(observations, env)
            except TypeError:
                actions = policy_fn(observations)
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
