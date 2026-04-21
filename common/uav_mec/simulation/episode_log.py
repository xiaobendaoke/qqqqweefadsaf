"""Episode 日志 schema 定义模块。

该模块负责描述 episode 日志导出的字段结构，
并汇总每架 UAV 在整个 episode 内的服务、负载和队列表现，便于论文分析和结果复现。
"""

from __future__ import annotations

from ..config import SystemConfig
from ..core.action import action_schema
from ..core.state import observation_schema, uav_state_schema
from ..entities import UAVNode
from ..metrics import episode_metric_schema
from ..metrics import step_signal_schema


def episode_log_schema(config: SystemConfig, agent_ids: list[str]) -> dict[str, object]:
    return {
        "schema_version": "episode_log.v2",
        "top_level_fields": [
            "chapter_name",
            "episode_index",
            "seed",
            "num_uavs",
            "num_users",
            "assignment_rule",
            "metric_schemas",
            "global_metrics",
            "per_uav_metrics",
            "action_schema",
            "observation_schema",
            "uav_state_schema",
            "step_signals",
            "energy_breakdown",
            "queue_breakdown",
            "cache_events",
            "task_lifecycle_counts",
        ],
        "metric_schemas": {
            "episode_metrics": episode_metric_schema(),
            "step_signals": step_signal_schema(),
        },
        "action_schema": action_schema(num_agents=config.num_uavs, agent_ids=agent_ids, config=config),
        "observation_schema": observation_schema(config),
        "uav_state_schema": uav_state_schema(config),
        "per_uav_metrics_fields": [
            "agent_id",
            "uav_id",
            "served_task_count_total",
            "assigned_task_count_total",
            "completed_task_count_total",
            "current_tx_queue_length",
            "current_tx_queue_delay",
            "current_compute_queue_length",
            "current_compute_queue_delay",
            "current_backlog_load",
            "current_coverage_load",
            "cumulative_tx_queue_delay",
            "cumulative_compute_queue_delay",
            "max_tx_queue_length",
            "max_compute_queue_length",
            "remaining_energy_j",
            "energy_used_j",
            "cache_usage_ratio",
            "cache_services",
            "cache_scores",
        ],
    }


def build_per_uav_metrics(uavs: list[UAVNode]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for uav in uavs:
        payload.append(
            {
                "agent_id": f"uav_{uav.uav_id}",
                "uav_id": uav.uav_id,
                "served_task_count_total": uav.served_task_count,
                "assigned_task_count_total": uav.total_assigned_task_count,
                "completed_task_count_total": uav.total_completed_task_count,
                "current_tx_queue_length": uav.current_tx_queue_length,
                "current_tx_queue_delay": uav.current_tx_queue_delay,
                "current_compute_queue_length": uav.current_compute_queue_length,
                "current_compute_queue_delay": uav.current_compute_queue_delay,
                "current_backlog_load": uav.current_backlog_load,
                "current_coverage_load": uav.current_coverage_load,
                "cumulative_tx_queue_delay": uav.cumulative_tx_queue_delay,
                "cumulative_compute_queue_delay": uav.cumulative_compute_queue_delay,
                "max_tx_queue_length": uav.max_tx_queue_length,
                "max_compute_queue_length": uav.max_compute_queue_length,
                "remaining_energy_j": uav.remaining_energy_j,
                "energy_used_j": uav.energy_capacity_j - uav.remaining_energy_j,
                "cache_usage_ratio": len(uav.service_cache) / max(1, uav.service_cache_capacity),
                "cache_services": sorted(uav.service_cache),
                "cache_scores": {str(key): float(value) for key, value in sorted(uav.cache_value_scores.items())},
            }
        )
    return payload
