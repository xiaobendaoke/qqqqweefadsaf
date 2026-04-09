from __future__ import annotations

from ..config import SystemConfig
from ..core.action import action_schema
from ..core.state import observation_schema, uav_state_schema
from ..entities import UAVNode


def episode_log_schema(config: SystemConfig, agent_ids: list[str]) -> dict[str, object]:
    return {
        "schema_version": "episode_log.v1",
        "top_level_fields": [
            "chapter_name",
            "episode_index",
            "seed",
            "num_uavs",
            "num_users",
            "assignment_rule",
            "global_metrics",
            "per_uav_metrics",
            "action_schema",
            "observation_schema",
            "uav_state_schema",
        ],
        "action_schema": action_schema(num_agents=config.num_uavs, agent_ids=agent_ids, config=config),
        "observation_schema": observation_schema(config),
        "uav_state_schema": uav_state_schema(config),
        "per_uav_metrics_fields": [
            "agent_id",
            "uav_id",
            "served_task_count_total",
            "assigned_task_count_total",
            "completed_task_count_total",
            "current_queue_length",
            "current_queue_delay",
            "cumulative_queue_delay",
            "max_queue_length",
            "remaining_energy_j",
            "energy_used_j",
            "cache_usage_ratio",
            "cache_services",
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
                "current_queue_length": uav.current_queue_length,
                "current_queue_delay": uav.current_queue_delay,
                "cumulative_queue_delay": uav.cumulative_queue_delay,
                "max_queue_length": uav.max_queue_length,
                "remaining_energy_j": uav.remaining_energy_j,
                "energy_used_j": uav.energy_capacity_j - uav.remaining_energy_j,
                "cache_usage_ratio": len(uav.service_cache) / max(1, uav.service_cache_capacity),
                "cache_services": sorted(uav.service_cache),
            }
        )
    return payload
