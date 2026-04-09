from __future__ import annotations

import math

from ..config import SystemConfig
from ..entities import UAVNode


def _safe_norm(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return float(value) / float(scale)


def _cache_bitmap(uav: UAVNode, config: SystemConfig) -> list[float]:
    return [1.0 if service_type in uav.service_cache else 0.0 for service_type in range(config.num_service_types)]


def build_uav_state(*, uav: UAVNode, all_uavs: list[UAVNode], config: SystemConfig) -> dict[str, list[float] | dict[str, list[float]]]:
    own_state = [
        uav.position[0] / config.area_width,
        uav.position[1] / config.area_height,
        uav.energy_ratio,
        _safe_norm(uav.current_queue_length, max(1, config.num_users)),
        _safe_norm(uav.current_queue_delay, max(config.time_slot_duration, 1e-6)),
        _safe_norm(uav.assigned_task_count_step, max(1, config.num_users)),
        _safe_norm(uav.completed_task_count_step, max(1, config.num_users)),
        len(uav.service_cache) / max(1, uav.service_cache_capacity),
    ]

    neighbor_blocks: list[list[float]] = []
    neighbors = [other for other in all_uavs if other.uav_id != uav.uav_id]
    neighbors.sort(key=lambda item: math.dist((item.position[0], item.position[1]), (uav.position[0], uav.position[1])))
    for neighbor in neighbors[: config.observation_max_neighbors]:
        neighbor_blocks.append(
            [
                (neighbor.position[0] - uav.position[0]) / config.area_width,
                (neighbor.position[1] - uav.position[1]) / config.area_height,
                _safe_norm(neighbor.current_queue_length, max(1, config.num_users)),
                _safe_norm(neighbor.current_queue_delay, max(config.time_slot_duration, 1e-6)),
                _safe_norm(neighbor.assigned_task_count_step, max(1, config.num_users)),
                _safe_norm(neighbor.completed_task_count_step, max(1, config.num_users)),
                len(neighbor.service_cache) / max(1, neighbor.service_cache_capacity),
            ]
        )
    while len(neighbor_blocks) < config.observation_max_neighbors:
        neighbor_blocks.append([0.0] * 7)

    return {
        "own_state": own_state,
        "neighbor_uav_states": neighbor_blocks,
        "queue_load_summary": own_state[3:7],
        "local_cache_summary": _cache_bitmap(uav, config),
    }


def _flat_schema(sections: list[tuple[str, list[str], int]]) -> dict[str, object]:
    flat_fields: list[str] = []
    section_slices: dict[str, list[int]] = {}
    cursor = 0
    for name, fields, repeat in sections:
        start = cursor
        if repeat == 1:
            flat_fields.extend(fields)
            cursor += len(fields)
        else:
            for index in range(repeat):
                flat_fields.extend([f"{name}[{index}].{field}" for field in fields])
                cursor += len(fields)
        section_slices[name] = [start, cursor]
    return {"flat_fields": flat_fields, "section_slices": section_slices, "total_length": cursor}


def _base_section_defs(config: SystemConfig) -> list[tuple[str, list[str], int]]:
    own_fields = [
        "self_x_norm",
        "self_y_norm",
        "self_energy_ratio",
        "self_queue_length_norm",
        "self_queue_delay_norm",
        "self_assigned_task_count_norm",
        "self_completed_task_count_norm",
        "self_cache_usage_ratio",
    ]
    neighbor_fields = [
        "neighbor_rel_x_norm",
        "neighbor_rel_y_norm",
        "neighbor_queue_length_norm",
        "neighbor_queue_delay_norm",
        "neighbor_assigned_task_count_norm",
        "neighbor_completed_task_count_norm",
        "neighbor_cache_usage_ratio",
    ]
    queue_fields = [
        "queue_length_norm",
        "queue_delay_norm",
        "assigned_task_count_norm",
        "completed_task_count_norm",
    ]
    cache_fields = [f"cache_service_{service_type}" for service_type in range(config.num_service_types)]
    user_fields = [
        "user_rel_x_norm",
        "user_rel_y_norm",
        "user_pending_task_count_norm",
        "user_min_slack_norm",
        "user_service_type_norm",
    ]
    return [
        ("own_state", own_fields, 1),
        ("neighbor_uav_states", neighbor_fields, config.observation_max_neighbors),
        ("queue_load_summary", queue_fields, 1),
        ("local_cache_summary", cache_fields, 1),
        ("associated_user_state", user_fields, config.observation_max_users),
    ]


def observation_schema(config: SystemConfig) -> dict[str, object]:
    sections = _base_section_defs(config)
    frozen = _flat_schema(sections)
    return {
        "schema_version": "observation.v1",
        "sections": [
            {"name": name, "fields": fields, "repeat": repeat}
            for name, fields, repeat in sections
        ],
        **frozen,
    }


def uav_state_schema(config: SystemConfig) -> dict[str, object]:
    own_fields = _base_section_defs(config)[0][1]
    neighbor_fields = _base_section_defs(config)[1][1]
    queue_fields = _base_section_defs(config)[2][1]
    cache_fields = _base_section_defs(config)[3][1]
    sections = [
        ("own_state", own_fields, 1),
        ("neighbor_uav_states", neighbor_fields, config.observation_max_neighbors),
        ("queue_load_summary", queue_fields, 1),
        ("local_cache_summary", cache_fields, 1),
    ]
    frozen = _flat_schema(sections)
    return {
        "schema_version": "uav_state.v1",
        "sections": [
            {"name": name, "fields": fields, "repeat": repeat}
            for name, fields, repeat in sections
        ],
        **frozen,
    }
