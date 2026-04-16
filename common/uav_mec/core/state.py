"""UAV 状态编码与 schema 定义模块。

该模块负责把 UAV 的位置、能量、队列和缓存状态组织为结构化状态，
并生成与之对应的扁平化字段 schema，供观测构造、日志导出和训练接口复用。

边界说明：
该模块只描述状态编码方式，不直接推进环境状态演化。
"""

from __future__ import annotations

import math

from ..config import SystemConfig
from ..entities import UAVNode
from ..scheduler.service_cache import cache_score_snapshot


def _safe_norm(value: float, scale: float) -> float:
    """避免零除的归一化辅助函数。"""
    if scale <= 0:
        return 0.0
    return float(value) / float(scale)


def _cache_bitmap(uav: UAVNode, config: SystemConfig) -> list[float]:
    return [1.0 if service_type in uav.service_cache else 0.0 for service_type in range(config.num_service_types)]


def _cache_score_norm(uav: UAVNode, config: SystemConfig) -> list[float]:
    scores = cache_score_snapshot(uav, num_service_types=config.num_service_types)
    max_score = max([1.0e-6, *scores])
    return [float(score) / max_score for score in scores]


def build_uav_state(*, uav: UAVNode, all_uavs: list[UAVNode], config: SystemConfig) -> dict[str, list[float] | dict[str, list[float]]]:
    """构造单架 UAV 的结构化状态，再供观测拼接或日志导出使用。"""
    own_state = [
        uav.position[0] / config.area_width,
        uav.position[1] / config.area_height,
        uav.energy_ratio,
        _safe_norm(uav.current_coverage_load, max(1, config.num_users)),
        _safe_norm(uav.assigned_task_count_step, max(1, config.num_users)),
        _safe_norm(uav.completed_task_count_step, max(1, config.num_users)),
        len(uav.service_cache) / max(1, uav.service_cache_capacity),
    ]
    tx_queue_summary = [
        _safe_norm(uav.current_tx_queue_length, max(1, config.num_users)),
        _safe_norm(uav.current_tx_queue_delay, max(config.time_slot_duration, 1e-6)),
    ]
    compute_queue_summary = [
        _safe_norm(uav.current_compute_queue_length, max(1, config.num_users)),
        _safe_norm(uav.current_compute_queue_delay, max(config.time_slot_duration, 1e-6)),
    ]
    backlog_summary = [
        _safe_norm(uav.current_backlog_load, max(1, config.num_users * config.task_arrival_max_per_step)),
        _safe_norm(uav.current_coverage_load, max(1, config.num_users)),
    ]

    neighbor_blocks: list[list[float]] = []
    neighbors = [other for other in all_uavs if other.uav_id != uav.uav_id]
    neighbors.sort(key=lambda item: math.dist((item.position[0], item.position[1]), (uav.position[0], uav.position[1])))
    # 邻居槽位固定长度，缺失位置用零填充，保证多配置下 schema 稳定。
    for neighbor in neighbors[: config.observation_max_neighbors]:
        neighbor_blocks.append(
            [
                (neighbor.position[0] - uav.position[0]) / config.area_width,
                (neighbor.position[1] - uav.position[1]) / config.area_height,
                _safe_norm(neighbor.current_tx_queue_length, max(1, config.num_users)),
                _safe_norm(neighbor.current_tx_queue_delay, max(config.time_slot_duration, 1e-6)),
                _safe_norm(neighbor.current_compute_queue_length, max(1, config.num_users)),
                _safe_norm(neighbor.current_compute_queue_delay, max(config.time_slot_duration, 1e-6)),
                _safe_norm(neighbor.current_backlog_load, max(1, config.num_users * config.task_arrival_max_per_step)),
                _safe_norm(neighbor.assigned_task_count_step, max(1, config.num_users)),
                len(neighbor.service_cache) / max(1, neighbor.service_cache_capacity),
            ]
        )
    while len(neighbor_blocks) < config.observation_max_neighbors:
        neighbor_blocks.append([0.0] * 9)

    return {
        "own_state": own_state,
        "neighbor_uav_states": neighbor_blocks,
        "tx_queue_summary": tx_queue_summary,
        "compute_queue_summary": compute_queue_summary,
        "task_backlog_summary": backlog_summary,
        "local_cache_summary": _cache_bitmap(uav, config),
        "cache_score_summary": _cache_score_norm(uav, config),
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
        "self_coverage_load_norm",
        "self_assigned_task_count_norm",
        "self_completed_task_count_norm",
        "self_cache_usage_ratio",
    ]
    neighbor_fields = [
        "neighbor_rel_x_norm",
        "neighbor_rel_y_norm",
        "neighbor_tx_queue_length_norm",
        "neighbor_tx_queue_delay_norm",
        "neighbor_compute_queue_length_norm",
        "neighbor_compute_queue_delay_norm",
        "neighbor_backlog_norm",
        "neighbor_assigned_task_count_norm",
        "neighbor_cache_usage_ratio",
    ]
    tx_fields = ["tx_queue_length_norm", "tx_queue_delay_norm"]
    compute_fields = ["compute_queue_length_norm", "compute_queue_delay_norm"]
    backlog_fields = ["backlog_load_norm", "coverage_load_norm"]
    cache_fields = [f"cache_service_{service_type}" for service_type in range(config.num_service_types)]
    cache_score_fields = [f"cache_score_service_{service_type}" for service_type in range(config.num_service_types)]
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
        ("tx_queue_summary", tx_fields, 1),
        ("compute_queue_summary", compute_fields, 1),
        ("task_backlog_summary", backlog_fields, 1),
        ("local_cache_summary", cache_fields, 1),
        ("cache_score_summary", cache_score_fields, 1),
        ("associated_user_state", user_fields, config.observation_max_users),
    ]


def observation_schema(config: SystemConfig) -> dict[str, object]:
    """返回完整观测向量的字段展开方式与切片位置。"""
    sections = _base_section_defs(config)
    frozen = _flat_schema(sections)
    return {
        "schema_version": "observation.v2",
        "sections": [{"name": name, "fields": fields, "repeat": repeat} for name, fields, repeat in sections],
        **frozen,
    }


def uav_state_schema(config: SystemConfig) -> dict[str, object]:
    """返回不含用户 backlog 块的 UAV 状态 schema。"""
    base = _base_section_defs(config)
    sections = base[:-1]
    frozen = _flat_schema(sections)
    return {
        "schema_version": "uav_state.v2",
        "sections": [{"name": name, "fields": fields, "repeat": repeat} for name, fields, repeat in sections],
        **frozen,
    }
