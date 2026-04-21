"""局部观测构造模块。

该模块先构造面向联合优化的结构化 observation，再按 legacy 训练接口
导出与旧版兼容的扁平向量表示，供第三章策略和第四章 MARL 继续复用。
"""

from __future__ import annotations

import math
from typing import Literal

from ..config import SystemConfig
from ..entities import BaseStation, ServiceCatalog, UAVNode, UserEquipment
from ..scheduler.compute_queue import ComputeQueue
from ..scheduler.offloading import OffloadingCandidate, enumerate_offloading_candidates
from ..scheduler.tdma import TDMAQueue
from ..task import Task
from .state import build_uav_state
from .state import cache_candidate_feature_fields
from .state import offloading_candidate_count
from .state import observation_task_slot_count
from .state import offloading_candidate_feature_fields
from .state import task_slot_feature_fields


def _safe_norm(value: float, scale: float) -> float:
    if scale <= 0.0:
        return 0.0
    return float(value) / float(scale)


def _remaining_margin(task: Task, current_time: float) -> float:
    return max(0.0, float(task.deadline - current_time))


def _build_associated_user_state(
    *,
    uav: UAVNode,
    users: list[UserEquipment],
    pending_tasks: list[Task],
    config: SystemConfig,
    current_time: float,
) -> tuple[list[dict[str, object]], list[float]]:
    pending_by_user: dict[int, list[Task]] = {}
    for task in pending_tasks:
        pending_by_user.setdefault(task.user_id, []).append(task)
    sorted_users = sorted(
        users,
        key=lambda item: (
            0 if item.user_id in pending_by_user else 1,
            math.dist((item.position[0], item.position[1]), (uav.position[0], uav.position[1])),
        ),
    )
    blocks: list[dict[str, object]] = []
    flat_values: list[float] = []
    for user in sorted_users[: config.observation_max_users]:
        user_tasks = pending_by_user.get(user.user_id, [])
        min_slack = min((_remaining_margin(task, current_time) for task in user_tasks), default=0.0)
        representative_task = min(user_tasks, key=lambda task: _remaining_margin(task, current_time)) if user_tasks else None
        service_type = representative_task.service_type if representative_task is not None else 0
        features = [
            (user.position[0] - uav.position[0]) / config.area_width,
            (user.position[1] - uav.position[1]) / config.area_height,
            float(len(user_tasks)) / max(1, config.num_users * config.task_arrival_max_per_step),
            float(min_slack) / max(config.task_slack_range[1], 1e-6),
            float(service_type) / max(1, config.num_service_types - 1),
        ]
        blocks.append(
            {
                "user_id": int(user.user_id),
                "active": bool(user_tasks),
                "user_rel_x_norm": float(features[0]),
                "user_rel_y_norm": float(features[1]),
                "user_pending_task_count_norm": float(features[2]),
                "user_min_slack_norm": float(features[3]),
                "user_service_type_norm": float(features[4]),
            }
        )
        flat_values.extend(features)
    while len(blocks) < config.observation_max_users:
        blocks.append(
            {
                "user_id": None,
                "active": False,
                "user_rel_x_norm": 0.0,
                "user_rel_y_norm": 0.0,
                "user_pending_task_count_norm": 0.0,
                "user_min_slack_norm": 0.0,
                "user_service_type_norm": 0.0,
            }
        )
        flat_values.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    return blocks, flat_values


def _select_task_slots(
    *,
    uav: UAVNode,
    users_by_id: dict[int, UserEquipment],
    pending_tasks: list[Task],
    config: SystemConfig,
    current_time: float,
) -> list[Task]:
    active_tasks = [task for task in pending_tasks if task.status == "pending"]
    ranked = sorted(
        active_tasks,
        key=lambda task: (
            0 if task.associated_uav_id == uav.uav_id or task.assigned_uav_id == uav.uav_id else 1,
            0
            if math.dist(
                (users_by_id[task.user_id].position[0], users_by_id[task.user_id].position[1]),
                (uav.position[0], uav.position[1]),
            )
            <= config.uav_coverage_radius
            else 1,
            _remaining_margin(task, current_time),
            math.dist(
                (users_by_id[task.user_id].position[0], users_by_id[task.user_id].position[1]),
                (uav.position[0], uav.position[1]),
            ),
            task.arrival_time,
            task.task_id,
        ),
    )
    return ranked[: observation_task_slot_count(config)]


def _build_task_slot_entry(
    *,
    slot_index: int,
    task: Task | None,
    uav: UAVNode,
    users_by_id: dict[int, UserEquipment],
    pending_by_user: dict[int, list[Task]],
    config: SystemConfig,
    current_time: float,
) -> dict[str, object]:
    field_names = task_slot_feature_fields()
    if task is None:
        payload = {field: 0.0 for field in field_names}
        payload.update({"slot_index": slot_index, "task_id": None, "user_id": None, "service_type": None, "active": False})
        return payload

    user = users_by_id[task.user_id]
    remaining_slack = _remaining_margin(task, current_time)
    distance = math.dist((user.position[0], user.position[1]), (uav.position[0], uav.position[1]))
    features = {
        "active_flag": 1.0,
        "user_rel_x_norm": (user.position[0] - uav.position[0]) / config.area_width,
        "user_rel_y_norm": (user.position[1] - uav.position[1]) / config.area_height,
        "input_size_norm": _safe_norm(task.input_size_bits, max(config.task_input_size_range_bits[1], 1e-6)),
        "cpu_cycles_norm": _safe_norm(task.cpu_cycles, max(config.task_cpu_cycles_range[1], 1e-6)),
        "remaining_slack_norm": _safe_norm(remaining_slack, max(config.task_slack_range[1], 1e-6)),
        "deadline_margin_norm": _safe_norm(remaining_slack, max(config.task_slack_range[1], 1e-6)),
        "service_type_norm": _safe_norm(float(task.service_type), max(1, config.num_service_types - 1)),
        "required_reliability": float(task.required_reliability),
        "same_user_backlog_norm": _safe_norm(len(pending_by_user.get(task.user_id, [])), max(1, config.num_users * config.task_arrival_max_per_step)),
        "within_coverage_flag": 1.0 if distance <= config.uav_coverage_radius else 0.0,
    }
    payload: dict[str, object] = {
        "slot_index": slot_index,
        "task_id": task.task_id,
        "user_id": int(task.user_id),
        "service_type": int(task.service_type),
        "active": True,
    }
    payload.update({field: float(features[field]) for field in field_names})
    return payload


def _empty_candidate(candidate_id: int) -> dict[str, object]:
    payload: dict[str, object] = {
        "candidate_id": int(candidate_id),
        "target": "invalid",
        "associated_uav_id": None,
        "assigned_uav_id": None,
        "fetch_source": None,
        "energy_ok": False,
        "reliability_ok": False,
        "deadline_ok": False,
        "completed": False,
        "reason": "padding",
    }
    payload.update({field: 0.0 for field in offloading_candidate_feature_fields()})
    return payload


def _candidate_dicts(
    *,
    candidates: list[OffloadingCandidate],
    fixed_count: int,
) -> tuple[list[dict[str, object]], list[float]]:
    payloads: list[dict[str, object]] = []
    mask: list[float] = []
    for candidate in candidates[:fixed_count]:
        payloads.append(candidate.to_observation_dict())
        mask.append(1.0)
    while len(payloads) < fixed_count:
        payloads.append(_empty_candidate(len(payloads)))
        mask.append(0.0)
    return payloads, mask


def _build_cache_candidates(
    *,
    uav: UAVNode,
    all_uavs: list[UAVNode],
    users_by_id: dict[int, UserEquipment],
    pending_tasks: list[Task],
    config: SystemConfig,
    service_catalog: ServiceCatalog,
) -> list[dict[str, object]]:
    field_names = cache_candidate_feature_fields()
    request_count_scale = max(1, *(uav.cache_request_counts.values() or [1]))
    ema_scale = max(1.0e-6, *(uav.cache_ema_scores.values() or [1.0e-6]))
    value_scale = max(1.0e-6, *(uav.cache_value_scores.values() or [1.0e-6]))
    max_fetch_bits = max(service_catalog.service_sizes_bits) if service_catalog.service_sizes_bits else 1
    demand_scale = max(1, config.num_users * config.task_arrival_max_per_step)
    peers = [peer for peer in all_uavs if peer.uav_id != uav.uav_id]
    payload: list[dict[str, object]] = []
    for service_type in range(config.num_service_types):
        global_pending = sum(1 for task in pending_tasks if task.service_type == service_type and task.status not in {"completed", "expired"})
        local_pending = sum(
            1
            for task in pending_tasks
            if task.service_type == service_type
            and task.status not in {"completed", "expired"}
            and math.dist(
                (users_by_id[task.user_id].position[0], users_by_id[task.user_id].position[1]),
                (uav.position[0], uav.position[1]),
            )
            <= config.uav_coverage_radius
        )
        peer_cached = sum(1 for peer in peers if service_type in peer.service_cache)
        item = {
            "service_type": int(service_type),
            "is_cached": bool(service_type in uav.service_cache),
            "cache_priority_allowed": bool(uav.service_cache_capacity > 0),
            "service_type_norm": _safe_norm(float(service_type), max(1, config.num_service_types - 1)),
            "cached_flag": 1.0 if service_type in uav.service_cache else 0.0,
            "cache_value_score_norm": _safe_norm(float(uav.cache_value_scores.get(service_type, 0.0)), value_scale),
            "request_count_norm": _safe_norm(float(uav.cache_request_counts.get(service_type, 0)), request_count_scale),
            "ema_score_norm": _safe_norm(float(uav.cache_ema_scores.get(service_type, 0.0)), ema_scale),
            "local_pending_demand_norm": _safe_norm(float(local_pending), demand_scale),
            "global_pending_demand_norm": _safe_norm(float(global_pending), demand_scale),
            "peer_cache_fraction": _safe_norm(float(peer_cached), max(1, len(peers))),
            "fetch_size_norm": _safe_norm(float(service_catalog.get_fetch_size_bits(service_type)), max_fetch_bits),
        }
        payload.append({field: float(item[field]) if isinstance(item[field], (int, float)) else item[field] for field in field_names} | {key: item[key] for key in ("service_type", "is_cached", "cache_priority_allowed")})
    return payload


def build_structured_observations(
    *,
    uavs: list[UAVNode],
    users: list[UserEquipment],
    pending_tasks: list[Task],
    config: SystemConfig,
    current_time: float = 0.0,
    bs: BaseStation | None = None,
    service_catalog: ServiceCatalog | None = None,
    tdma_queue: TDMAQueue | None = None,
    compute_queue: ComputeQueue | None = None,
) -> list[dict[str, object]]:
    """构造面向联合优化的结构化局部观测。"""
    resolved_bs = bs or BaseStation.from_config(config)
    resolved_catalog = service_catalog or ServiceCatalog.from_config(config)
    resolved_tdma = tdma_queue or TDMAQueue()
    resolved_compute = compute_queue or ComputeQueue()
    users_by_id = {user.user_id: user for user in users}
    pending_by_user: dict[int, list[Task]] = {}
    for task in pending_tasks:
        pending_by_user.setdefault(task.user_id, []).append(task)

    task_slot_count = observation_task_slot_count(config)
    candidate_count = offloading_candidate_count(config)
    structured: list[dict[str, object]] = []
    for uav in uavs:
        uav_state = build_uav_state(uav=uav, all_uavs=uavs, config=config)
        associated_user_state, associated_user_flat = _build_associated_user_state(
            uav=uav,
            users=users,
            pending_tasks=pending_tasks,
            config=config,
            current_time=current_time,
        )
        selected_tasks = _select_task_slots(
            uav=uav,
            users_by_id=users_by_id,
            pending_tasks=pending_tasks,
            config=config,
            current_time=current_time,
        )

        task_slots: list[dict[str, object]] = []
        offload_candidates: list[list[dict[str, object]]] = []
        task_slot_mask: list[float] = []
        offloading_candidate_mask: list[list[float]] = []
        for slot_index in range(task_slot_count):
            task = selected_tasks[slot_index] if slot_index < len(selected_tasks) else None
            slot_payload = _build_task_slot_entry(
                slot_index=slot_index,
                task=task,
                uav=uav,
                users_by_id=users_by_id,
                pending_by_user=pending_by_user,
                config=config,
                current_time=current_time,
            )
            task_slots.append(slot_payload)
            if task is None:
                padded_candidates, candidate_mask = _candidate_dicts(candidates=[], fixed_count=candidate_count)
                offload_candidates.append(padded_candidates)
                offloading_candidate_mask.append(candidate_mask)
                task_slot_mask.append(0.0)
                continue
            user = users_by_id[task.user_id]
            candidates = enumerate_offloading_candidates(
                task=task,
                ue=user,
                associated_uav=uav,
                all_uavs=uavs,
                bs=resolved_bs,
                service_catalog=resolved_catalog,
                config=config,
                current_time=current_time,
                tdma_queue=resolved_tdma,
                compute_queue=resolved_compute,
            )
            padded_candidates, candidate_mask = _candidate_dicts(candidates=candidates, fixed_count=candidate_count)
            offload_candidates.append(padded_candidates)
            offloading_candidate_mask.append(candidate_mask)
            task_slot_mask.append(1.0)

        cache_candidates = _build_cache_candidates(
            uav=uav,
            all_uavs=uavs,
            users_by_id=users_by_id,
            pending_tasks=pending_tasks,
            config=config,
            service_catalog=resolved_catalog,
        )

        own = list(uav_state["own_state"])
        neighbor_flat = [value for block in uav_state["neighbor_uav_states"] for value in block]
        tx_queue = list(uav_state["tx_queue_summary"])
        compute_queue_summary = list(uav_state["compute_queue_summary"])
        backlog_summary = list(uav_state["task_backlog_summary"])
        cache_summary = list(uav_state["local_cache_summary"])
        cache_scores = list(uav_state["cache_score_summary"])
        flat_legacy = own + neighbor_flat + tx_queue + compute_queue_summary + backlog_summary + cache_summary + cache_scores + associated_user_flat

        structured.append(
            {
                "uav_state": uav_state,
                "associated_user_state": associated_user_state,
                "task_slots": task_slots,
                "offload_candidates": offload_candidates,
                "cache_candidates": cache_candidates,
                "action_masks": {
                    "mobility_mask": [1.0, 1.0],
                    "task_slot_mask": task_slot_mask,
                    "offloading_candidate_mask": offloading_candidate_mask,
                    "offloading_defer_mask": list(task_slot_mask),
                    "cache_service_mask": [1.0 if uav.service_cache_capacity > 0 else 0.0 for _ in range(config.num_service_types)],
                },
                "flat_legacy": flat_legacy,
            }
        )
    return structured


def build_observations(
    *,
    uavs: list[UAVNode],
    users: list[UserEquipment],
    pending_tasks: list[Task],
    config: SystemConfig,
    current_time: float = 0.0,
    bs: BaseStation | None = None,
    service_catalog: ServiceCatalog | None = None,
    tdma_queue: TDMAQueue | None = None,
    compute_queue: ComputeQueue | None = None,
    export_mode: Literal["flat_legacy", "structured_joint"] = "flat_legacy",
) -> list[list[float]] | list[dict[str, object]]:
    """构造每架 UAV 的局部观测，默认导出 legacy flat 版本。"""
    structured = build_structured_observations(
        uavs=uavs,
        users=users,
        pending_tasks=pending_tasks,
        config=config,
        current_time=current_time,
        bs=bs,
        service_catalog=service_catalog,
        tdma_queue=tdma_queue,
        compute_queue=compute_queue,
    )
    if export_mode == "structured_joint":
        return structured
    if export_mode == "flat_legacy":
        return [list(observation["flat_legacy"]) for observation in structured]
    raise ValueError(f"Unsupported observation export_mode: {export_mode}")
