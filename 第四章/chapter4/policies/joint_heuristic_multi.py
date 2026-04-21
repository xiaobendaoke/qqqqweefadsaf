"""第四章 joint heuristic 多 UAV 联合策略模块。

该策略直接读取结构化 joint observation，
同时生成 mobility / offloading / caching 三部分动作，
作为与 joint RL policy 公平对比的新启发式 baseline。
"""

from __future__ import annotations

from typing import Any

from common.uav_mec.core.observation import build_observations

from .mobility_heuristic_multi import select_actions as select_legacy_mobility


POLICY_ID = "joint_heuristic"
POLICY_LABEL = "Joint heuristic baseline"
DEFAULT_DEFER_PLAN_ID = -1


def _structured_observations(env: Any) -> list[dict[str, object]]:
    current_time = float(env.current_step) * float(env.config.time_slot_duration)
    structured = build_observations(
        uavs=env.uavs,
        users=env.users,
        pending_tasks=env.pending_tasks,
        config=env.config,
        current_time=current_time,
        bs=env.bs,
        service_catalog=env.service_catalog,
        tdma_queue=env.tdma_queue,
        compute_queue=env.compute_queue,
        export_mode="structured_joint",
    )
    assert isinstance(structured, list)
    return structured


def _candidate_score(candidate: dict[str, object]) -> float:
    feasible_flag = float(candidate.get("feasible_flag", 0.0))
    cache_hit = float(candidate.get("cache_hit_flag", 0.0))
    success_prob = float(candidate.get("success_prob_est", 0.0))
    total_latency = float(candidate.get("total_latency_est", 0.0))
    energy_est = float(candidate.get("energy_est", 0.0))
    deadline_margin = float(candidate.get("deadline_margin", 0.0))
    reliability_bonus = 0.8 * success_prob
    feasibility_bonus = 1.2 if feasible_flag > 0.5 else -0.6
    cache_bonus = 0.35 * cache_hit
    latency_penalty = 0.55 * total_latency
    energy_penalty = 0.15 * energy_est
    deadline_bonus = 0.25 * deadline_margin
    return feasibility_bonus + reliability_bonus + cache_bonus + deadline_bonus - latency_penalty - energy_penalty


def _select_offloading_plan_ids(structured: dict[str, object], *, claimed_task_ids: set[str]) -> list[int]:
    task_slot_mask = list(structured["action_masks"]["task_slot_mask"])
    candidate_mask = list(structured["action_masks"]["offloading_candidate_mask"])
    task_slots = list(structured["task_slots"])
    offload_candidates = list(structured["offload_candidates"])

    selected: list[int] = []
    for slot_index, slot_active in enumerate(task_slot_mask):
        if slot_index >= len(task_slots) or slot_index >= len(offload_candidates):
            selected.append(DEFAULT_DEFER_PLAN_ID)
            continue
        slot = task_slots[slot_index]
        task_id = slot.get("task_id")
        if float(slot_active) <= 0.0 or not task_id or str(task_id) in claimed_task_ids:
            selected.append(DEFAULT_DEFER_PLAN_ID)
            continue
        best_plan_id = DEFAULT_DEFER_PLAN_ID
        best_score = float("-inf")
        for candidate_index, candidate in enumerate(offload_candidates[slot_index]):
            if candidate_index >= len(candidate_mask[slot_index]) or float(candidate_mask[slot_index][candidate_index]) <= 0.0:
                continue
            if str(candidate.get("target", "invalid")) == "invalid":
                continue
            score = _candidate_score(candidate)
            if score > best_score:
                best_score = score
                best_plan_id = int(candidate.get("candidate_id", DEFAULT_DEFER_PLAN_ID))
        if best_plan_id != DEFAULT_DEFER_PLAN_ID:
            claimed_task_ids.add(str(task_id))
        selected.append(best_plan_id)
    return selected


def _cache_priority_scores(structured: dict[str, object]) -> list[float]:
    cache_candidates = list(structured["cache_candidates"])
    cache_mask = list(structured["action_masks"]["cache_service_mask"])
    priorities: list[float] = []
    for service_index, item in enumerate(cache_candidates):
        if service_index >= len(cache_mask) or float(cache_mask[service_index]) <= 0.0:
            priorities.append(0.0)
            continue
        score = (
            0.65 * float(item.get("cache_value_score_norm", 0.0))
            + 0.55 * float(item.get("local_pending_demand_norm", 0.0))
            + 0.35 * float(item.get("global_pending_demand_norm", 0.0))
            + 0.30 * float(item.get("request_count_norm", 0.0))
            + 0.20 * float(item.get("ema_score_norm", 0.0))
            - 0.20 * float(item.get("peer_cache_fraction", 0.0))
            - 0.10 * float(item.get("cached_flag", 0.0))
        )
        priorities.append(float(score))
    return priorities


def select_actions(observations: list[list[float]], env=None) -> list[dict[str, object]]:
    if env is None:
        raise ValueError("joint_heuristic_multi.select_actions requires env to build structured observations.")

    mobility_actions = select_legacy_mobility(observations, env)
    structured_observations = _structured_observations(env)
    claimed_task_ids: set[str] = set()
    actions: list[dict[str, object]] = []
    for agent_index, structured in enumerate(structured_observations):
        mobility = mobility_actions[agent_index] if agent_index < len(mobility_actions) else [0.0, 0.0]
        actions.append(
            {
                "mobility": {"dx": float(mobility[0]), "dy": float(mobility[1])},
                "offloading": {"task_slot_plan_ids": _select_offloading_plan_ids(structured, claimed_task_ids=claimed_task_ids)},
                "caching": {"service_priorities": _cache_priority_scores(structured)},
            }
        )
    return actions
