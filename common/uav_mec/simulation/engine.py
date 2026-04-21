"""单时隙仿真引擎模块。

该模块同时维护两条执行路径：
1. `legacy_heuristic`：保留旧的自动关联/卸载/缓存决策链路，供基线与旧训练流程使用。
2. `joint_action`：把环境作为动作条件仿真器，只执行策略选中的 mobility/offloading/caching 动作。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Literal

from ..config import SystemConfig
from ..core.action import JointAgentAction, LEGACY_ACTION_MODE, scale_action
from ..core.observation import build_structured_observations
from ..entities import BaseStation, ServiceCatalog, UAVNode, UserEquipment
from ..metrics import MetricTracker
from ..scheduler import ComputeQueue, TDMAQueue
from ..scheduler.assignment import assign_uav
from ..scheduler.offloading import OffloadingCandidate, OffloadingDecision, commit_offloading_decision, decide_offloading, enumerate_offloading_candidates, select_candidate_by_plan_id
from ..scheduler.service_cache import apply_cache_action, apply_service_cache_policy, record_service_request, refresh_service_cache
from ..scheduler.tdma import EDGE_ACCESS_QUEUE_ID
from ..task import Task
from .task_generator import generate_tasks


@dataclass(slots=True)
class StepExecution:
    """汇总单个时隙执行结果，供环境层更新状态和日志。"""

    generated_tasks: list[Task]
    finalized_tasks: list[Task]
    pending_tasks: list[Task]
    decisions: list[OffloadingDecision]
    energy_breakdown: dict[str, float]
    cache_events: list[dict[str, object]]
    task_lifecycle_counts: dict[str, int]
    scheduler_mode: str
    action_feedback: dict[str, int]


def _empty_energy_breakdown() -> dict[str, float]:
    return {
        "uav_move_energy": 0.0,
        "uav_compute_energy": 0.0,
        "ue_local_energy": 0.0,
        "ue_uplink_energy": 0.0,
        "bs_compute_energy": 0.0,
        "relay_fetch_energy": 0.0,
        "bs_fetch_tx_energy": 0.0,
    }


def _serialize_cache_events(events: list[object]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for event in events:
        payload.append(
            {
                "uav_id": event.uav_id,
                "action": event.action,
                "service_type": event.service_type,
                "value_score": event.value_score,
                "reason": event.reason,
            }
        )
    return payload


def _finalize_remaining_tasks_at_episode_end(
    *,
    pending_tasks: list[Task],
    expired_tasks: list[Task],
    episode_end_time: float,
) -> list[Task]:
    """在 episode 结束时强制清算剩余任务，避免终局 pending 漏统。"""
    forced_finalized: list[Task] = []
    for task in list(pending_tasks):
        pending_tasks.remove(task)
        task.status = "expired"
        task.completed = False
        if task.total_latency <= 0.0:
            task.total_latency = max(0.0, episode_end_time - task.arrival_time)
        forced_finalized.append(task)
        expired_tasks.append(task)
    return forced_finalized


def _refresh_runtime_counters(
    *,
    config: SystemConfig,
    users: list[UserEquipment],
    uavs: list[UAVNode],
    tdma_queue: TDMAQueue,
    compute_queue: ComputeQueue,
    pending_tasks: list[Task],
    current_time: float,
    accumulate_delay: bool = False,
) -> None:
    """刷新 UAV 的瞬时队列/覆盖/积压计数，供观测与日志共享。"""
    for uav in uavs:
        uav.current_tx_queue_length = tdma_queue.get_queue_length(queue_id=EDGE_ACCESS_QUEUE_ID, current_time=current_time)
        uav.current_tx_queue_delay = tdma_queue.estimate_wait(current_time, queue_id=EDGE_ACCESS_QUEUE_ID)
        uav.current_compute_queue_length = compute_queue.get_queue_length(queue_id=f"uav:{uav.uav_id}", current_time=current_time)
        uav.current_compute_queue_delay = compute_queue.estimate_wait(current_time, queue_id=f"uav:{uav.uav_id}")
        uav.current_backlog_load = sum(
            1
            for task in pending_tasks
            if task.status in {"pending", "scheduled"}
            and task.assigned_uav_id == uav.uav_id
        )
        uav.current_coverage_load = sum(
            1
            for user in users
            if math.dist((user.position[0], user.position[1]), (uav.position[0], uav.position[1])) <= config.uav_coverage_radius
        )
        uav.max_tx_queue_length = max(uav.max_tx_queue_length, uav.current_tx_queue_length)
        uav.max_compute_queue_length = max(uav.max_compute_queue_length, uav.current_compute_queue_length)
        if accumulate_delay:
            uav.cumulative_tx_queue_delay += uav.current_tx_queue_delay
            uav.cumulative_compute_queue_delay += uav.current_compute_queue_delay


def _resolve_scheduler_mode(
    *,
    actions: list[JointAgentAction],
    scheduler_mode: str,
) -> Literal["legacy_heuristic", "joint_action"]:
    if scheduler_mode == "legacy_heuristic":
        return "legacy_heuristic"
    if scheduler_mode == "joint_action":
        return "joint_action"
    if scheduler_mode != "auto":
        raise ValueError(f"Unsupported scheduler_mode: {scheduler_mode}")
    return "legacy_heuristic" if all(action.action_mode == LEGACY_ACTION_MODE for action in actions) else "joint_action"


def _decision_energy_available_now(
    *,
    decision: OffloadingDecision,
    ue: UserEquipment,
    uav_lookup: dict[int, UAVNode],
) -> bool:
    if ue.remaining_energy_j + 1.0e-9 < (decision.ue_local_energy + decision.ue_uplink_energy):
        return False
    if decision.assigned_uav_id is not None:
        executed_uav = uav_lookup[decision.assigned_uav_id]
        if executed_uav.remaining_energy_j + 1.0e-9 < decision.uav_compute_energy:
            return False
    for source_uav_id, relay_energy in (decision.uav_tx_energy_by_id or {}).items():
        if source_uav_id not in uav_lookup:
            return False
        if uav_lookup[source_uav_id].remaining_energy_j + 1.0e-9 < float(relay_energy):
            return False
    return True


def _apply_decision_result(
    *,
    task: Task,
    decision: OffloadingDecision,
    current_time: float,
    users: list[UserEquipment],
    uav_lookup: dict[int, UAVNode],
    bs: BaseStation,
    service_catalog: ServiceCatalog,
    config: SystemConfig,
    metrics: MetricTracker,
    energy_breakdown: dict[str, float],
    cache_events: list[dict[str, object]],
    use_legacy_cache_policy: bool,
) -> None:
    task.mark_result(
        execution_target=decision.target,
        cache_hit=decision.cache_hit,
        fetch_source=decision.fetch_source,
        fetch_wait_delay=decision.fetch_wait_delay,
        fetch_delay=decision.fetch_delay,
        queue_delay=decision.queue_delay,
        transmission_delay=decision.transmission_delay,
        compute_wait_delay=decision.compute_wait_delay,
        compute_delay=decision.compute_delay,
        total_latency=decision.total_latency,
        success_probability=decision.success_probability,
        completed=decision.completed,
        status="scheduled",
        scheduled_time=current_time,
        completion_time=decision.completion_time,
        associated_uav_id=decision.associated_uav_id,
        assigned_uav_id=decision.assigned_uav_id,
        ue_local_energy=decision.ue_local_energy,
        ue_uplink_energy=decision.ue_uplink_energy,
        uav_compute_energy=decision.uav_compute_energy,
        bs_compute_energy=decision.bs_compute_energy,
        relay_fetch_energy=decision.relay_fetch_energy,
        bs_fetch_tx_energy=decision.bs_fetch_tx_energy,
    )

    if decision.energy_ok and decision.target in {"uav", "collaborator"} and decision.assigned_uav_id is not None:
        scheduled_uav = uav_lookup[decision.assigned_uav_id]
        scheduled_uav.assigned_task_count_step += 1
        scheduled_uav.total_assigned_task_count += 1
    if decision.energy_ok:
        metrics.record_assignment(task)

    if not decision.energy_ok:
        return

    energy_breakdown["ue_local_energy"] += decision.ue_local_energy
    energy_breakdown["ue_uplink_energy"] += decision.ue_uplink_energy
    energy_breakdown["uav_compute_energy"] += decision.uav_compute_energy
    energy_breakdown["bs_compute_energy"] += decision.bs_compute_energy
    energy_breakdown["relay_fetch_energy"] += decision.relay_fetch_energy
    energy_breakdown["bs_fetch_tx_energy"] += decision.bs_fetch_tx_energy

    ue = users[task.user_id]
    ue.spend_local_compute_energy(decision.ue_local_energy)
    ue.spend_uplink_energy(decision.ue_uplink_energy)
    if decision.assigned_uav_id is not None:
        executed_uav = uav_lookup[decision.assigned_uav_id]
        executed_uav.remaining_energy_j = max(0.0, executed_uav.remaining_energy_j - decision.uav_compute_energy)
        for source_uav_id, relay_energy in (decision.uav_tx_energy_by_id or {}).items():
            if source_uav_id in uav_lookup:
                source_uav = uav_lookup[source_uav_id]
                source_uav.remaining_energy_j = max(0.0, source_uav.remaining_energy_j - float(relay_energy))
        if use_legacy_cache_policy:
            if decision.cache_hit:
                record_service_request(executed_uav, task.service_type, config=config, service_catalog=service_catalog)
            else:
                cache_events.extend(
                    _serialize_cache_events(
                        apply_service_cache_policy(
                            executed_uav,
                            task.service_type,
                            config=config,
                            service_catalog=service_catalog,
                            opportunistic=True,
                        )
                    )
                )
        else:
            record_service_request(executed_uav, task.service_type, config=config, service_catalog=service_catalog)

    bs.cumulative_compute_energy_j += decision.bs_compute_energy
    bs.cumulative_fetch_tx_energy_j += decision.bs_fetch_tx_energy


def _finalize_tasks_for_step(
    *,
    config: SystemConfig,
    users: list[UserEquipment],
    uavs: list[UAVNode],
    pending_tasks: list[Task],
    completed_tasks: list[Task],
    expired_tasks: list[Task],
    current_step: int,
    next_time: float,
) -> list[Task]:
    uav_lookup = {uav.uav_id: uav for uav in uavs}
    finalized_tasks: list[Task] = []
    for task in list(pending_tasks):
        finish_time = task.completion_time if task.completion_time is not None else float("inf")
        should_finalize = finish_time <= next_time or task.deadline <= next_time
        if not should_finalize:
            continue
        pending_tasks.remove(task)
        if task.completed and finish_time <= task.deadline and finish_time <= next_time:
            task.status = "completed"
            completed_tasks.append(task)
            users[task.user_id].completed_tasks += 1
            if task.assigned_uav_id is not None:
                served_uav = uav_lookup[task.assigned_uav_id]
                served_uav.served_task_count += 1
                served_uav.completed_task_count_step += 1
                served_uav.total_completed_task_count += 1
        else:
            task.status = "expired"
            if task.total_latency <= 0.0:
                task.total_latency = max(0.0, next_time - task.arrival_time)
            expired_tasks.append(task)
        finalized_tasks.append(task)

    if current_step + 1 >= config.steps_per_episode and pending_tasks:
        finalized_tasks.extend(
            _finalize_remaining_tasks_at_episode_end(
                pending_tasks=pending_tasks,
                expired_tasks=expired_tasks,
                episode_end_time=next_time,
            )
        )
    return finalized_tasks


def _build_joint_runtime_candidates(
    *,
    structured_observations: list[dict[str, object]],
    pending_tasks: list[Task],
    users: list[UserEquipment],
    uavs: list[UAVNode],
    bs: BaseStation,
    service_catalog: ServiceCatalog,
    config: SystemConfig,
    current_time: float,
    tdma_queue: TDMAQueue,
    compute_queue: ComputeQueue,
) -> dict[tuple[int, int], list[OffloadingCandidate]]:
    task_lookup = {task.task_id: task for task in pending_tasks if task.status == "pending"}
    runtime_candidates: dict[tuple[int, int], list[OffloadingCandidate]] = {}
    for uav_index, observation in enumerate(structured_observations):
        for slot_index, slot in enumerate(observation["task_slots"]):
            if not bool(slot["active"]):
                continue
            task = task_lookup.get(str(slot["task_id"]))
            if task is None:
                continue
            runtime_candidates[(uav_index, slot_index)] = enumerate_offloading_candidates(
                task=task,
                ue=users[task.user_id],
                associated_uav=uavs[uav_index],
                all_uavs=uavs,
                bs=bs,
                service_catalog=service_catalog,
                config=config,
                current_time=current_time,
                tdma_queue=tdma_queue,
                compute_queue=compute_queue,
            )
    return runtime_candidates


def _select_joint_proposals(
    *,
    actions: list[JointAgentAction],
    structured_observations: list[dict[str, object]],
    runtime_candidates: dict[tuple[int, int], list[OffloadingCandidate]],
    pending_tasks: list[Task],
) -> tuple[list[tuple[Task, OffloadingCandidate, int, int]], dict[str, int]]:
    task_lookup = {task.task_id: task for task in pending_tasks if task.status == "pending"}
    feedback = {
        "proposed_plan_count": 0,
        "explicit_defer_count": 0,
        "masked_slot_reject_count": 0,
        "invalid_plan_reject_count": 0,
        "infeasible_plan_reject_count": 0,
        "duplicate_conflict_count": 0,
        "energy_reject_count": 0,
        "executed_plan_count": 0,
        "cache_event_count": 0,
    }
    proposals_by_task: dict[str, list[tuple[Task, OffloadingCandidate, int, int]]] = {}

    for uav_index, action in enumerate(actions):
        slot_plan_ids = list(action.offloading.task_slot_plan_ids)
        observation = structured_observations[uav_index]
        slot_mask = list(observation["action_masks"]["task_slot_mask"])
        candidate_masks = list(observation["action_masks"]["offloading_candidate_mask"])
        task_slots = list(observation["task_slots"])
        for slot_index, plan_id in enumerate(slot_plan_ids):
            if slot_index >= len(task_slots):
                if plan_id >= 0:
                    feedback["invalid_plan_reject_count"] += 1
                else:
                    feedback["explicit_defer_count"] += 1
                continue
            if plan_id < 0:
                feedback["explicit_defer_count"] += 1
                continue
            if slot_index >= len(slot_mask) or float(slot_mask[slot_index]) <= 0.0:
                feedback["masked_slot_reject_count"] += 1
                continue
            task = task_lookup.get(str(task_slots[slot_index]["task_id"]))
            if task is None:
                feedback["masked_slot_reject_count"] += 1
                continue
            row_mask = candidate_masks[slot_index] if slot_index < len(candidate_masks) else []
            if plan_id >= len(row_mask) or float(row_mask[plan_id]) <= 0.0:
                feedback["invalid_plan_reject_count"] += 1
                continue
            candidate = select_candidate_by_plan_id(runtime_candidates.get((uav_index, slot_index), []), plan_id)
            if candidate is None:
                feedback["invalid_plan_reject_count"] += 1
                continue
            if float(candidate.feasible_flag) <= 0.0:
                feedback["infeasible_plan_reject_count"] += 1
                continue
            feedback["proposed_plan_count"] += 1
            proposals_by_task.setdefault(task.task_id, []).append((task, candidate, uav_index, slot_index))

    selected: list[tuple[Task, OffloadingCandidate, int, int]] = []
    for task_id, proposals in proposals_by_task.items():
        proposals.sort(key=lambda item: (item[1].total_latency_est, item[2], item[3]))
        selected.append(proposals[0])
        feedback["duplicate_conflict_count"] += max(0, len(proposals) - 1)
    selected.sort(key=lambda item: (item[0].deadline, item[1].total_latency_est, item[2], item[3]))
    return selected, feedback


def _run_legacy_scheduler(
    *,
    config: SystemConfig,
    users: list[UserEquipment],
    uavs: list[UAVNode],
    bs: BaseStation,
    service_catalog: ServiceCatalog,
    tdma_queue: TDMAQueue,
    compute_queue: ComputeQueue,
    metrics: MetricTracker,
    current_step: int,
    pending_tasks: list[Task],
    completed_tasks: list[Task],
    expired_tasks: list[Task],
    actions: list[JointAgentAction],
    rng: random.Random,
) -> StepExecution:
    current_time = current_step * config.time_slot_duration
    next_time = (current_step + 1) * config.time_slot_duration

    for user in users:
        user.move(config, rng)
    for uav in uavs:
        uav.reset_step_counters()

    energy_breakdown = _empty_energy_breakdown()
    for index, uav in enumerate(uavs):
        energy_breakdown["uav_move_energy"] += uav.move(scale_action(actions[index], config), config)

    new_tasks = generate_tasks(
        users=users,
        current_time=current_time,
        step_index=current_step,
        config=config,
        service_catalog=service_catalog,
        rng=rng,
    )
    pending_tasks.extend(new_tasks)

    _refresh_runtime_counters(
        config=config,
        users=users,
        uavs=uavs,
        tdma_queue=tdma_queue,
        compute_queue=compute_queue,
        pending_tasks=pending_tasks,
        current_time=current_time,
    )

    decisions: list[OffloadingDecision] = []
    cache_events: list[dict[str, object]] = []
    uav_lookup = {uav.uav_id: uav for uav in uavs}
    feedback = {
        "proposed_plan_count": 0,
        "explicit_defer_count": 0,
        "masked_slot_reject_count": 0,
        "invalid_plan_reject_count": 0,
        "infeasible_plan_reject_count": 0,
        "duplicate_conflict_count": 0,
        "energy_reject_count": 0,
        "executed_plan_count": 0,
        "cache_event_count": 0,
    }

    for task in list(pending_tasks):
        if task.status != "pending":
            continue
        associated_uav = assign_uav(
            uavs=uavs,
            ue=users[task.user_id],
            tdma_queue=tdma_queue,
            compute_queue=compute_queue,
            current_time=current_time,
            coverage_radius=config.uav_coverage_radius,
            rule=config.assignment_rule,
        )
        decision = decide_offloading(
            task=task,
            ue=users[task.user_id],
            associated_uav=associated_uav,
            all_uavs=uavs,
            bs=bs,
            service_catalog=service_catalog,
            config=config,
            current_time=current_time,
            tdma_queue=tdma_queue,
            compute_queue=compute_queue,
        )
        _apply_decision_result(
            task=task,
            decision=decision,
            current_time=current_time,
            users=users,
            uav_lookup=uav_lookup,
            bs=bs,
            service_catalog=service_catalog,
            config=config,
            metrics=metrics,
            energy_breakdown=energy_breakdown,
            cache_events=cache_events,
            use_legacy_cache_policy=True,
        )
        decisions.append(decision)
        feedback["executed_plan_count"] += 1

    if config.cache_refresh_interval_steps > 0 and (current_step + 1) % config.cache_refresh_interval_steps == 0:
        for uav in uavs:
            refreshed = refresh_service_cache(uav, config=config, service_catalog=service_catalog)
            serialized = _serialize_cache_events(refreshed)
            cache_events.extend(serialized)
            feedback["cache_event_count"] += len(serialized)

    finalized_tasks = _finalize_tasks_for_step(
        config=config,
        users=users,
        uavs=uavs,
        pending_tasks=pending_tasks,
        completed_tasks=completed_tasks,
        expired_tasks=expired_tasks,
        current_step=current_step,
        next_time=next_time,
    )
    _refresh_runtime_counters(
        config=config,
        users=users,
        uavs=uavs,
        tdma_queue=tdma_queue,
        compute_queue=compute_queue,
        pending_tasks=pending_tasks,
        current_time=next_time,
        accumulate_delay=True,
    )
    metrics.record_step(
        generated_tasks=new_tasks,
        finalized_tasks=finalized_tasks,
        pending_tasks=pending_tasks,
        energy_breakdown=energy_breakdown,
    )
    return StepExecution(
        generated_tasks=new_tasks,
        finalized_tasks=finalized_tasks,
        pending_tasks=list(pending_tasks),
        decisions=decisions,
        energy_breakdown=energy_breakdown,
        cache_events=cache_events,
        task_lifecycle_counts={
            "new_tasks": len(new_tasks),
            "pending_tasks": len(pending_tasks),
            "completed_tasks": sum(1 for task in finalized_tasks if task.status == "completed"),
            "expired_tasks": sum(1 for task in finalized_tasks if task.status == "expired"),
        },
        scheduler_mode="legacy_heuristic",
        action_feedback=feedback,
    )


def _run_joint_action_scheduler(
    *,
    config: SystemConfig,
    users: list[UserEquipment],
    uavs: list[UAVNode],
    bs: BaseStation,
    service_catalog: ServiceCatalog,
    tdma_queue: TDMAQueue,
    compute_queue: ComputeQueue,
    metrics: MetricTracker,
    current_step: int,
    pending_tasks: list[Task],
    completed_tasks: list[Task],
    expired_tasks: list[Task],
    actions: list[JointAgentAction],
    rng: random.Random,
) -> StepExecution:
    current_time = current_step * config.time_slot_duration
    next_time = (current_step + 1) * config.time_slot_duration

    for uav in uavs:
        uav.reset_step_counters()
    energy_breakdown = _empty_energy_breakdown()

    _refresh_runtime_counters(
        config=config,
        users=users,
        uavs=uavs,
        tdma_queue=tdma_queue,
        compute_queue=compute_queue,
        pending_tasks=pending_tasks,
        current_time=current_time,
    )
    structured_observations = build_structured_observations(
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
    runtime_candidates = _build_joint_runtime_candidates(
        structured_observations=structured_observations,
        pending_tasks=pending_tasks,
        users=users,
        uavs=uavs,
        bs=bs,
        service_catalog=service_catalog,
        config=config,
        current_time=current_time,
        tdma_queue=tdma_queue,
        compute_queue=compute_queue,
    )
    selected_proposals, feedback = _select_joint_proposals(
        actions=actions,
        structured_observations=structured_observations,
        runtime_candidates=runtime_candidates,
        pending_tasks=pending_tasks,
    )

    decisions: list[OffloadingDecision] = []
    cache_events: list[dict[str, object]] = []
    uav_lookup = {uav.uav_id: uav for uav in uavs}
    for task, candidate, _, _ in selected_proposals:
        if task.status != "pending":
            feedback["duplicate_conflict_count"] += 1
            continue
        if not _decision_energy_available_now(decision=candidate.decision, ue=users[task.user_id], uav_lookup=uav_lookup):
            feedback["energy_reject_count"] += 1
            continue
        actual_decision = commit_offloading_decision(
            decision=candidate.decision,
            task=task,
            current_time=current_time,
            tdma_queue=tdma_queue,
            compute_queue=compute_queue,
        )
        _apply_decision_result(
            task=task,
            decision=actual_decision,
            current_time=current_time,
            users=users,
            uav_lookup=uav_lookup,
            bs=bs,
            service_catalog=service_catalog,
            config=config,
            metrics=metrics,
            energy_breakdown=energy_breakdown,
            cache_events=cache_events,
            use_legacy_cache_policy=False,
        )
        decisions.append(actual_decision)
        feedback["executed_plan_count"] += 1

    for uav_index, uav in enumerate(uavs):
        serialized = _serialize_cache_events(
            apply_cache_action(
                uav,
                actions[uav_index].caching.service_priorities,
                config=config,
                service_catalog=service_catalog,
            )
        )
        cache_events.extend(serialized)
        feedback["cache_event_count"] += len(serialized)

    for user in users:
        user.move(config, rng)
    for index, uav in enumerate(uavs):
        energy_breakdown["uav_move_energy"] += uav.move(scale_action(actions[index], config), config)

    new_tasks = generate_tasks(
        users=users,
        current_time=next_time,
        step_index=current_step,
        config=config,
        service_catalog=service_catalog,
        rng=rng,
    )
    pending_tasks.extend(new_tasks)

    finalized_tasks = _finalize_tasks_for_step(
        config=config,
        users=users,
        uavs=uavs,
        pending_tasks=pending_tasks,
        completed_tasks=completed_tasks,
        expired_tasks=expired_tasks,
        current_step=current_step,
        next_time=next_time,
    )
    _refresh_runtime_counters(
        config=config,
        users=users,
        uavs=uavs,
        tdma_queue=tdma_queue,
        compute_queue=compute_queue,
        pending_tasks=pending_tasks,
        current_time=next_time,
        accumulate_delay=True,
    )
    metrics.record_step(
        generated_tasks=new_tasks,
        finalized_tasks=finalized_tasks,
        pending_tasks=pending_tasks,
        energy_breakdown=energy_breakdown,
    )
    return StepExecution(
        generated_tasks=new_tasks,
        finalized_tasks=finalized_tasks,
        pending_tasks=list(pending_tasks),
        decisions=decisions,
        energy_breakdown=energy_breakdown,
        cache_events=cache_events,
        task_lifecycle_counts={
            "new_tasks": len(new_tasks),
            "pending_tasks": len(pending_tasks),
            "completed_tasks": sum(1 for task in finalized_tasks if task.status == "completed"),
            "expired_tasks": sum(1 for task in finalized_tasks if task.status == "expired"),
        },
        scheduler_mode="joint_action",
        action_feedback=feedback,
    )


def run_step(
    *,
    config: SystemConfig,
    users: list[UserEquipment],
    uavs: list[UAVNode],
    bs: BaseStation,
    service_catalog: ServiceCatalog,
    tdma_queue: TDMAQueue,
    compute_queue: ComputeQueue,
    metrics: MetricTracker,
    current_step: int,
    pending_tasks: list[Task],
    completed_tasks: list[Task],
    expired_tasks: list[Task],
    actions: list[JointAgentAction],
    rng: random.Random,
    scheduler_mode: str = "auto",
) -> StepExecution:
    """执行一个完整时隙；根据模式走 legacy 或 joint-action 主路径。"""
    resolved_mode = _resolve_scheduler_mode(actions=actions, scheduler_mode=scheduler_mode)
    if resolved_mode == "joint_action":
        return _run_joint_action_scheduler(
            config=config,
            users=users,
            uavs=uavs,
            bs=bs,
            service_catalog=service_catalog,
            tdma_queue=tdma_queue,
            compute_queue=compute_queue,
            metrics=metrics,
            current_step=current_step,
            pending_tasks=pending_tasks,
            completed_tasks=completed_tasks,
            expired_tasks=expired_tasks,
            actions=actions,
            rng=rng,
        )
    return _run_legacy_scheduler(
        config=config,
        users=users,
        uavs=uavs,
        bs=bs,
        service_catalog=service_catalog,
        tdma_queue=tdma_queue,
        compute_queue=compute_queue,
        metrics=metrics,
        current_step=current_step,
        pending_tasks=pending_tasks,
        completed_tasks=completed_tasks,
        expired_tasks=expired_tasks,
        actions=actions,
        rng=rng,
    )
