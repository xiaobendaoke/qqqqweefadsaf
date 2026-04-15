from __future__ import annotations

from dataclasses import dataclass
import math
import random

from ..config import SystemConfig
from ..core.action import scale_action
from ..entities import BaseStation, ServiceCatalog, UAVNode, UserEquipment
from ..metrics import MetricTracker
from ..scheduler import ComputeQueue, TDMAQueue
from ..scheduler.assignment import assign_uav
from ..scheduler.offloading import OffloadingDecision, decide_offloading
from ..scheduler.service_cache import apply_service_cache_policy, record_service_request, refresh_service_cache
from ..task import Task
from .task_generator import generate_tasks


@dataclass(slots=True)
class StepExecution:
    generated_tasks: list[Task]
    finalized_tasks: list[Task]
    pending_tasks: list[Task]
    decisions: list[OffloadingDecision]
    energy_breakdown: dict[str, float]
    cache_events: list[dict[str, object]]
    task_lifecycle_counts: dict[str, int]


def _finalize_remaining_tasks_at_episode_end(
    *,
    pending_tasks: list[Task],
    expired_tasks: list[Task],
    episode_end_time: float,
) -> list[Task]:
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
    for uav in uavs:
        uav.current_tx_queue_length = tdma_queue.get_queue_length(queue_id=f"uav:{uav.uav_id}", current_time=current_time)
        uav.current_tx_queue_delay = tdma_queue.estimate_wait(current_time, queue_id=f"uav:{uav.uav_id}")
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
    actions: list[list[float]],
    rng: random.Random,
) -> StepExecution:
    current_time = current_step * config.time_slot_duration
    next_time = (current_step + 1) * config.time_slot_duration
    for user in users:
        user.move(config, rng)

    for uav in uavs:
        uav.reset_step_counters()

    energy_breakdown = {
        "uav_move_energy": 0.0,
        "uav_compute_energy": 0.0,
        "ue_local_energy": 0.0,
        "ue_uplink_energy": 0.0,
        "bs_compute_energy": 0.0,
        "relay_fetch_energy": 0.0,
    }

    for idx, uav in enumerate(uavs):
        energy_breakdown["uav_move_energy"] += uav.move(scale_action(actions[idx], config), config)

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

    for task in list(pending_tasks):
        if task.status != "pending":
            continue
        ue = users[task.user_id]
        associated_uav = assign_uav(
            uavs=uavs,
            ue=ue,
            tdma_queue=tdma_queue,
            current_time=current_time,
            coverage_radius=config.uav_coverage_radius,
            rule=config.assignment_rule,
        )
        decision = decide_offloading(
            task=task,
            ue=ue,
            associated_uav=associated_uav,
            all_uavs=uavs,
            bs=bs,
            service_catalog=service_catalog,
            config=config,
            current_time=current_time,
            tdma_queue=tdma_queue,
            compute_queue=compute_queue,
        )
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
        )
        if decision.target in {"uav", "collaborator"} and decision.assigned_uav_id is not None:
            scheduled_uav = uav_lookup[decision.assigned_uav_id]
            scheduled_uav.assigned_task_count_step += 1
            scheduled_uav.total_assigned_task_count += 1
        metrics.record_assignment(task)
        decisions.append(decision)

        energy_breakdown["ue_local_energy"] += decision.ue_local_energy
        energy_breakdown["ue_uplink_energy"] += decision.ue_uplink_energy
        energy_breakdown["uav_compute_energy"] += decision.uav_compute_energy
        energy_breakdown["bs_compute_energy"] += decision.bs_compute_energy
        energy_breakdown["relay_fetch_energy"] += decision.relay_fetch_energy

        ue.spend_local_compute_energy(decision.ue_local_energy)
        ue.spend_uplink_energy(decision.ue_uplink_energy)
        if decision.assigned_uav_id is not None:
            executed_uav = uav_lookup[decision.assigned_uav_id]
            executed_uav.remaining_energy_j = max(0.0, executed_uav.remaining_energy_j - decision.uav_compute_energy)
            for source_uav_id, relay_energy in (decision.uav_tx_energy_by_id or {}).items():
                if source_uav_id in uav_lookup:
                    source_uav = uav_lookup[source_uav_id]
                    source_uav.remaining_energy_j = max(0.0, source_uav.remaining_energy_j - float(relay_energy))
            if decision.cache_hit:
                record_service_request(executed_uav, task.service_type, config=config, service_catalog=service_catalog)
            else:
                applied_events = apply_service_cache_policy(
                    executed_uav,
                    task.service_type,
                    config=config,
                    service_catalog=service_catalog,
                    opportunistic=True,
                )
                cache_events.extend(
                    {
                        "uav_id": event.uav_id,
                        "action": event.action,
                        "service_type": event.service_type,
                        "value_score": event.value_score,
                        "reason": event.reason,
                    }
                    for event in applied_events
                )
        bs.cumulative_compute_energy_j += decision.bs_compute_energy

    if config.cache_refresh_interval_steps > 0 and (current_step + 1) % config.cache_refresh_interval_steps == 0:
        for uav in uavs:
            refreshed = refresh_service_cache(uav, config=config, service_catalog=service_catalog)
            cache_events.extend(
                {
                    "uav_id": event.uav_id,
                    "action": event.action,
                    "service_type": event.service_type,
                    "value_score": event.value_score,
                    "reason": event.reason,
                }
                for event in refreshed
            )

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
    )
