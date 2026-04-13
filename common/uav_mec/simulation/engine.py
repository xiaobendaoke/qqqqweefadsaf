from __future__ import annotations

from dataclasses import dataclass
import random

from ..config import SystemConfig
from ..core.action import scale_action
from ..entities import BaseStation, ServiceCatalog, UAVNode, UserEquipment
from ..metrics import MetricTracker
from ..scheduler.assignment import assign_uav
from ..scheduler.offloading import OffloadingDecision, decide_offloading
from ..scheduler.tdma import TDMAQueue
from ..task import Task
from .task_generator import generate_tasks


@dataclass(slots=True)
class StepExecution:
    generated_tasks: list[Task]
    decisions: list[OffloadingDecision]
    energy_spent: float

def run_step(
    *,
    config: SystemConfig,
    users: list[UserEquipment],
    uavs: list[UAVNode],
    bs: BaseStation,
    service_catalog: ServiceCatalog,
    tdma_queue: TDMAQueue,
    metrics: MetricTracker,
    current_step: int,
    actions: list[list[float]],
    rng: random.Random,
) -> StepExecution:
    current_time = current_step * config.time_slot_duration
    for user in users:
        user.move(config, rng)

    energy_spent = 0.0
    for idx, uav in enumerate(uavs):
        uav.reset_step_counters()
        energy_spent += uav.move(scale_action(actions[idx], config), config)
        uav.current_queue_length = tdma_queue.get_queue_length(queue_id=f"uav:{uav.uav_id}")
        uav.current_queue_delay = tdma_queue.estimate_wait(current_time, queue_id=f"uav:{uav.uav_id}")
        uav.max_queue_length = max(uav.max_queue_length, uav.current_queue_length)
        uav.cumulative_queue_delay += uav.current_queue_delay

    tasks = generate_tasks(
        users=users,
        current_time=current_time,
        step_index=current_step,
        config=config,
        service_catalog=service_catalog,
        rng=rng,
    )

    decisions: list[OffloadingDecision] = []
    if uavs:
        uav_lookup = {uav.uav_id: uav for uav in uavs}
        for task in tasks:
            ue = users[task.user_id]
            selected_uav = assign_uav(
                uavs=uavs,
                ue=ue,
                tdma_queue=tdma_queue,
                current_time=current_time,
                rule=config.assignment_rule,
            )
            if selected_uav is None:
                continue
            decision = decide_offloading(
                task=task,
                ue=ue,
                uav=selected_uav,
                all_uavs=uavs,
                bs=bs,
                service_catalog=service_catalog,
                config=config,
                current_time=current_time,
                tdma_queue=tdma_queue,
            )
            task.mark_result(
                execution_target=decision.target,
                cache_hit=decision.cache_hit,
                fetch_delay=decision.fetch_delay,
                queue_delay=decision.queue_delay,
                transmission_delay=decision.transmission_delay,
                compute_delay=decision.compute_delay,
                total_latency=decision.total_latency,
                success_probability=decision.success_probability,
                completed=decision.completed,
                assigned_uav_id=decision.assigned_uav_id,
            )
            if decision.assigned_uav_id is not None:
                executed_uav = uav_lookup[decision.assigned_uav_id]
                executed_uav.assigned_task_count_step += 1
                executed_uav.total_assigned_task_count += 1
            if task.completed:
                users[task.user_id].completed_tasks += 1
                if decision.assigned_uav_id is not None:
                    executed_uav = uav_lookup[decision.assigned_uav_id]
                    executed_uav.served_task_count += 1
                    executed_uav.completed_task_count_step += 1
                    executed_uav.total_completed_task_count += 1
                    compute_energy = task.cpu_cycles * config.uav_compute_energy_per_cycle
                    energy_spent += compute_energy
                    executed_uav.remaining_energy_j = max(0.0, executed_uav.remaining_energy_j - compute_energy)
            decisions.append(decision)

    for uav in uavs:
        uav.current_queue_length = tdma_queue.get_queue_length(queue_id=f"uav:{uav.uav_id}")
        uav.current_queue_delay = tdma_queue.estimate_wait(current_time, queue_id=f"uav:{uav.uav_id}")
        uav.max_queue_length = max(uav.max_queue_length, uav.current_queue_length)
        uav.cumulative_queue_delay += uav.current_queue_delay

    metrics.record_tasks(tasks, energy_spent=energy_spent)
    return StepExecution(generated_tasks=tasks, decisions=decisions, energy_spent=energy_spent)
