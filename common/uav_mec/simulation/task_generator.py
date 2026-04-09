from __future__ import annotations

import random
from typing import Iterable

from ..config import SystemConfig
from ..entities import ServiceCatalog, UserEquipment
from ..task import Task


def generate_tasks(
    *,
    users: Iterable[UserEquipment],
    current_time: float,
    step_index: int,
    config: SystemConfig,
    service_catalog: ServiceCatalog,
    rng: random.Random,
) -> list[Task]:
    tasks: list[Task] = []
    trigger_probability = min(0.95, config.task_arrival_rate * config.time_slot_duration)
    for user in users:
        if rng.random() > trigger_probability:
            continue
        service_type = int(rng.randrange(service_catalog.num_service_types))
        input_size_bits = float(rng.uniform(*config.task_input_size_range_bits))
        cpu_cycles = float(rng.uniform(*config.task_cpu_cycles_range))
        slack = float(rng.uniform(*config.task_slack_range))
        tasks.append(
            Task(
                task_id=f"step{step_index}_user{user.user_id}_{user.generated_tasks}",
                user_id=user.user_id,
                service_type=service_type,
                input_size_bits=input_size_bits,
                cpu_cycles=cpu_cycles,
                arrival_time=current_time,
                slack=slack,
                deadline=current_time + slack,
                required_reliability=config.required_reliability,
            )
        )
        user.generated_tasks += 1
    return tasks
