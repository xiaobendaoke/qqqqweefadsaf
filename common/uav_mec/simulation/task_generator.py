"""任务生成模块。

该模块根据到达率、任务规模范围和用户集合，
在每个离散时隙内随机生成新的任务对象，是环境工作负载的主要来源。
"""

from __future__ import annotations

import math
import random
from typing import Iterable

from ..config import SystemConfig
from ..entities import ServiceCatalog, UserEquipment
from ..task import Task


def _sample_poisson(lam: float, rng: random.Random, *, max_count: int) -> int:
    if lam <= 0.0:
        return 0
    limit = math.exp(-lam)
    count = 0
    product = 1.0
    while count < max_count:
        product *= max(rng.random(), 1.0e-9)
        if product <= limit:
            break
        count += 1
    return count


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
    arrival_lambda = max(0.0, config.task_arrival_rate * config.time_slot_duration)
    for user in users:
        task_count = _sample_poisson(arrival_lambda, rng, max_count=config.task_arrival_max_per_step)
        for _ in range(task_count):
            service_type = int(rng.randrange(service_catalog.num_service_types))
            input_size_bits = float(rng.uniform(*config.task_input_size_range_bits))
            cpu_cycles = float(rng.uniform(*config.task_cpu_cycles_range))
            slack = float(rng.uniform(*config.task_slack_range))
            task_index = user.generated_tasks
            user.generated_tasks += 1
            tasks.append(
                Task(
                    task_id=f"step{step_index}_user{user.user_id}_{task_index}",
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
    return tasks
