from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Task:
    task_id: str
    user_id: int
    service_type: int
    input_size_bits: float
    cpu_cycles: float
    arrival_time: float
    slack: float
    deadline: float
    required_reliability: float
    assigned_uav_id: int | None = None
    execution_target: str | None = None
    cache_hit: bool | None = None
    fetch_delay: float = 0.0
    queue_delay: float = 0.0
    transmission_delay: float = 0.0
    compute_delay: float = 0.0
    total_latency: float = 0.0
    success_probability: float = 0.0
    completed: bool = False

    def mark_result(
        self,
        *,
        execution_target: str,
        cache_hit: bool,
        fetch_delay: float,
        queue_delay: float,
        transmission_delay: float,
        compute_delay: float,
        total_latency: float,
        success_probability: float,
        completed: bool,
        assigned_uav_id: int | None = None,
    ) -> None:
        self.execution_target = execution_target
        self.cache_hit = cache_hit
        self.fetch_delay = fetch_delay
        self.queue_delay = queue_delay
        self.transmission_delay = transmission_delay
        self.compute_delay = compute_delay
        self.total_latency = total_latency
        self.success_probability = success_probability
        self.completed = completed
        self.assigned_uav_id = assigned_uav_id
