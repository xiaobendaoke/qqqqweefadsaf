from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

from ..config import SystemConfig


@dataclass(slots=True)
class UAVNode:
    uav_id: int
    position: list[float]
    altitude: float
    max_speed: float
    compute_hz: float
    energy_capacity_j: float
    remaining_energy_j: float
    service_cache_capacity: int
    service_cache: set[int] = field(default_factory=set)
    served_task_count: int = 0
    current_queue_length: int = 0
    current_queue_delay: float = 0.0
    assigned_task_count_step: int = 0
    completed_task_count_step: int = 0
    total_assigned_task_count: int = 0
    total_completed_task_count: int = 0
    cumulative_queue_delay: float = 0.0
    max_queue_length: int = 0

    @classmethod
    def random_init(cls, uav_id: int, config: SystemConfig, rng: random.Random) -> "UAVNode":
        if config.fixed_uav_positions and uav_id < len(config.fixed_uav_positions):
            fixed = config.fixed_uav_positions[uav_id]
            position = [float(fixed[0]), float(fixed[1])]
        else:
            center_x = config.area_width / 2.0
            center_y = config.area_height / 2.0
            jitter_x = rng.uniform(-20.0, 20.0)
            jitter_y = rng.uniform(-20.0, 20.0)
            position = [center_x + jitter_x, center_y + jitter_y]
        return cls(
            uav_id=uav_id,
            position=position,
            altitude=config.uav_altitude,
            max_speed=config.uav_speed,
            compute_hz=config.uav_compute_hz,
            energy_capacity_j=config.uav_energy_capacity_j,
            remaining_energy_j=config.uav_energy_capacity_j,
            service_cache_capacity=config.uav_service_cache_capacity,
        )

    def move(self, delta: list[float], config: SystemConfig) -> float:
        next_x = min(max(self.position[0] + float(delta[0]), 0.0), config.area_width)
        next_y = min(max(self.position[1] + float(delta[1]), 0.0), config.area_height)
        distance = math.dist((next_x, next_y), (self.position[0], self.position[1]))
        self.position = [next_x, next_y]
        energy_cost = distance * config.uav_move_energy_per_meter
        self.remaining_energy_j = max(0.0, self.remaining_energy_j - energy_cost)
        return energy_cost

    @property
    def energy_ratio(self) -> float:
        if self.energy_capacity_j <= 0:
            return 0.0
        return self.remaining_energy_j / self.energy_capacity_j

    def reset_step_counters(self) -> None:
        self.current_queue_length = 0
        self.current_queue_delay = 0.0
        self.assigned_task_count_step = 0
        self.completed_task_count_step = 0
