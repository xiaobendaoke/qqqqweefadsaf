"""UAV 节点实体模块。

该模块定义 UAV 在仿真中的核心运行时状态，
包括位置、能量、缓存、队列负载以及 episode 级任务统计信息。

输入输出与关键参数：
UAV 状态会被环境、调度器、观测编码和日志导出模块共同读取与更新，
其中移动速度、算力和缓存容量等关键属性由系统配置决定。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

from ..config import SystemConfig


@dataclass(slots=True)
class UAVNode:
    """表示单架 UAV 的运行时状态、缓存状态与队列统计。"""

    uav_id: int
    position: list[float]
    altitude: float
    max_speed: float
    compute_hz: float
    energy_capacity_j: float
    remaining_energy_j: float
    service_cache_capacity: int
    service_cache: set[int] = field(default_factory=set)
    cache_request_counts: dict[int, int] = field(default_factory=dict)
    cache_ema_scores: dict[int, float] = field(default_factory=dict)
    cache_value_scores: dict[int, float] = field(default_factory=dict)
    served_task_count: int = 0
    current_tx_queue_length: int = 0
    current_tx_queue_delay: float = 0.0
    current_compute_queue_length: int = 0
    current_compute_queue_delay: float = 0.0
    current_backlog_load: int = 0
    current_coverage_load: int = 0
    assigned_task_count_step: int = 0
    completed_task_count_step: int = 0
    total_assigned_task_count: int = 0
    total_completed_task_count: int = 0
    cumulative_tx_queue_delay: float = 0.0
    cumulative_compute_queue_delay: float = 0.0
    max_tx_queue_length: int = 0
    max_compute_queue_length: int = 0

    @classmethod
    def random_init(cls, uav_id: int, config: SystemConfig, rng: random.Random) -> "UAVNode":
        """按配置初始化 UAV；若给定固定坐标则直接采用固定布局。"""
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
        """执行一次位移并扣减移动能耗，返回本次移动消耗。"""
        if config.fixed_uav_positions and self.uav_id < len(config.fixed_uav_positions):
            return 0.0
        target_x = min(max(self.position[0] + float(delta[0]), 0.0), config.area_width)
        target_y = min(max(self.position[1] + float(delta[1]), 0.0), config.area_height)
        distance = math.dist((target_x, target_y), (self.position[0], self.position[1]))
        if distance <= 1.0e-9 or self.remaining_energy_j <= 0.0:
            return 0.0
        energy_cost = distance * config.uav_move_energy_per_meter
        if energy_cost > self.remaining_energy_j and energy_cost > 1.0e-9:
            ratio = self.remaining_energy_j / energy_cost
            target_x = self.position[0] + (target_x - self.position[0]) * ratio
            target_y = self.position[1] + (target_y - self.position[1]) * ratio
            distance = math.dist((target_x, target_y), (self.position[0], self.position[1]))
            energy_cost = self.remaining_energy_j
        self.position = [target_x, target_y]
        self.remaining_energy_j = max(0.0, self.remaining_energy_j - energy_cost)
        return energy_cost

    @property
    def energy_ratio(self) -> float:
        if self.energy_capacity_j <= 0:
            return 0.0
        return self.remaining_energy_j / self.energy_capacity_j

    def reset_step_counters(self) -> None:
        """清空仅在单个时隙内统计的瞬时负载计数。"""
        self.current_tx_queue_length = 0
        self.current_tx_queue_delay = 0.0
        self.current_compute_queue_length = 0
        self.current_compute_queue_delay = 0.0
        self.current_backlog_load = 0
        self.current_coverage_load = 0
        self.assigned_task_count_step = 0
        self.completed_task_count_step = 0
