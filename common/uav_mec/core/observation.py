from __future__ import annotations

import math

from ..config import SystemConfig
from ..entities import UAVNode, UserEquipment
from ..task import Task
from .state import build_uav_state


def build_observations(
    *,
    uavs: list[UAVNode],
    users: list[UserEquipment],
    pending_tasks: list[Task],
    config: SystemConfig,
    current_time: float = 0.0,
) -> list[list[float]]:
    observations: list[list[float]] = []
    for uav in uavs:
        uav_state = build_uav_state(uav=uav, all_uavs=uavs, config=config)
        own = list(uav_state["own_state"])
        neighbor_flat = [value for block in uav_state["neighbor_uav_states"] for value in block]
        tx_queue = list(uav_state["tx_queue_summary"])
        compute_queue = list(uav_state["compute_queue_summary"])
        backlog_summary = list(uav_state["task_backlog_summary"])
        cache_summary = list(uav_state["local_cache_summary"])
        cache_scores = list(uav_state["cache_score_summary"])

        backlog_features: list[float] = []
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
        for user in sorted_users[: config.observation_max_users]:
            user_tasks = pending_by_user.get(user.user_id, [])
            min_slack = min((max(0.0, task.deadline - current_time) for task in user_tasks), default=0.0)
            service_type = user_tasks[0].service_type if user_tasks else 0
            backlog_features.extend(
                [
                    (user.position[0] - uav.position[0]) / config.area_width,
                    (user.position[1] - uav.position[1]) / config.area_height,
                    float(len(user_tasks)) / max(1, config.num_users * config.task_arrival_max_per_step),
                    float(min_slack) / max(config.task_slack_range[1], 1e-6),
                    float(service_type) / max(1, config.num_service_types - 1),
                ]
            )
        while len(backlog_features) < config.observation_max_users * 5:
            backlog_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        observations.append(own + neighbor_flat + tx_queue + compute_queue + backlog_summary + cache_summary + cache_scores + backlog_features)
    return observations
