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
) -> list[list[float]]:
    observations: list[list[float]] = []
    for uav in uavs:
        uav_state = build_uav_state(uav=uav, all_uavs=uavs, config=config)
        own = list(uav_state["own_state"])
        neighbor_flat = [value for block in uav_state["neighbor_uav_states"] for value in block]
        queue_summary = list(uav_state["queue_load_summary"])
        cache_summary = list(uav_state["local_cache_summary"])
        nearest_features: list[float] = []
        sorted_users = sorted(
            users,
            key=lambda item: math.dist((item.position[0], item.position[1]), (uav.position[0], uav.position[1])),
        )
        for user in sorted_users[: config.observation_max_users]:
            user_tasks = [task for task in pending_tasks if task.user_id == user.user_id]
            min_slack = min((task.slack for task in user_tasks), default=0.0)
            service_type = user_tasks[0].service_type if user_tasks else 0
            nearest_features.extend(
                [
                    (user.position[0] - uav.position[0]) / config.area_width,
                    (user.position[1] - uav.position[1]) / config.area_height,
                    float(len(user_tasks)) / max(1, config.num_users),
                    float(min_slack) / max(config.task_slack_range[1], 1e-6),
                    float(service_type) / max(1, config.num_service_types - 1),
                ]
            )
        while len(nearest_features) < config.observation_max_users * 5:
            nearest_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        observations.append(own + neighbor_flat + queue_summary + cache_summary + nearest_features)
    return observations
