from __future__ import annotations

import math

from ..entities import UAVNode, UserEquipment
from .tdma import TDMAQueue


def assign_uav(
    *,
    uavs: list[UAVNode],
    ue: UserEquipment,
    tdma_queue: TDMAQueue,
    current_time: float,
    coverage_radius: float,
    rule: str,
) -> UAVNode | None:
    if not uavs:
        return None
    covering_uavs = [
        item
        for item in uavs
        if math.dist((item.position[0], item.position[1]), (ue.position[0], ue.position[1])) <= coverage_radius
    ]
    if not covering_uavs:
        return None
    if rule == "nearest_uav":
        return min(
            covering_uavs,
            key=lambda item: math.dist((item.position[0], item.position[1]), (ue.position[0], ue.position[1])),
        )
    if rule == "least_loaded_uav":
        return min(
            covering_uavs,
            key=lambda item: (
                tdma_queue.get_queue_length(queue_id=f"uav:{item.uav_id}", current_time=current_time),
                tdma_queue.estimate_wait(current_time, queue_id=f"uav:{item.uav_id}"),
                item.current_compute_queue_length,
                item.current_compute_queue_delay,
                item.served_task_count,
                math.dist((item.position[0], item.position[1]), (ue.position[0], ue.position[1])),
            ),
        )
    raise ValueError(f"Unsupported assignment rule: {rule}")
