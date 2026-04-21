"""用户关联 UAV 选择模块。

该模块负责在覆盖范围内为用户选择关联 UAV，
支持按距离最近或按链路/计算负载最小等规则进行关联决策。
"""

from __future__ import annotations

import math

from ..entities import UAVNode, UserEquipment
from .compute_queue import ComputeQueue
from .tdma import EDGE_ACCESS_QUEUE_ID
from .tdma import TDMAQueue


def assign_uav(
    *,
    uavs: list[UAVNode],
    ue: UserEquipment,
    tdma_queue: TDMAQueue,
    compute_queue: ComputeQueue | None,
    current_time: float,
    coverage_radius: float,
    rule: str,
) -> UAVNode | None:
    """为用户选择关联 UAV，支持距离优先与负载优先两种规则。"""
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
                tdma_queue.get_queue_length(queue_id=EDGE_ACCESS_QUEUE_ID, current_time=current_time),
                tdma_queue.estimate_wait(current_time, queue_id=EDGE_ACCESS_QUEUE_ID),
                (
                    compute_queue.get_queue_length(queue_id=f"uav:{item.uav_id}", current_time=current_time)
                    if compute_queue is not None
                    else item.current_compute_queue_length
                ),
                (
                    compute_queue.estimate_wait(current_time, queue_id=f"uav:{item.uav_id}")
                    if compute_queue is not None
                    else item.current_compute_queue_delay
                ),
                item.total_assigned_task_count - item.total_completed_task_count,
                math.dist((item.position[0], item.position[1]), (ue.position[0], ue.position[1])),
            ),
        )
    raise ValueError(f"Unsupported assignment rule: {rule}")
