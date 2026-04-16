"""第四章多 UAV 启发式协同移动策略模块。

该模块根据多 UAV 局部观测中的用户积压、邻居状态和局部负载信息，
为每架 UAV 生成启发式移动方向，作为多 UAV 实验中的轻量基线方法。
"""

from __future__ import annotations

import math


def select_actions(observations: list[list[float]], env=None) -> list[list[float]]:
    actions: list[list[float]] = []
    associated_start = None
    associated_end = None
    if env is not None:
        schema = env.get_observation_schema()
        associated_start = int(schema["section_slices"]["associated_user_state"][0])
        associated_end = int(schema["section_slices"]["associated_user_state"][1])
    for obs in observations:
        start = associated_start if associated_start is not None else max(0, len(obs) - 15)
        end = associated_end if associated_end is not None else len(obs)
        best_block = [0.0, 0.0, 0.0, 1.0, 0.0]
        best_priority = float("-inf")
        for offset in range(start, end, 5):
            block = obs[offset : offset + 5]
            if len(block) < 5:
                continue
            pending = float(block[2])
            if pending <= 0.0:
                continue
            distance = math.hypot(float(block[0]), float(block[1]))
            slack = min(1.0, max(0.0, float(block[3])))
            priority = 1.4 * pending + (1.0 - slack) - 0.25 * distance
            if priority > best_priority:
                best_priority = priority
                best_block = [float(value) for value in block]
        nearest_dx = best_block[0]
        nearest_dy = best_block[1]
        norm = math.hypot(nearest_dx, nearest_dy)
        if norm > 1e-6:
            scale = min(1.0, max(0.35, norm))
            nearest_dx = scale * nearest_dx / norm
            nearest_dy = scale * nearest_dy / norm
        actions.append([nearest_dx, nearest_dy])
    return actions
