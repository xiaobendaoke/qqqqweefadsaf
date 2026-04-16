"""第三章启发式移动策略。

该模块实现第三章单 UAV 场景下的轻量启发式基线方法，
用于在不引入显式优化求解器的前提下，根据局部观测直接生成移动动作。

核心方法说明：
策略从观测中的关联用户块中筛选“任务积压更高、时延更紧、距离更近”的目标，
并据此确定 UAV 的下一步移动方向，是第三章中的最小可运行对照方法。

输入输出与关键参数：
输入为统一观测向量 `observations`，可选输入为环境对象 `env`，
用于读取观测 schema 中关联用户状态块的切片位置。
输出为二维归一化动作列表；优先级计算中的 backlog、slack 和距离系数由当前实现固定给定。

边界说明：
该策略不进行多步预测，也不显式建模缓存、队列和长期能耗收益，
主要作为与 MPC shell、固定点、固定巡航策略对比的基线。
"""

from __future__ import annotations

import math


def select_actions(observations: list[list[float]], env=None) -> list[list[float]]:
    """根据当前观测生成启发式移动动作。

    参数：
        observations: 环境返回的局部观测列表，每个元素对应一个 UAV 的观测向量。
        env: 可选环境实例；若提供，则使用其中的 observation schema 精确定位用户状态块。

    返回：
        与观测条目数量一致的二维动作列表，每个动作表示归一化平面移动方向。

    说明：
        策略对每个观测独立处理，优先选择 backlog 更高、slack 更小且相对距离更短的用户方向，
        再对动作幅度做保守裁剪，避免始终满速移动。
    """
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
            priority = 1.5 * pending + (1.0 - slack) - 0.35 * distance
            if priority > best_priority:
                best_priority = priority
                best_block = [float(value) for value in block]
        nearest_dx = best_block[0]
        nearest_dy = best_block[1]
        norm = math.hypot(nearest_dx, nearest_dy)
        if norm > 1e-6:
            # 在保留方向的同时限制幅度，避免启发式策略因远距离目标而始终满速移动。
            scale = min(1.0, max(0.35, norm))
            nearest_dx = scale * nearest_dx / norm
            nearest_dy = scale * nearest_dy / norm
        actions.append([nearest_dx, nearest_dy])
    return actions
