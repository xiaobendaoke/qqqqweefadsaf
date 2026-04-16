"""第三章固定巡航路径基线策略。

该模块实现单 UAV 的预定义巡航轨迹基线方法，
用于在不依赖观测反馈的情况下，考察规则化移动模式的覆盖与服务效果。

核心方法说明：
策略按照预设的八方向巡航序列循环输出动作，
使 UAV 在 episode 内沿近似环绕路径移动，形成与静止基线和启发式基线不同的行为模式。

输入输出与关键参数：
输入为观测列表 `observations` 和可选环境对象 `env`；
输出为与观测条目数量一致的二维归一化动作列表。
关键参数是 `_PATROL_SEQUENCE` 中定义的固定巡航方向序列。

边界说明：
该策略不根据任务积压、时延紧迫性或缓存状态调整行为，
其移动模式完全由预设路径决定。
"""

from __future__ import annotations

import math


_PATROL_SEQUENCE = [
    [0.75, 0.0],
    [0.53, 0.53],
    [0.0, 0.75],
    [-0.53, 0.53],
    [-0.75, 0.0],
    [-0.53, -0.53],
    [0.0, -0.75],
    [0.53, -0.53],
]


def select_actions(observations: list[list[float]], env=None) -> list[list[float]]:
    """根据当前时刻所在巡航相位生成固定巡航动作。

    参数：
        observations: 当前时刻的观测列表，仅用于确定返回动作数量。
        env: 当前环境实例；若提供，则读取 `current_step` 以确定巡航相位。

    返回：
        与观测条目数量一致的二维归一化动作列表。

    说明：
        当 `env` 为空时无法确定当前巡航步次，因此退化为零动作；
        否则按 `current_step` 在固定巡航序列中循环取值。
    """
    if env is None:
        return [[0.0, 0.0] for _ in observations]
    phase = int(getattr(env, "current_step", 0)) % len(_PATROL_SEQUENCE)
    action = _PATROL_SEQUENCE[phase]
    normalized = _normalize(action)
    return [normalized[:] for _ in observations]


def _normalize(action: list[float]) -> list[float]:
    """将巡航序列中的动作向量规范到环境允许的归一化范围内。

    参数：
        action: 待归一化的二维动作向量。

    返回：
        长度不超过 1 的二维动作向量；若输入近似零向量，则返回零动作。
    """
    norm = math.hypot(action[0], action[1])
    if norm <= 1e-6:
        return [0.0, 0.0]
    return [float(action[0]) / max(1.0, norm), float(action[1]) / max(1.0, norm)]
