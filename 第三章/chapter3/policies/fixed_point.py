"""第三章固定悬停点基线策略。

该模块实现最简单的单 UAV 对照方法，使 UAV 在整个 episode 中保持静止，
用于衡量“完全不移动”条件下的任务完成率、时延和能耗表现。

核心方法说明：
策略在每个时刻都输出零位移动作，不依赖观测内容，也不根据环境状态调整行为。

输入输出与关键参数：
输入为观测列表 `observations`，可选输入为环境对象 `env`；
输出为与观测条目数量一致的二维零动作列表。

边界说明：
该策略仅作为消融型基线，不包含任何调度优化、轨迹规划或目标选择逻辑。
"""

from __future__ import annotations


def select_actions(observations: list[list[float]], env=None) -> list[list[float]]:
    """为所有 UAV 返回零位移动作。

    参数：
        observations: 当前时刻的观测列表，仅用于确定需要返回多少个动作。
        env: 当前环境实例；该策略不使用环境状态，保留该参数仅为统一接口。

    返回：
        与观测条目数量一致的二维零动作列表。
    """
    return [[0.0, 0.0] for _ in observations]
