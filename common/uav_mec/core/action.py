"""环境动作编码模块。

该模块定义 UAV 平面移动动作的归一化表示、缩放规则与 schema，
用于统一单 UAV 与多 UAV 场景下策略输出与环境执行之间的接口契约。

输入输出与关键参数：
输入可以是单个二维动作或按智能体组织的动作列表；
输出为环境可执行的标准化动作结构或对应的动作 schema 描述。
"""

from __future__ import annotations

from ..config import SystemConfig

ACTION_DIM = 2


def scale_action(action: list[float] | tuple[float, float], config: SystemConfig) -> list[float]:
    """将归一化动作裁剪到单位圆内，再映射为真实位移距离。"""
    clipped = [min(max(float(action[0]), -1.0), 1.0), min(max(float(action[1]), -1.0), 1.0)]
    norm = (clipped[0] ** 2 + clipped[1] ** 2) ** 0.5
    if norm > 1.0 and norm > 1e-8:
        clipped = [clipped[0] / norm, clipped[1] / norm]
    max_distance = config.uav_speed * config.time_slot_duration
    return [clipped[0] * max_distance, clipped[1] * max_distance]


def normalize_actions(
    actions: list[list[float]] | list[tuple[float, float]] | list[float] | tuple[float, float],
    *,
    num_agents: int,
) -> list[list[float]]:
    """统一动作输入形状，兼容单 UAV 与多 UAV 调用方式。"""
    if num_agents == 1 and isinstance(actions, (list, tuple)) and len(actions) == ACTION_DIM and not isinstance(actions[0], (list, tuple)):
        return [[float(actions[0]), float(actions[1])]]

    normalized: list[list[float]] = []
    if not isinstance(actions, (list, tuple)) or len(actions) != num_agents:
        raise ValueError(f"Expected canonical action shape [{num_agents}, {ACTION_DIM}]")
    for index, action in enumerate(actions):
        if not isinstance(action, (list, tuple)) or len(action) != ACTION_DIM:
            raise ValueError(f"Action for agent {index} must have length {ACTION_DIM}")
        normalized.append([float(action[0]), float(action[1])])
    return normalized


def action_schema(*, num_agents: int, agent_ids: list[str], config: SystemConfig) -> dict[str, object]:
    """导出动作张量契约，供实验日志和训练脚本复用。"""
    return {
        "schema_version": "action.v1",
        "canonical_shape": [num_agents, ACTION_DIM],
        "single_agent_compatibility": [ACTION_DIM] if num_agents == 1 else None,
        "agent_order": agent_ids,
        "fields_per_agent": ["dx", "dy"],
        "value_range": [-1.0, 1.0],
        "scaled_max_displacement_per_step": config.uav_speed * config.time_slot_duration,
    }
