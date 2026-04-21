"""环境动作编码模块。

该模块定义 joint action schema，并在不破坏旧接口的前提下，
为 mobility-only 与 joint-action 两条输入路径提供统一归一化与 schema 导出。

当前实现状态：
- `legacy_mobility_only` 仍保留为 baseline 路径
- `joint_action_v2` 已真实接入 `run_step` 执行链路
- offloading 与 caching 不再只是 schema，占位动作会被 joint scheduler 直接消费
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import SystemConfig

ACTION_DIM = 2
MOBILITY_ACTION_DIM = ACTION_DIM
DEFAULT_DEFER_PLAN_ID = -1
DEFAULT_TASK_SLOT_COUNT = 3
LEGACY_ACTION_MODE = "legacy_mobility_only"
JOINT_ACTION_MODE = "joint_action_v2"


@dataclass(slots=True)
class MobilityAction:
    """连续 mobility 动作。"""

    dx: float
    dy: float


@dataclass(slots=True)
class OffloadingAction:
    """固定 K 个 task slot 的 candidate plan 选择。"""

    task_slot_plan_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class CachingAction:
    """每类服务的 cache priority 分数。"""

    service_priorities: list[float] = field(default_factory=list)


@dataclass(slots=True)
class JointAgentAction:
    """单个 UAV 的 joint action 标准化载体。"""

    mobility: MobilityAction
    offloading: OffloadingAction
    caching: CachingAction
    action_mode: str = JOINT_ACTION_MODE


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_mobility_vector(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == MOBILITY_ACTION_DIM
        and all(_is_number(item) for item in value)
    )


def _resolve_task_slot_count(*, config: SystemConfig | None = None, task_slot_count: int | None = None) -> int:
    if task_slot_count is not None:
        return max(1, int(task_slot_count))
    if config is not None:
        return max(1, int(getattr(config, "task_arrival_max_per_step", DEFAULT_TASK_SLOT_COUNT)))
    return DEFAULT_TASK_SLOT_COUNT


def _normalize_plan_id(value: Any) -> int:
    if value is None:
        return DEFAULT_DEFER_PLAN_ID
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"defer", "skip", "none"}:
            return DEFAULT_DEFER_PLAN_ID
        try:
            parsed = int(lowered)
        except ValueError as exc:
            raise ValueError(f"Unsupported candidate plan token: {value!r}") from exc
    else:
        parsed = int(value)
    if parsed < DEFAULT_DEFER_PLAN_ID:
        raise ValueError(f"candidate_plan_id must be >= {DEFAULT_DEFER_PLAN_ID}, got {parsed}")
    return parsed


def _parse_mobility_action(value: Any) -> MobilityAction:
    if isinstance(value, MobilityAction):
        return MobilityAction(dx=float(value.dx), dy=float(value.dy))
    if isinstance(value, dict):
        if "dx" in value and "dy" in value:
            return MobilityAction(dx=float(value["dx"]), dy=float(value["dy"]))
        raise ValueError("Mobility action dict must contain 'dx' and 'dy'.")
    if _is_mobility_vector(value):
        return MobilityAction(dx=float(value[0]), dy=float(value[1]))
    raise ValueError("Mobility action must be [dx, dy], (dx, dy), MobilityAction, or {'dx', 'dy'} dict.")


def _parse_offloading_action(value: Any, *, task_slot_count: int) -> OffloadingAction:
    if value is None:
        return OffloadingAction(task_slot_plan_ids=[DEFAULT_DEFER_PLAN_ID] * task_slot_count)
    if isinstance(value, OffloadingAction):
        plan_ids = value.task_slot_plan_ids
    elif isinstance(value, dict):
        if "task_slot_plan_ids" in value:
            plan_ids = value["task_slot_plan_ids"]
        elif "task_slots" in value:
            plan_ids = value["task_slots"]
        else:
            raise ValueError("Offloading action dict must contain 'task_slot_plan_ids' or 'task_slots'.")
    elif isinstance(value, (list, tuple)):
        plan_ids = value
    else:
        raise ValueError("Offloading action must be a list/tuple, OffloadingAction, or dict.")
    if len(plan_ids) != task_slot_count:
        raise ValueError(f"Offloading action must provide exactly {task_slot_count} task slot decisions.")
    return OffloadingAction(task_slot_plan_ids=[_normalize_plan_id(item) for item in plan_ids])


def _parse_caching_action(value: Any, *, num_service_types: int) -> CachingAction:
    if value is None:
        return CachingAction(service_priorities=[0.0] * num_service_types)
    if isinstance(value, CachingAction):
        priorities = value.service_priorities
    elif isinstance(value, dict):
        if "service_priorities" in value:
            priorities = value["service_priorities"]
        elif "cache_priority_scores" in value:
            priorities = value["cache_priority_scores"]
        else:
            raise ValueError("Caching action dict must contain 'service_priorities' or 'cache_priority_scores'.")
    elif isinstance(value, (list, tuple)):
        priorities = value
    else:
        raise ValueError("Caching action must be a list/tuple, CachingAction, or dict.")
    if len(priorities) != num_service_types:
        raise ValueError(f"Caching action must provide {num_service_types} service priority scores.")
    return CachingAction(service_priorities=[float(item) for item in priorities])


def _from_legacy_mobility(
    value: Any,
    *,
    task_slot_count: int,
    num_service_types: int,
) -> JointAgentAction:
    mobility = _parse_mobility_action(value)
    return JointAgentAction(
        mobility=mobility,
        offloading=OffloadingAction(task_slot_plan_ids=[DEFAULT_DEFER_PLAN_ID] * task_slot_count),
        caching=CachingAction(service_priorities=[0.0] * num_service_types),
        action_mode=LEGACY_ACTION_MODE,
    )


def _from_joint_payload(
    value: Any,
    *,
    task_slot_count: int,
    num_service_types: int,
) -> JointAgentAction:
    if isinstance(value, JointAgentAction):
        return JointAgentAction(
            mobility=_parse_mobility_action(value.mobility),
            offloading=_parse_offloading_action(value.offloading, task_slot_count=task_slot_count),
            caching=_parse_caching_action(value.caching, num_service_types=num_service_types),
            action_mode=JOINT_ACTION_MODE,
        )
    if not isinstance(value, dict):
        raise ValueError("Joint action must be a dict or JointAgentAction instance.")
    if "mobility" not in value:
        raise ValueError("Joint action dict must contain a 'mobility' component.")
    return JointAgentAction(
        mobility=_parse_mobility_action(value.get("mobility")),
        offloading=_parse_offloading_action(value.get("offloading"), task_slot_count=task_slot_count),
        caching=_parse_caching_action(value.get("caching"), num_service_types=num_service_types),
        action_mode=JOINT_ACTION_MODE,
    )


def _is_joint_payload(value: Any) -> bool:
    return isinstance(value, JointAgentAction) or (isinstance(value, dict) and any(key in value for key in ("mobility", "offloading", "caching")))


def action_to_dict(action: JointAgentAction) -> dict[str, object]:
    """把标准化后的单智能体动作转成可序列化 dict。"""
    return {
        "action_mode": action.action_mode,
        "mobility": {"dx": float(action.mobility.dx), "dy": float(action.mobility.dy)},
        "offloading": {"task_slot_plan_ids": [int(item) for item in action.offloading.task_slot_plan_ids]},
        "caching": {"service_priorities": [float(item) for item in action.caching.service_priorities]},
    }


def mobility_action_vector(action: JointAgentAction | list[float] | tuple[float, float] | dict[str, Any]) -> list[float]:
    """提取 mobility 分量，供当前 legacy 执行链路复用。"""
    if isinstance(action, JointAgentAction):
        mobility = action.mobility
    elif isinstance(action, dict) and "mobility" in action:
        mobility = _parse_mobility_action(action["mobility"])
    else:
        mobility = _parse_mobility_action(action)
    return [float(mobility.dx), float(mobility.dy)]


def extract_mobility_actions(actions: list[JointAgentAction]) -> list[list[float]]:
    """从 joint action 列表中提取当前环境可执行的 mobility 分量。"""
    return [mobility_action_vector(action) for action in actions]


def scale_action(
    action: JointAgentAction | list[float] | tuple[float, float] | dict[str, Any],
    config: SystemConfig,
) -> list[float]:
    """将 mobility 分量裁剪到单位圆内，再映射为真实位移距离。"""
    mobility = mobility_action_vector(action)
    clipped = [min(max(float(mobility[0]), -1.0), 1.0), min(max(float(mobility[1]), -1.0), 1.0)]
    norm = (clipped[0] ** 2 + clipped[1] ** 2) ** 0.5
    if norm > 1.0 and norm > 1e-8:
        clipped = [clipped[0] / norm, clipped[1] / norm]
    max_distance = config.uav_speed * config.time_slot_duration
    return [clipped[0] * max_distance, clipped[1] * max_distance]


def normalize_actions(
    actions: list[dict[str, Any]] | list[list[float]] | list[tuple[float, float]] | list[float] | tuple[float, float] | dict[str, Any],
    *,
    num_agents: int,
    num_service_types: int,
    task_slot_count: int | None = None,
    config: SystemConfig | None = None,
) -> list[JointAgentAction]:
    """统一动作输入形状，兼容 legacy mobility-only 与新的 joint action。"""
    resolved_task_slot_count = _resolve_task_slot_count(config=config, task_slot_count=task_slot_count)

    if num_agents == 1 and (_is_mobility_vector(actions) or _is_joint_payload(actions)):
        candidate_actions = [actions]
    else:
        if not isinstance(actions, (list, tuple)) or len(actions) != num_agents:
            raise ValueError(
                f"Expected either legacy mobility shape [{num_agents}, {MOBILITY_ACTION_DIM}] "
                f"or joint action list with {num_agents} agent payloads."
            )
        candidate_actions = list(actions)

    normalized: list[JointAgentAction] = []
    for index, action in enumerate(candidate_actions):
        if _is_joint_payload(action):
            normalized.append(
                _from_joint_payload(
                    action,
                    task_slot_count=resolved_task_slot_count,
                    num_service_types=num_service_types,
                )
            )
            continue
        if _is_mobility_vector(action):
            normalized.append(
                _from_legacy_mobility(
                    action,
                    task_slot_count=resolved_task_slot_count,
                    num_service_types=num_service_types,
                )
            )
            continue
        raise ValueError(
            f"Action for agent {index} must be either a legacy mobility vector [dx, dy] "
            "or a joint action payload with mobility/offloading/caching components."
        )
    return normalized


def action_schema(*, num_agents: int, agent_ids: list[str], config: SystemConfig, task_slot_count: int | None = None) -> dict[str, object]:
    """导出兼容 legacy 的 joint action schema，供环境、日志和后续模型升级复用。"""
    resolved_task_slot_count = _resolve_task_slot_count(config=config, task_slot_count=task_slot_count)
    max_distance = config.uav_speed * config.time_slot_duration
    legacy_schema = {
        "mode": LEGACY_ACTION_MODE,
        "canonical_shape": [num_agents, MOBILITY_ACTION_DIM],
        "single_agent_compatibility": [MOBILITY_ACTION_DIM] if num_agents == 1 else None,
        "fields_per_agent": ["dx", "dy"],
        "value_range": [-1.0, 1.0],
        "scaled_max_displacement_per_step": max_distance,
    }
    return {
        "schema_version": "action.v2",
        "agent_order": agent_ids,
        "accepted_action_modes": [LEGACY_ACTION_MODE, JOINT_ACTION_MODE],
        # Keep these top-level keys for trainer/evaluator compatibility until
        # the learning stack is upgraded to the joint-action interface.
        "canonical_shape": legacy_schema["canonical_shape"],
        "single_agent_compatibility": legacy_schema["single_agent_compatibility"],
        "fields_per_agent": legacy_schema["fields_per_agent"],
        "value_range": legacy_schema["value_range"],
        "scaled_max_displacement_per_step": legacy_schema["scaled_max_displacement_per_step"],
        "legacy_mobility_only": legacy_schema,
        "joint_action": {
            "mode": JOINT_ACTION_MODE,
            "agent_keys": ["mobility", "offloading", "caching"],
            "mobility": {
                "type": "continuous_vector",
                "fields": ["dx", "dy"],
                "shape": [MOBILITY_ACTION_DIM],
                "value_range": [-1.0, 1.0],
                "scaled_max_displacement_per_step": max_distance,
            },
            "offloading": {
                "type": "task_slot_plan_selection",
                "field": "task_slot_plan_ids",
                "task_slot_count": resolved_task_slot_count,
                "candidate_plan_id_domain": "int >= 0",
                "defer_plan_id": DEFAULT_DEFER_PLAN_ID,
                "defer_token": "defer",
                "default_payload": [DEFAULT_DEFER_PLAN_ID] * resolved_task_slot_count,
                "binding_note": "active in joint_action mode; invalid or infeasible selections fall back to defer",
            },
            "caching": {
                "type": "service_priority_scores",
                "field": "service_priorities",
                "num_service_types": config.num_service_types,
                "default_payload": [0.0] * config.num_service_types,
                "binding_note": "active in joint_action mode; scores are projected to a cache set at slot end",
            },
        },
        "execution_binding": {
            "active_components_by_mode": {
                LEGACY_ACTION_MODE: ["mobility"],
                JOINT_ACTION_MODE: ["mobility", "offloading", "caching"],
            },
            "fallback_policy": "invalid or infeasible offloading selections fall back to defer in joint_action mode",
            "note": "Legacy mobility-only mode keeps the old heuristic scheduler; joint_action mode executes policy-selected mobility/offloading/caching directly.",
        },
        "migration_note": {
            "task_slot_count_source": "task_arrival_max_per_step",
            "future_override": "Step 2 can replace this with a dedicated action_task_slot_count config when candidate slots are added to observations.",
        },
    }
