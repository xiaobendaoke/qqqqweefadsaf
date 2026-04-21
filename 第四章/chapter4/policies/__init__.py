"""第四章策略模块聚合入口。"""

from .joint_heuristic_multi import select_actions as select_joint_actions
from .mobility_heuristic_multi import select_actions as select_legacy_mobility_actions

__all__ = [
    "select_joint_actions",
    "select_legacy_mobility_actions",
]
