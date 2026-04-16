"""第四章策略模块聚合入口。

该模块统一导出第四章多 UAV 启发式移动策略，
使实验脚本与验证脚本可以通过一致接口调用策略实现。
"""

from .mobility_heuristic_multi import select_actions

__all__ = ["select_actions"]
