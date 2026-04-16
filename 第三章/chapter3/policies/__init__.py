"""第三章策略模块聚合入口。

该模块统一导出第三章实验中使用的单 UAV 策略接口，
包括启发式基线、固定悬停点、固定巡航路径以及 MPC shell 策略。

输入输出与边界说明：
各策略模块均遵循统一的 `select_actions(observations, env=None)` 调用约定，
从而可以被实验脚本按同一方式调度。
该聚合模块本身不实现决策逻辑，仅负责对外暴露统一导入入口。
"""

from .mobility_heuristic import select_actions
from .fixed_point import select_actions as select_actions_fixed_point
from .fixed_patrol import select_actions as select_actions_fixed_patrol
from .mpc_shell import select_actions as select_actions_mpc

__all__ = ["select_actions", "select_actions_fixed_point", "select_actions_fixed_patrol", "select_actions_mpc"]
