"""第四章启发式实验模块聚合入口。

该模块统一导出第四章中的启发式实验、多 UAV 单回合示例和 smoke test 入口，
便于命令行脚本与论文实验流程共享调用方式。
"""

from .experiment import recommended_experiment_matrix, run_experiment, run_sensitive_experiment
from .multi_agent_episode import run_multi_agent_episode
from .smoke import run_smoke

__all__ = [
    "recommended_experiment_matrix",
    "run_experiment",
    "run_multi_agent_episode",
    "run_sensitive_experiment",
    "run_smoke",
]
