"""第四章 MARL 模块聚合入口。

该模块统一导出第四章中的 MARL 训练、评估、论文实验与终稿打包入口，
并通过惰性导入隔离 stage-5 论文实验代码对其他命令行流程的影响。
"""

from .eval import run_marl_evaluation
from .finalize import run_final_paper_package
from .train import run_marl_training


def run_paper_experiments(*args, **kwargs):
    # Keep stage-5 helpers lazily imported so train/eval/finalize CLIs do not
    # fail if paper-specific code is temporarily inconsistent.
    from .paper import run_paper_experiments as _run_paper_experiments

    return _run_paper_experiments(*args, **kwargs)


__all__ = [
    "run_marl_training",
    "run_marl_evaluation",
    "run_paper_experiments",
    "run_final_paper_package",
]
