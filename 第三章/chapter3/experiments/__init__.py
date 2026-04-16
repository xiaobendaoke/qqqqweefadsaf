"""第三章实验模块聚合入口。

该模块统一导出第三章的实验运行、退化一致性比较和 smoke test 入口，
便于命令行脚本与上层调用方使用统一接口访问实验能力。
"""

from .experiment import compare_with_chapter4, run_experiment
from .finalize import run_chapter3_figure_package
from .smoke import run_smoke

__all__ = ["compare_with_chapter4", "run_chapter3_figure_package", "run_experiment", "run_smoke"]
