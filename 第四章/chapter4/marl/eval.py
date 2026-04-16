"""第四章 MARL 评估入口模块。

该模块负责从外部参数构造评估配置，
并调度底层评估器运行已训练模型与启发式基线的比较流程。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import build_marl_config
from .evaluator import run_evaluation


def run_marl_evaluation(
    *,
    seed: int,
    eval_episodes: int,
    num_uavs: int,
    assignment_rule: str,
    model_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造评估配置并运行 MARL 模型评估。

    参数：
        seed: 评估使用的随机种子。
        eval_episodes: 评估回合数。
        num_uavs: 评估时的 UAV 数量。
        assignment_rule: 用户关联 UAV 的规则名称。
        model_path: 待评估模型的 checkpoint 路径。
        overrides: 额外的评估配置覆盖项。

    返回：
        包含评估指标、对比结果和输出路径信息的结果字典。
    """
    config_overrides = {
        "seed": seed,
        "eval_episodes": eval_episodes,
        "num_uavs": num_uavs,
        "assignment_rule": assignment_rule,
    }
    if overrides:
        config_overrides.update(overrides)
    config = build_marl_config(config_overrides)
    return run_evaluation(config, model_path=model_path)
