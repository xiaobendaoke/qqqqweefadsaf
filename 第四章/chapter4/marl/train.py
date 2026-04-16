"""第四章 MARL 训练入口模块。

该模块负责从外部参数构造训练配置，
并调度底层训练器运行 shared-actor centralized-critic 训练流程。
"""

from __future__ import annotations

from typing import Any

from .config import build_marl_config
from .trainer import run_training


def run_marl_training(
    *,
    seed: int,
    train_episodes: int,
    num_uavs: int,
    assignment_rule: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造训练配置并运行第四章 MARL 训练。

    参数：
        seed: 训练起始随机种子。
        train_episodes: 训练回合数。
        num_uavs: 训练场景中的 UAV 数量。
        assignment_rule: 用户关联 UAV 的规则名称。
        overrides: 额外的训练配置覆盖项。

    返回：
        包含训练日志、模型路径和接口 schema 的结果字典。
    """
    config_overrides = {
        "seed": seed,
        "train_episodes": train_episodes,
        "num_uavs": num_uavs,
        "assignment_rule": assignment_rule,
    }
    if overrides:
        config_overrides.update(overrides)
    config = build_marl_config(config_overrides)
    return run_training(config)
