from __future__ import annotations

from typing import Any

from .config import build_marl_config
from .trainer import run_training


def run_marl_training(*, seed: int, train_episodes: int, num_uavs: int, assignment_rule: str) -> dict[str, Any]:
    config = build_marl_config(
        {
            "seed": seed,
            "train_episodes": train_episodes,
            "num_uavs": num_uavs,
            "assignment_rule": assignment_rule,
        }
    )
    return run_training(config)
