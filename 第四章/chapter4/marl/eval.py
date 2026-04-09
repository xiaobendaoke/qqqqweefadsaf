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
) -> dict[str, Any]:
    config = build_marl_config(
        {
            "seed": seed,
            "eval_episodes": eval_episodes,
            "num_uavs": num_uavs,
            "assignment_rule": assignment_rule,
        }
    )
    return run_evaluation(config, model_path=model_path)
