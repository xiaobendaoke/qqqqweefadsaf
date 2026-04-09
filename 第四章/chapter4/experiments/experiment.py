from __future__ import annotations

from pathlib import Path
from typing import Any

from common.uav_mec.logging_utils import write_json
from common.uav_mec.simulation import run_short_experiment

from ..env import Chapter4Env
from ..policies.mobility_heuristic_multi import select_actions

CHAPTER4_RESULTS = Path(__file__).resolve().parents[2] / "results"


def run_experiment(*, seed: int, episodes: int, hard: bool, num_uavs: int, assignment_rule: str) -> dict[str, Any]:
    overrides = {"num_uavs": num_uavs, "assignment_rule": assignment_rule}
    if hard:
        overrides.update(
            {
                "num_users": 10,
                "steps_per_episode": 10,
                "task_arrival_rate": 0.95,
                "task_input_size_range_bits": (3.0e6, 6.0e6),
                "task_cpu_cycles_range": (1.5e9, 4.5e9),
                "task_slack_range": (0.35, 1.1),
                "required_reliability": 0.999,
                "bandwidth_edge_hz": 2.0e6,
                "bandwidth_backhaul_hz": 4.0e6,
                "bs_compute_hz": 2.0e9,
                "uav_compute_hz": 9.0e9,
                "uav_service_cache_capacity": 2,
            }
        )
    elif assignment_rule in {"nearest_uav", "least_loaded_uav"} and num_uavs > 1:
        overrides.update({})
    result = run_short_experiment(
        env_factory=Chapter4Env,
        policy_fn=select_actions,
        overrides=overrides,
        episodes=episodes,
        seed=seed,
    )
    result["chapter"] = "chapter4"
    result["profile"] = "hard" if hard else "default"
    result["assignment_rule"] = assignment_rule
    output_name = (
        f"experiment_hard_u{num_uavs}_{assignment_rule}.json"
        if hard
        else f"experiment_short_u{num_uavs}_{assignment_rule}.json"
    )
    write_json(CHAPTER4_RESULTS / output_name, result)
    return result


def run_sensitive_experiment(*, seed: int, episodes: int, num_uavs: int, assignment_rule: str) -> dict[str, Any]:
    fixed_uavs = (
        (130.0, 200.0),
        (220.0, 200.0),
        (330.0, 320.0),
    )
    fixed_users = (
        (150.0, 190.0),
        (155.0, 195.0),
        (160.0, 205.0),
        (165.0, 198.0),
        (170.0, 202.0),
        (175.0, 206.0),
        (180.0, 200.0),
        (185.0, 203.0),
        (260.0, 250.0),
        (270.0, 255.0),
    )
    overrides = {
        "num_uavs": num_uavs,
        "assignment_rule": assignment_rule,
        "num_users": len(fixed_users),
        "steps_per_episode": 6,
        "task_arrival_rate": 1.0,
        "task_input_size_range_bits": (2.0e6, 3.0e6),
        "task_cpu_cycles_range": (1.0e9, 1.8e9),
        "task_slack_range": (1.4, 2.0),
        "required_reliability": 0.90,
        "bandwidth_edge_hz": 1.2e6,
        "bandwidth_backhaul_hz": 8.0e6,
        "uav_compute_hz": 4.0e9,
        "bs_compute_hz": 5.0e9,
        "uav_service_cache_capacity": 2,
        "fixed_uav_positions": fixed_uavs[:num_uavs],
        "fixed_user_positions": fixed_users,
        "observation_max_neighbors": max(0, num_uavs - 1),
    }
    result = run_short_experiment(
        env_factory=Chapter4Env,
        policy_fn=select_actions,
        overrides=overrides,
        episodes=episodes,
        seed=seed,
    )
    result["chapter"] = "chapter4"
    result["profile"] = "sensitive"
    result["assignment_rule"] = assignment_rule
    write_json(CHAPTER4_RESULTS / f"experiment_sensitive_u{num_uavs}_{assignment_rule}.json", result)
    return result
