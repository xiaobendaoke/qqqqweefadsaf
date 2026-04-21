"""第四章启发式实验模块。

该模块现在显式区分两类 baseline：

- `legacy_mobility_only`: 只输出 mobility，卸载/缓存仍由旧环境路径决定
- `joint_heuristic`: 同时输出 mobility/offloading/caching 的联合启发式策略
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Callable

from common.uav_mec.logging_utils import write_json
from common.uav_mec.simulation import run_short_experiment

from ..env import Chapter4Env
from ..policies.joint_heuristic_multi import POLICY_LABEL as JOINT_HEURISTIC_LABEL
from ..policies.joint_heuristic_multi import select_actions as select_joint_actions
from ..policies.mobility_heuristic_multi import POLICY_LABEL as LEGACY_POLICY_LABEL
from ..policies.mobility_heuristic_multi import select_actions as select_legacy_mobility_actions
from ..results_paths import baseline_results_dir

PolicyFn = Callable[..., list[Any]]

POLICY_REGISTRY: dict[str, dict[str, Any]] = {
    "legacy_mobility_only": {
        "label": LEGACY_POLICY_LABEL,
        "family": "legacy_baseline",
        "scheduler_mode": "legacy_heuristic",
        "policy_fn": select_legacy_mobility_actions,
    },
    "joint_heuristic": {
        "label": JOINT_HEURISTIC_LABEL,
        "family": "joint_baseline",
        "scheduler_mode": "joint_action",
        "policy_fn": select_joint_actions,
    },
}


def recommended_experiment_matrix() -> dict[str, list[dict[str, Any]]]:
    """返回推荐实验矩阵，供 CLI 和论文脚本复用。"""
    return {
        "main_tracks": [
            {
                "track": "mobility_only_rl",
                "description": "legacy RL baseline: only mobility is learned; environment keeps heuristic offloading/cache",
                "trainer_mode": "legacy_mobility_only",
                "baseline_policy_id": "legacy_mobility_only",
            },
            {
                "track": "joint_heuristic",
                "description": "new joint heuristic baseline: mobility + offloading + caching are all chosen heuristically",
                "policy_id": "joint_heuristic",
            },
            {
                "track": "joint_rl",
                "description": "new joint RL policy: hybrid MAPPO/PPO with joint actor and centralized critic",
                "trainer_mode": "hybrid_joint",
                "baseline_policy_id": "joint_heuristic",
            },
        ],
        "ablations": [
            {
                "track": "joint_rl_no_cache",
                "description": "joint RL with cache branch masked to zero to isolate caching contribution",
            },
            {
                "track": "joint_rl_no_offloading",
                "description": "joint RL with offloading branch forced to defer to isolate offloading contribution",
            },
            {
                "track": "joint_rl_no_joint_critic_summary",
                "description": "joint RL with reduced critic summary to measure centralized critic context value",
            },
            {
                "track": "joint_rl_vs_legacy_baseline",
                "description": "fairness sanity check: report joint RL against both joint heuristic and legacy mobility-only baseline",
            },
        ],
    }


def _resolve_policy(policy_id: str) -> dict[str, Any]:
    if policy_id not in POLICY_REGISTRY:
        raise KeyError(f"Unknown Chapter 4 policy_id: {policy_id}")
    return POLICY_REGISTRY[policy_id]


def _run_policy_experiment(
    *,
    policy_id: str,
    seed: int,
    episodes: int,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    policy_spec = _resolve_policy(policy_id)
    result = run_short_experiment(
        env_factory=Chapter4Env,
        policy_fn=policy_spec["policy_fn"],
        overrides=overrides,
        episodes=episodes,
        seed=seed,
    )
    result["policy_id"] = policy_id
    result["policy_label"] = policy_spec["label"]
    result["policy_family"] = policy_spec["family"]
    result["scheduler_mode"] = policy_spec["scheduler_mode"]
    return result


def run_experiment(
    *,
    seed: int,
    episodes: int,
    hard: bool,
    num_uavs: int,
    assignment_rule: str,
    policy_id: str = "legacy_mobility_only",
) -> dict[str, Any]:
    """运行第四章主实验，可显式选择 baseline 口径。"""
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
    result = _run_policy_experiment(policy_id=policy_id, seed=seed, episodes=episodes, overrides=overrides)
    result["chapter"] = "chapter4"
    result["profile"] = "hard" if hard else "default"
    result["assignment_rule"] = assignment_rule
    output_name = f"experiment_{result['profile']}_{policy_id}_u{num_uavs}_{assignment_rule}.json"
    write_json(baseline_results_dir(policy_id) / output_name, result)
    return result


def run_sensitive_experiment(
    *,
    seed: int,
    episodes: int,
    num_uavs: int,
    assignment_rule: str,
    policy_id: str = "legacy_mobility_only",
) -> dict[str, Any]:
    """运行第四章固定布局 sensitive 实验，可切换 baseline 口径。"""
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
    }
    result = _run_policy_experiment(policy_id=policy_id, seed=seed, episodes=episodes, overrides=overrides)
    result["chapter"] = "chapter4"
    result["profile"] = "sensitive"
    result["assignment_rule"] = assignment_rule
    write_json(baseline_results_dir(policy_id) / f"experiment_sensitive_{policy_id}_u{num_uavs}_{assignment_rule}.json", result)
    return result
