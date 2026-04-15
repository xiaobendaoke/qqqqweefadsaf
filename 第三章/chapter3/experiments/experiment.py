from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from common.uav_mec.logging_utils import write_json
from common.uav_mec.simulation import compare_metric_dicts, run_short_experiment

from ..env import Chapter3Env
from ..policies.fixed_patrol import select_actions as select_actions_fixed_patrol
from ..policies.fixed_point import select_actions as select_actions_fixed_point
from ..policies.mobility_heuristic import select_actions as select_actions_heuristic
from ..policies.mpc_shell import select_actions as select_actions_mpc
from .trajectory import EpisodeTrajectoryRecorder, export_trajectory_artifacts

REPO_ROOT = Path(__file__).resolve().parents[3]
CHAPTER3_ROOT = Path(__file__).resolve().parents[2]
CHAPTER3_RESULTS = CHAPTER3_ROOT / "results"
CHAPTER3_TRAJECTORIES = CHAPTER3_RESULTS / "trajectories"


def _find_chapter4_package_dir(search_root: Path) -> Path:
    candidates = sorted(
        path.parent for path in search_root.rglob("__init__.py") if path.parent.name == "chapter4"
    )
    if not candidates:
        raise FileNotFoundError(f"Could not find a 'chapter4' package under {search_root}")
    return candidates[0]


def _ensure_chapter4_package_loaded(repo_root: Path) -> None:
    package_dir = _find_chapter4_package_dir(repo_root)
    package_root = package_dir.parent
    init_file = package_dir / "__init__.py"

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    existing = sys.modules.get("chapter4")
    if existing is not None:
        module_file = getattr(existing, "__file__", "")
        if module_file and Path(module_file).resolve().parent == package_dir:
            return

    spec = importlib.util.spec_from_file_location(
        "chapter4",
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for chapter4 from {init_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["chapter4"] = module
    spec.loader.exec_module(module)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def run_experiment(
    *,
    seed: int,
    episodes: int,
    hard: bool,
    policy: str = "heuristic",
    export_trajectory: bool = True,
    steps_per_episode: int | None = None,
) -> dict[str, Any]:
    overrides = {}
    if hard:
        overrides = {
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
    if steps_per_episode is not None:
        overrides["steps_per_episode"] = int(steps_per_episode)
    if policy not in {"heuristic", "mpc", "fixed_point", "fixed_patrol"}:
        raise ValueError(f"Unsupported Chapter 3 policy: {policy}")
    policy_map = {
        "heuristic": select_actions_heuristic,
        "mpc": select_actions_mpc,
        "fixed_point": select_actions_fixed_point,
        "fixed_patrol": select_actions_fixed_patrol,
    }
    policy_fn = policy_map[policy]
    episode_summaries: list[dict[str, Any]] = []
    episode_logs: list[dict[str, Any]] = []
    aggregate_metrics: list[dict[str, float | None]] = []
    trajectory_exports: list[dict[str, Any]] = []
    log_schema: dict[str, Any] | None = None
    base_profile = "hard" if hard else "default"
    profile = f"{base_profile}_s{int(steps_per_episode)}" if steps_per_episode is not None else base_profile

    for episode_idx in range(episodes):
        env = Chapter3Env(overrides)
        episode_seed = seed + episode_idx
        reset_result = env.reset(seed=episode_seed)
        log_schema = env.get_episode_log_schema()
        observations = reset_result["observations"]
        recorder = (
            EpisodeTrajectoryRecorder(
                env=env,
                episode_index=episode_idx,
                seed=episode_seed,
                policy=policy,
                profile=profile,
            )
            if export_trajectory
            else None
        )
        last_step = None
        step_index = 0
        while True:
            try:
                actions = policy_fn(observations, env)
            except TypeError:
                actions = policy_fn(observations)
            last_step = env.step(actions)
            step_index += 1
            if recorder is not None:
                recorder.record_step(step_index=step_index, metrics=last_step["step_metrics"])
            observations = last_step["observations"]
            if last_step["terminated"] or last_step["truncated"]:
                break

        summary = env.export_episode_summary()
        episode_log = env.export_episode_log(episode_index=episode_idx, seed=episode_seed)
        episode_summaries.append(
            {
                "episode": episode_idx,
                "seed": episode_seed,
                "metrics": summary["metrics"],
                "last_info": last_step["info"] if last_step else {},
            }
        )
        episode_logs.append(episode_log)
        aggregate_metrics.append(summary["metrics"])
        if recorder is not None:
            exported = export_trajectory_artifacts(
                recorder=recorder,
                summary_metrics=summary["metrics"],
                output_dir=CHAPTER3_TRAJECTORIES,
            )
            trajectory_exports.append(
                {
                    "episode": episode_idx,
                    "seed": episode_seed,
                    "policy": policy,
                    "profile": profile,
                    **exported,
                }
            )

    metric_keys = aggregate_metrics[0].keys() if aggregate_metrics else []
    averaged_metrics: dict[str, float | None] = {}
    for key in metric_keys:
        numeric_values = [metrics[key] for metrics in aggregate_metrics if metrics[key] is not None]
        averaged_metrics[key] = _mean([float(value) for value in numeric_values]) if numeric_values else None

    result = {
        "episodes": episodes,
        "seed": seed,
        "overrides": overrides,
        "averaged_metrics": averaged_metrics,
        "episode_summaries": episode_summaries,
        "episode_log_schema": log_schema,
        "episode_logs": episode_logs,
        "trajectory_exports": trajectory_exports,
    }
    result["chapter"] = "chapter3"
    result["profile"] = profile
    result["policy"] = policy
    if policy == "heuristic" and steps_per_episode is None:
        output_name = "experiment_hard.json" if hard else "experiment_short.json"
    elif policy == "mpc" and steps_per_episode is None:
        output_name = "experiment_hard_mpc.json" if hard else "experiment_short_mpc.json"
    else:
        step_suffix = f"_s{int(steps_per_episode)}" if steps_per_episode is not None else ""
        output_name = f"experiment_{base_profile}_{policy}{step_suffix}.json"
    write_json(CHAPTER3_RESULTS / output_name, result)
    return result


def compare_with_chapter4(*, seed: int, episodes: int) -> dict[str, Any]:
    _ensure_chapter4_package_loaded(REPO_ROOT)
    importlib.invalidate_caches()
    chapter4_env_module = importlib.import_module("chapter4.env")
    chapter4_policy_module = importlib.import_module("chapter4.policies.mobility_heuristic_multi")
    Chapter4Env = chapter4_env_module.Chapter4Env
    select_actions_ch4 = chapter4_policy_module.select_actions

    left = run_short_experiment(
        env_factory=Chapter3Env,
        policy_fn=select_actions_heuristic,
        overrides={},
        episodes=episodes,
        seed=seed,
    )
    right = run_short_experiment(
        env_factory=Chapter4Env,
        policy_fn=select_actions_ch4,
        overrides={"num_uavs": 1},
        episodes=episodes,
        seed=seed,
    )
    comparison = {
        "seed": seed,
        "episodes": episodes,
        "chapter3_metrics": left["averaged_metrics"],
        "chapter4_metrics": right["averaged_metrics"],
        "comparison": compare_metric_dicts(left["averaged_metrics"], right["averaged_metrics"]),
    }
    write_json(CHAPTER3_RESULTS / "experiment_compare_ch3_ch4.json", comparison)
    return comparison
