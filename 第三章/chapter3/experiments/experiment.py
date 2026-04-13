from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from common.uav_mec.logging_utils import write_json
from common.uav_mec.simulation import compare_metric_dicts, run_short_experiment

from ..env import Chapter3Env
from ..policies.mobility_heuristic import select_actions as select_actions_heuristic
from ..policies.mpc_shell import select_actions as select_actions_mpc

REPO_ROOT = Path(__file__).resolve().parents[3]
CHAPTER3_ROOT = Path(__file__).resolve().parents[2]
CHAPTER3_RESULTS = CHAPTER3_ROOT / "results"


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


def run_experiment(*, seed: int, episodes: int, hard: bool, policy: str = "heuristic") -> dict[str, Any]:
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
    if policy not in {"heuristic", "mpc"}:
        raise ValueError(f"Unsupported Chapter 3 policy: {policy}")
    policy_fn = select_actions_heuristic if policy == "heuristic" else select_actions_mpc
    result = run_short_experiment(
        env_factory=Chapter3Env,
        policy_fn=policy_fn,
        overrides=overrides,
        episodes=episodes,
        seed=seed,
    )
    result["chapter"] = "chapter3"
    result["profile"] = "hard" if hard else "default"
    result["policy"] = policy
    if policy == "mpc":
        output_name = "experiment_hard_mpc.json" if hard else "experiment_short_mpc.json"
    else:
        output_name = "experiment_hard.json" if hard else "experiment_short.json"
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
