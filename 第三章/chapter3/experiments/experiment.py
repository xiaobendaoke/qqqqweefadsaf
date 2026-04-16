"""第三章实验主流程模块。

该模块负责组织第三章单 UAV 实验的完整运行流程，
包括环境创建、策略执行、episode 结果汇总、轨迹导出以及与第四章退化一致性的对比实验。

输入输出与关键参数：
实验入口支持随机种子、episode 数、hard/default 场景配置、策略类型、
轨迹导出开关和每回合步数等参数；输出为包含平均指标、episode 日志和轨迹文件路径的结果字典。
"""

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
    """在仓库内定位 `chapter4` 包目录，供退化一致性实验复用。"""
    candidates = sorted(
        path.parent for path in search_root.rglob("__init__.py") if path.parent.name == "chapter4"
    )
    if not candidates:
        raise FileNotFoundError(f"Could not find a 'chapter4' package under {search_root}")
    return candidates[0]


def _ensure_chapter4_package_loaded(repo_root: Path) -> None:
    """确保第四章包能被当前进程导入，而不依赖外部安装。"""
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

    # 这里显式构造 module spec，是为了在论文仓库的本地目录结构下稳定导入 chapter4。
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
    """对非空数值列表求均值，空列表保留为 None。"""
    if not values:
        return None
    return sum(values) / len(values)


def _build_profile_overrides(*, hard: bool, steps_per_episode: int | None = None) -> dict[str, Any]:
    """构造 default/hard 场景的统一覆盖配置。"""
    overrides: dict[str, Any] = {}
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
    if steps_per_episode is not None:
        overrides["steps_per_episode"] = int(steps_per_episode)
    return overrides


def run_experiment(
    *,
    seed: int,
    episodes: int,
    hard: bool,
    policy: str = "heuristic",
    export_trajectory: bool = True,
    steps_per_episode: int | None = None,
) -> dict[str, Any]:
    """运行第三章实验并写出结果文件。

    参数：
        seed: 实验起始随机种子。
        episodes: 需要运行的 episode 数量。
        hard: 是否启用更高负载、更严格约束的 hard 场景。
        policy: 采用的策略名称，可选 heuristic、mpc、fixed_point、fixed_patrol。
        export_trajectory: 是否导出 UAV/UE 轨迹 JSON 与 PNG。
        steps_per_episode: 可选的每回合步数覆盖值。

    返回：
        包含平均指标、每回合摘要、episode 日志和轨迹导出信息的结果字典。
    """
    overrides = _build_profile_overrides(hard=hard, steps_per_episode=steps_per_episode)
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
            actions = policy_fn(observations, env)
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


def compare_with_chapter4(
    *,
    seed: int,
    episodes: int,
    hard: bool = False,
    steps_per_episode: int | None = None,
) -> dict[str, Any]:
    """比较第三章与第四章在单 UAV 条件下的结果一致性。

    参数：
        seed: 实验起始随机种子。
        episodes: 用于比较的 episode 数量。
        hard: 是否按 hard 场景覆盖配置运行比较。
        steps_per_episode: 可选的步数覆盖值；若提供，则第三章与第四章都按相同步数运行。

    返回：
        包含第三章指标、第四章指标及其逐项差值的比较结果字典。
    """
    _ensure_chapter4_package_loaded(REPO_ROOT)
    importlib.invalidate_caches()
    chapter4_env_module = importlib.import_module("chapter4.env")
    Chapter4Env = chapter4_env_module.Chapter4Env

    chapter3_overrides = _build_profile_overrides(hard=hard, steps_per_episode=steps_per_episode)
    chapter4_overrides = {"num_uavs": 1, **chapter3_overrides}

    # 两侧都使用共享 heuristic 和相同随机种子，尽量把差异约束在环境实现本身。
    left = run_short_experiment(
        env_factory=Chapter3Env,
        policy_fn=select_actions_heuristic,
        overrides=chapter3_overrides,
        episodes=episodes,
        seed=seed,
    )
    right = run_short_experiment(
        env_factory=Chapter4Env,
        policy_fn=select_actions_heuristic,
        overrides=chapter4_overrides,
        episodes=episodes,
        seed=seed,
    )
    comparison = {
        "seed": seed,
        "episodes": episodes,
        "validation_policy": "chapter3_heuristic_shared",
        "chapter3_metrics": left["averaged_metrics"],
        "chapter4_metrics": right["averaged_metrics"],
        "comparison": compare_metric_dicts(left["averaged_metrics"], right["averaged_metrics"]),
    }
    write_json(CHAPTER3_RESULTS / "experiment_compare_ch3_ch4.json", comparison)
    return comparison
