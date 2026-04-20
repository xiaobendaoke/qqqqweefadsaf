"""第四章终稿复现实验打包模块。

该模块负责组织 stage-6 多随机种子复现实验，
生成终稿统计表、图表和可复现性结果包。
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from common.uav_mec.logging_utils import write_json
from common.uav_mec.plot_i18n import (
    ENERGY_COMPONENT_LABEL_CN,
    assignment_rule_label,
    configure_matplotlib_for_chinese,
    metric_label,
    method_label,
    variant_label,
)

from ..experiments import run_sensitive_experiment
from .eval import run_marl_evaluation
from .train import run_marl_training


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
FINAL_DIR = RESULTS_DIR / "paper_stage6"
TABLES_DIR = FINAL_DIR / "tables"
FIGURES_DIR = FINAL_DIR / "figures"
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
CHAPTER3_DIR = WORKSPACE_ROOT / "第三章"

DEFAULT_SEEDS = [42, 52, 62]
FINAL_MAIN_CONFIG: dict[str, Any] = {
    "train_episodes": 240,
    "actor_lr": 8.0e-5,
    "critic_lr": 5.0e-4,
    "ppo_clip_eps": 0.10,
    "entropy_coef": 0.0003,
    "value_loss_coef": 0.82,
    "reward_energy_weight": 0.00,
    "reward_action_magnitude_weight": 0.00,
    "action_std_init": 0.04,
    "action_std_min": 0.005,
    "action_std_decay": 0.984,
    "use_movement_budget": True,
}
MAIN_SETTINGS = [
    {"num_uavs": 2, "assignment_rule": "nearest_uav"},
    {"num_uavs": 3, "assignment_rule": "nearest_uav"},
]
ABLATION_SETTINGS = [
    {
        "label": "main",
        "description": "fixed freeze_noshaping_240 PPO configuration",
        "overrides": dict(FINAL_MAIN_CONFIG),
    },
    {
        "label": "with_reward_shaping",
        "description": "reward_energy_weight=2.0 and reward_action_magnitude_weight=1.0",
        "overrides": {
            **FINAL_MAIN_CONFIG,
            "reward_energy_weight": 2.0,
            "reward_action_magnitude_weight": 1.0,
        },
    },
    {
        "label": "no_movement_budget",
        "description": "use_movement_budget=False with all other settings fixed",
        "overrides": {
            **FINAL_MAIN_CONFIG,
            "use_movement_budget": False,
        },
    },
]

ENERGY_COMPONENTS = [
    "uav_move_energy",
    "uav_compute_energy",
    "ue_local_energy",
    "ue_uplink_energy",
    "bs_compute_energy",
    "relay_fetch_energy",
]
BASE_METRICS = [
    "completion_rate",
    "average_latency",
    "average_latency_completed",
    "latency_per_generated_task",
    "total_energy",
    "cache_hit_rate",
    "fairness_user_completion",
    "fairness_uav_load",
    "deadline_violation_rate",
    "reliability_violation_rate",
    *ENERGY_COMPONENTS,
]


def _load_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for stage-6 final plots. Install dependencies from 第四章/requirements.txt first."
        ) from exc
    return configure_matplotlib_for_chinese(plt)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _metric_stats(values: list[float]) -> tuple[float, float]:
    mean = float(statistics.fmean(values))
    std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    return mean, std


def _format_mean_std(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def _extract_prefixed_metrics(metrics: dict[str, Any], *, prefix: str) -> dict[str, float]:
    payload: dict[str, float] = {}
    for name in BASE_METRICS:
        value = metrics.get(name)
        if value is None:
            continue
        key = f"{prefix}_{name}" if prefix else name
        payload[key] = float(value)
    return payload


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    group_keys: list[str],
    metrics: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict[str, Any]] = []
    for key, group in grouped.items():
        entry = {group_key: value for group_key, value in zip(group_keys, key)}
        entry["num_seeds"] = len(group)
        for metric in metrics:
            values = [float(item[metric]) for item in group if item.get(metric) is not None]
            if not values:
                entry[f"{metric}_mean"] = None
                entry[f"{metric}_std"] = None
                entry[f"{metric}_mean_std"] = "null"
                continue
            mean, std = _metric_stats(values)
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_std"] = std
            entry[f"{metric}_mean_std"] = _format_mean_std(mean, std)
        aggregated.append(entry)
    aggregated.sort(key=lambda item: tuple(item[key] for key in group_keys))
    return aggregated


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    headers = [label for label, _ in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row[key]) for _, key in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _ensure_chapter3_import() -> None:
    if str(CHAPTER3_DIR) not in sys.path:
        sys.path.insert(0, str(CHAPTER3_DIR))


def _run_train_eval(
    *,
    seed: int,
    eval_seed: int,
    num_uavs: int,
    assignment_rule: str,
    output_tag: str,
    eval_episodes: int,
    overrides: dict[str, Any],
    device: str = "auto",
) -> dict[str, Any]:
    train_episodes = int(overrides["train_episodes"])
    train_payload = run_marl_training(
        seed=seed,
        train_episodes=train_episodes,
        num_uavs=num_uavs,
        assignment_rule=assignment_rule,
        overrides={**overrides, "output_tag": output_tag, "device": device},
    )
    eval_payload = run_marl_evaluation(
        seed=eval_seed,
        eval_episodes=eval_episodes,
        num_uavs=num_uavs,
        assignment_rule=assignment_rule,
        model_path=train_payload["checkpoint_path"],
        overrides={
            "output_tag": output_tag,
            "use_movement_budget": bool(overrides.get("use_movement_budget", True)),
            "device": device,
        },
    )
    return {"train": train_payload, "eval": eval_payload}


def _run_compare_ch4(seeds: list[int], *, episodes: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _ensure_chapter3_import()
    from chapter3.experiments import compare_with_chapter4

    raw_rows: list[dict[str, Any]] = []
    for seed in seeds:
        comparison = compare_with_chapter4(seed=seed, episodes=episodes)["comparison"]
        for metric, payload in comparison.items():
            raw_rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "delta": None if payload["delta"] is None else float(payload["delta"]),
                }
            )
    aggregated = _aggregate_rows(raw_rows, group_keys=["metric"], metrics=["delta"])
    return raw_rows, aggregated


def _run_assignment_multiseed(seeds: list[int], *, eval_episodes: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for num_uavs in (2, 3):
            for assignment_rule in ("nearest_uav", "least_loaded_uav"):
                result = run_sensitive_experiment(
                    seed=seed,
                    episodes=eval_episodes,
                    num_uavs=num_uavs,
                    assignment_rule=assignment_rule,
                )
                metrics = result["averaged_metrics"]
                raw_rows.append(
                    {
                        "seed": seed,
                        "profile": "sensitive",
                        "num_uavs": num_uavs,
                        "assignment_rule": assignment_rule,
                        **_extract_prefixed_metrics(metrics, prefix=""),
                    }
                )
    aggregated = _aggregate_rows(
        raw_rows,
        group_keys=["profile", "num_uavs", "assignment_rule"],
        metrics=BASE_METRICS,
    )
    return raw_rows, aggregated


def _run_main_multiseed(
    seeds: list[int],
    *,
    eval_episodes: int,
    train_episodes: int,
    device: str = "auto",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[list[dict[str, Any]]]]]:
    raw_rows: list[dict[str, Any]] = []
    training_logs: dict[str, list[list[dict[str, Any]]]] = {"main": []}
    for seed in seeds:
        eval_seed = seed + 100
        for setting in MAIN_SETTINGS:
            num_uavs = int(setting["num_uavs"])
            assignment_rule = str(setting["assignment_rule"])
            output_tag = f"final_main_s{seed}_u{num_uavs}"
            train_eval = _run_train_eval(
                seed=seed,
                eval_seed=eval_seed,
                num_uavs=num_uavs,
                assignment_rule=assignment_rule,
                output_tag=output_tag,
                eval_episodes=eval_episodes,
                overrides={**dict(FINAL_MAIN_CONFIG), "train_episodes": int(train_episodes)},
                device=device,
            )
            marl_metrics = train_eval["eval"]["marl_metrics"]
            heuristic_metrics = train_eval["eval"]["heuristic_metrics"]
            raw_rows.append(
                {
                    "seed": seed,
                    "eval_seed": eval_seed,
                    "num_uavs": num_uavs,
                    "assignment_rule": assignment_rule,
                    **_extract_prefixed_metrics(marl_metrics, prefix="ppo"),
                    **_extract_prefixed_metrics(heuristic_metrics, prefix="heuristic"),
                    "delta_completion_rate": float(marl_metrics["completion_rate"]) - float(heuristic_metrics["completion_rate"]),
                    "delta_average_latency": float(marl_metrics["average_latency"]) - float(heuristic_metrics["average_latency"]),
                    "delta_total_energy": float(marl_metrics["total_energy"]) - float(heuristic_metrics["total_energy"]),
                    "checkpoint_path": train_eval["train"]["checkpoint_path"],
                    "train_log_path": train_eval["train"]["train_log_path"],
                    "eval_path": train_eval["eval"]["eval_path"],
                }
            )
            if num_uavs == 2:
                training_logs["main"].append(train_eval["train"]["training_log"])
    aggregated = _aggregate_rows(
        raw_rows,
        group_keys=["num_uavs", "assignment_rule"],
        metrics=[
            *[f"ppo_{metric}" for metric in BASE_METRICS],
            *[f"heuristic_{metric}" for metric in BASE_METRICS],
            "delta_completion_rate",
            "delta_average_latency",
            "delta_total_energy",
        ],
    )
    return raw_rows, aggregated, training_logs


def _run_ablation_multiseed(
    seeds: list[int],
    *,
    eval_episodes: int,
    train_episodes: int,
    device: str = "auto",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[list[dict[str, Any]]]]]:
    raw_rows: list[dict[str, Any]] = []
    training_logs: dict[str, list[list[dict[str, Any]]]] = {}
    for seed in seeds:
        eval_seed = seed + 100
        for ablation in ABLATION_SETTINGS:
            label = str(ablation["label"])
            output_tag = f"final_{label}_s{seed}"
            train_eval = _run_train_eval(
                seed=seed,
                eval_seed=eval_seed,
                num_uavs=2,
                assignment_rule="nearest_uav",
                output_tag=output_tag,
                eval_episodes=eval_episodes,
                overrides={**dict(ablation["overrides"]), "train_episodes": int(train_episodes)},
                device=device,
            )
            marl_metrics = train_eval["eval"]["marl_metrics"]
            heuristic_metrics = train_eval["eval"]["heuristic_metrics"]
            raw_rows.append(
                {
                    "seed": seed,
                    "eval_seed": eval_seed,
                    "variant": label,
                    "description": str(ablation["description"]),
                    **_extract_prefixed_metrics(marl_metrics, prefix=""),
                    "heuristic_total_energy": float(heuristic_metrics["total_energy"]),
                    "delta_total_energy": float(marl_metrics["total_energy"]) - float(heuristic_metrics["total_energy"]),
                    "checkpoint_path": train_eval["train"]["checkpoint_path"],
                    "train_log_path": train_eval["train"]["train_log_path"],
                    "eval_path": train_eval["eval"]["eval_path"],
                }
            )
            training_logs.setdefault(label, []).append(train_eval["train"]["training_log"])
    aggregated = _aggregate_rows(
        raw_rows,
        group_keys=["variant", "description"],
        metrics=[
            *BASE_METRICS,
            "heuristic_total_energy",
            "delta_total_energy",
        ],
    )
    return raw_rows, aggregated, training_logs


def _aggregate_training_curve(logs: list[list[dict[str, Any]]], metric: str) -> tuple[list[int], list[float], list[float]]:
    episodes = [int(entry["episode"]) for entry in logs[0]]
    means: list[float] = []
    stds: list[float] = []
    for index in range(len(episodes)):
        values = [float(log[index][metric]) for log in logs]
        mean, std = _metric_stats(values)
        means.append(mean)
        stds.append(std)
    return episodes, means, stds


def _style_axis(axis: Any) -> None:
    axis.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _plot_training_metric_curves(
    training_logs: dict[str, list[list[dict[str, Any]]]],
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt = _load_matplotlib()
    figure, axis = plt.subplots(figsize=(8.8, 5.0))
    colors = {
        "main": "#2E86AB",
        "with_reward_shaping": "#E67E22",
        "no_movement_budget": "#C0392B",
    }
    label_map = {
        "main": variant_label("main"),
        "with_reward_shaping": variant_label("with_reward_shaping"),
        "no_movement_budget": variant_label("no_movement_budget"),
    }
    for variant, logs in training_logs.items():
        episodes, means, stds = _aggregate_training_curve(logs, metric)
        color = colors.get(variant, "#555555")
        axis.plot(episodes, means, label=label_map.get(variant, variant), color=color, linewidth=2.2)
        axis.fill_between(
            episodes,
            [value - delta for value, delta in zip(means, stds)],
            [value + delta for value, delta in zip(means, stds)],
            color=color,
            alpha=0.14,
        )
    axis.set_xlabel("训练回合")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    _style_axis(axis)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_training_behavior_overview(
    training_logs: dict[str, list[list[dict[str, Any]]]],
    *,
    output_path: Path,
) -> None:
    plt = _load_matplotlib()
    figure, axes = plt.subplots(2, 3, figsize=(13.6, 7.8), sharex=True)
    colors = {
        "main": "#2E86AB",
        "with_reward_shaping": "#E67E22",
        "no_movement_budget": "#C0392B",
    }
    label_map = {
        "main": variant_label("main"),
        "with_reward_shaping": variant_label("with_reward_shaping"),
        "no_movement_budget": variant_label("no_movement_budget"),
    }
    panels = [
        ("team_return", "团队回报"),
        ("completion_rate", "任务完成率"),
        ("average_latency", "平均时延"),
        ("total_energy", "总能耗"),
        ("mean_step_action_magnitude", "平均动作幅度"),
        ("mean_step_energy_norm", "平均归一化步能耗"),
    ]

    for axis, (metric, title) in zip(axes.flat, panels):
        for variant, logs in training_logs.items():
            episodes, means, stds = _aggregate_training_curve(logs, metric)
            color = colors.get(variant, "#555555")
            axis.plot(episodes, means, label=label_map.get(variant, variant), color=color, linewidth=2.0)
            axis.fill_between(
                episodes,
                [value - delta for value, delta in zip(means, stds)],
                [value + delta for value, delta in zip(means, stds)],
                color=color,
                alpha=0.14,
            )
        axis.set_title(title)
        axis.set_xlabel("训练回合")
        _style_axis(axis)
        if "rate" in metric:
            axis.set_ylim(0.0, 1.05)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("训练行为总览（3 个种子的均值 ± 标准差）", y=0.98)
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_compare_ch4_delta(compare_rows: list[dict[str, Any]], output_path: Path) -> None:
    plt = _load_matplotlib()
    filtered = [row for row in compare_rows if row.get("delta_mean") is not None]
    labels = [metric_label(str(row["metric"])) for row in filtered]
    values = [float(row["delta_mean"]) for row in filtered]
    colors = ["#2A9D8F" if value < 0 else "#E76F51" if value > 0 else "#7A7A7A" for value in values]
    positions = list(range(len(labels)))
    max_abs = max([abs(value) for value in values], default=0.0)
    if max_abs <= 1.0e-12:
        max_abs = 1.0e-6

    figure, axis = plt.subplots(figsize=(9.8, 6.4))
    axis.barh(positions, values, color=colors, alpha=0.9)
    axis.axvline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    axis.set_yticks(positions, labels)
    axis.set_xlabel("第三章 - 第四章 指标差值")
    axis.set_title("第三章与第四章（无人机数量=1）差值校验")
    axis.set_xlim(-1.15 * max_abs, 1.15 * max_abs)
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_energy_breakdown(main_rows: list[dict[str, Any]], output_path: Path) -> None:
    plt = _load_matplotlib()
    figure, axes = plt.subplots(1, len(main_rows), figsize=(12.5, 4.8), sharey=True)
    if len(main_rows) == 1:
        axes = [axes]
    colors = {
        "uav_move_energy": "#457B9D",
        "uav_compute_energy": "#1D3557",
        "ue_local_energy": "#E9C46A",
        "ue_uplink_energy": "#F4A261",
        "bs_compute_energy": "#E76F51",
        "relay_fetch_energy": "#2A9D8F",
    }
    label_map = {
        "uav_move_energy": ENERGY_COMPONENT_LABEL_CN["uav_move_energy"],
        "uav_compute_energy": ENERGY_COMPONENT_LABEL_CN["uav_compute_energy"],
        "ue_local_energy": ENERGY_COMPONENT_LABEL_CN["ue_local_energy"],
        "ue_uplink_energy": ENERGY_COMPONENT_LABEL_CN["ue_uplink_energy"],
        "bs_compute_energy": ENERGY_COMPONENT_LABEL_CN["bs_compute_energy"],
        "relay_fetch_energy": ENERGY_COMPONENT_LABEL_CN["relay_fetch_energy"],
    }

    for axis, row in zip(axes, main_rows):
        labels = [method_label("ppo"), method_label("heuristic")]
        positions = [0, 1]
        bottoms = [0.0, 0.0]
        for component in ENERGY_COMPONENTS:
            values = [row[f"ppo_{component}_mean"], row[f"heuristic_{component}_mean"]]
            axis.bar(
                positions,
                values,
                bottom=bottoms,
                color=colors[component],
                width=0.55,
                label=label_map[component],
            )
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        axis.set_title(f"{int(row['num_uavs'])} 架无人机")
        axis.set_xticks(positions, labels)
        axis.set_ylabel("能耗")
        _style_axis(axis)

    handles = [axes[0].containers[index] for index in range(len(ENERGY_COMPONENTS))]
    labels = [label_map[component] for component in ENERGY_COMPONENTS]
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("最终配置下的能耗分解", y=0.98)
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.94))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _collect_per_uav_diagnostics(main_raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    for row in main_raw_rows:
        payload = _load_json(str(row["eval_path"]))
        logs_by_method = {
            "ppo": payload.get("marl_episode_logs", []),
            "heuristic": payload.get("heuristic_episode_logs", []),
        }
        for method, episode_logs in logs_by_method.items():
            for episode_log in episode_logs:
                for item in episode_log.get("per_uav_metrics", []):
                    raw_rows.append(
                        {
                            "num_uavs": int(row["num_uavs"]),
                            "method": method,
                            "uav_id": int(item["uav_id"]),
                            "assigned_task_count_total": float(item["assigned_task_count_total"]),
                            "completed_task_count_total": float(item["completed_task_count_total"]),
                            "current_coverage_load": float(item["current_coverage_load"]),
                            "energy_used_j": float(item["energy_used_j"]),
                        }
                    )
    return _aggregate_rows(
        raw_rows,
        group_keys=["num_uavs", "method", "uav_id"],
        metrics=[
            "assigned_task_count_total",
            "completed_task_count_total",
            "current_coverage_load",
            "energy_used_j",
        ],
    )


def _plot_per_uav_diagnostics(diag_rows: list[dict[str, Any]], output_path: Path) -> None:
    plt = _load_matplotlib()
    settings = sorted({int(row["num_uavs"]) for row in diag_rows})
    figure, axes = plt.subplots(len(settings), 4, figsize=(15.2, 4.0 * len(settings)), sharex=False)
    if len(settings) == 1:
        axes = [axes]
    metric_panels = [
        ("assigned_task_count_total", "分配任务数", "数量"),
        ("completed_task_count_total", "完成任务数", "数量"),
        ("current_coverage_load", "覆盖负载", "覆盖用户数"),
        ("energy_used_j", "能耗", "J"),
    ]
    method_colors = {"ppo": "#2E86AB", "heuristic": "#E76F51"}

    for row_axes, num_uavs in zip(axes, settings):
        subset = [row for row in diag_rows if int(row["num_uavs"]) == num_uavs]
        ppo_rows = sorted([row for row in subset if row["method"] == "ppo"], key=lambda item: int(item["uav_id"]))
        heuristic_rows = sorted([row for row in subset if row["method"] == "heuristic"], key=lambda item: int(item["uav_id"]))
        labels = [f"无人机 {int(row['uav_id'])}" for row in ppo_rows]
        positions = list(range(len(labels)))
        width = 0.34
        for axis, (metric, title, ylabel) in zip(row_axes, metric_panels):
            axis.bar(
                [index - width / 2 for index in positions],
                [row[f"{metric}_mean"] for row in ppo_rows],
                width=width,
                yerr=[row[f"{metric}_std"] for row in ppo_rows],
                color=method_colors["ppo"],
                label=method_label("ppo"),
                capsize=4,
            )
            axis.bar(
                [index + width / 2 for index in positions],
                [row[f"{metric}_mean"] for row in heuristic_rows],
                width=width,
                yerr=[row[f"{metric}_std"] for row in heuristic_rows],
                color=method_colors["heuristic"],
                label=method_label("heuristic"),
                capsize=4,
            )
            axis.set_title(f"{num_uavs} 架无人机 - {title}")
            axis.set_ylabel(ylabel)
            axis.set_xticks(positions, labels)
            _style_axis(axis)

    handles, labels = axes[0][0].get_legend_handles_labels() if len(settings) > 1 else axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle("分无人机负载与能耗诊断", y=0.98)
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

def _plot_ppo_vs_heuristic(
    main_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    plt = _load_matplotlib()
    figure, axes = plt.subplots(2, 3, figsize=(13.4, 7.8), sharex=False)

    labels = [f"{int(row['num_uavs'])} 架无人机" for row in main_rows]
    positions = list(range(len(labels)))
    width = 0.34
    metric_panels = [
        ("completion_rate", "任务完成率", "比例"),
        ("average_latency", "平均时延", "时延"),
        ("total_energy", "总能耗", "能耗"),
        ("cache_hit_rate", "缓存命中率", "比例"),
        ("fairness_user_completion", "用户公平性", "Jain 公平指数"),
        ("fairness_uav_load", "无人机负载公平性", "Jain 公平指数"),
    ]

    for axis, (metric, title, ylabel) in zip(axes.flat, metric_panels):
        axis.bar(
            [index - width / 2 for index in positions],
            [row[f"ppo_{metric}_mean"] for row in main_rows],
            width=width,
            yerr=[row[f"ppo_{metric}_std"] for row in main_rows],
            label=method_label("ppo"),
            color="#2E86AB",
            capsize=4,
        )
        axis.bar(
            [index + width / 2 for index in positions],
            [row[f"heuristic_{metric}_mean"] for row in main_rows],
            width=width,
            yerr=[row[f"heuristic_{metric}_std"] for row in main_rows],
            label=method_label("heuristic"),
            color="#E76F51",
            capsize=4,
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, labels)
        if "rate" in metric or "fairness" in metric:
            axis.set_ylim(0.0, 1.05)
        _style_axis(axis)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle("最终配置下的本文方法（PPO）与启发式对比", y=0.98)
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_assignment_comparison(
    main_rows: list[dict[str, Any]],
    assignment_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    plt = _load_matplotlib()
    del main_rows
    figure, axes = plt.subplots(2, 3, figsize=(13.4, 7.8))
    width = 0.35
    nearest_rows = [row for row in assignment_rows if row["assignment_rule"] == "nearest_uav"]
    balanced_rows = [row for row in assignment_rows if row["assignment_rule"] == "least_loaded_uav"]
    labels = [f"{int(row['num_uavs'])} 架无人机" for row in nearest_rows]
    positions = list(range(len(labels)))
    metric_panels = [
        ("completion_rate", "任务完成率", "比例"),
        ("average_latency", "平均时延", "时延"),
        ("total_energy", "总能耗", "能耗"),
        ("cache_hit_rate", "缓存命中率", "比例"),
        ("fairness_user_completion", "用户公平性", "Jain 公平指数"),
        ("fairness_uav_load", "无人机负载公平性", "Jain 公平指数"),
    ]

    for axis, (metric, title, ylabel) in zip(axes.flat, metric_panels):
        axis.bar(
            [index - width / 2 for index in positions],
            [row[f"{metric}_mean"] for row in nearest_rows],
            width=width,
            yerr=[row[f"{metric}_std"] for row in nearest_rows],
            label=assignment_rule_label("nearest_uav"),
            color="#4C78A8",
            capsize=4,
        )
        axis.bar(
            [index + width / 2 for index in positions],
            [row[f"{metric}_mean"] for row in balanced_rows],
            width=width,
            yerr=[row[f"{metric}_std"] for row in balanced_rows],
            label=assignment_rule_label("least_loaded_uav"),
            color="#59A14F",
            capsize=4,
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, labels)
        if "rate" in metric or "fairness" in metric:
            axis.set_ylim(0.0, 1.05)
        _style_axis(axis)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle("敏感场景下的任务分配规则对比", y=0.98)
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_ablation_energy(
    ablation_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    plt = _load_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    labels = [variant_label(str(row["variant"])) for row in ablation_rows]
    positions = list(range(len(labels)))
    colors = ["#2E86AB", "#F39C12", "#C0392B"]

    axes[0].bar(
        positions,
        [row["total_energy_mean"] for row in ablation_rows],
        yerr=[row["total_energy_std"] for row in ablation_rows],
        color=colors[: len(labels)],
        capsize=4,
    )
    axes[0].set_title("能耗")
    axes[0].set_ylabel("总能耗")
    axes[0].set_xticks(positions, labels, rotation=12)
    _style_axis(axes[0])

    axes[1].bar(
        positions,
        [row["delta_total_energy_mean"] for row in ablation_rows],
        yerr=[row["delta_total_energy_std"] for row in ablation_rows],
        color=colors[: len(labels)],
        capsize=4,
    )
    axes[1].axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    axes[1].set_title("相对启发式能耗差")
    axes[1].set_ylabel("能耗差值")
    axes[1].set_xticks(positions, labels, rotation=12)
    _style_axis(axes[1])

    figure.suptitle("2 无人机场景（最近无人机分配）消融实验", y=1.02)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_summary_panel(
    main_rows: list[dict[str, Any]],
    assignment_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    plt = _load_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(11, 8.4))

    labels = [f"{int(row['num_uavs'])} 架无人机" for row in main_rows]
    positions = list(range(len(labels)))
    width = 0.34
    axes[0, 0].bar(
        [index - width / 2 for index in positions],
        [row["ppo_total_energy_mean"] for row in main_rows],
        width=width,
        yerr=[row["ppo_total_energy_std"] for row in main_rows],
        label=method_label("ppo"),
        color="#2E86AB",
        capsize=4,
    )
    axes[0, 0].bar(
        [index + width / 2 for index in positions],
        [row["heuristic_total_energy_mean"] for row in main_rows],
        width=width,
        yerr=[row["heuristic_total_energy_std"] for row in main_rows],
        label=method_label("heuristic"),
        color="#E76F51",
        capsize=4,
    )
    axes[0, 0].set_title("本文方法（PPO）与启发式能耗")
    axes[0, 0].set_ylabel("能耗")
    axes[0, 0].set_xticks(positions, labels)
    _style_axis(axes[0, 0])
    axes[0, 0].legend(frameon=False)

    nearest_rows = [row for row in assignment_rows if row["assignment_rule"] == "nearest_uav"]
    balanced_rows = [row for row in assignment_rows if row["assignment_rule"] == "least_loaded_uav"]
    axes[0, 1].bar(
        [index - width / 2 for index in positions],
        [row["fairness_uav_load_mean"] for row in nearest_rows],
        width=width,
        yerr=[row["fairness_uav_load_std"] for row in nearest_rows],
        label=assignment_rule_label("nearest_uav"),
        color="#4C78A8",
        capsize=4,
    )
    axes[0, 1].bar(
        [index + width / 2 for index in positions],
        [row["fairness_uav_load_mean"] for row in balanced_rows],
        width=width,
        yerr=[row["fairness_uav_load_std"] for row in balanced_rows],
        label=assignment_rule_label("least_loaded_uav"),
        color="#59A14F",
        capsize=4,
    )
    axes[0, 1].set_title("任务分配公平性")
    axes[0, 1].set_ylabel("Jain 公平指数")
    axes[0, 1].set_xticks(positions, labels)
    _style_axis(axes[0, 1])
    axes[0, 1].legend(frameon=False)

    variant_positions = list(range(len(ablation_rows)))
    variant_labels = [variant_label(str(row["variant"])) for row in ablation_rows]
    axes[1, 0].bar(
        variant_positions,
        [row["total_energy_mean"] for row in ablation_rows],
        yerr=[row["total_energy_std"] for row in ablation_rows],
        color=["#2E86AB", "#F39C12", "#C0392B"][: len(ablation_rows)],
        capsize=4,
    )
    axes[1, 0].set_title("消融能耗")
    axes[1, 0].set_ylabel("能耗")
    axes[1, 0].set_xticks(variant_positions, variant_labels, rotation=12)
    _style_axis(axes[1, 0])

    axes[1, 1].bar(
        positions,
        [row["ppo_average_latency_mean"] for row in main_rows],
        yerr=[row["ppo_average_latency_std"] for row in main_rows],
        color="#2E86AB",
        capsize=4,
    )
    axes[1, 1].set_title("本文方法（PPO）时延")
    axes[1, 1].set_ylabel("时延")
    axes[1, 1].set_xticks(positions, labels)
    _style_axis(axes[1, 1])

    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _write_tables(
    *,
    compare_rows: list[dict[str, Any]],
    assignment_rows: list[dict[str, Any]],
    main_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
) -> dict[str, str]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    main_table_md = TABLES_DIR / "table_main_results.md"
    assignment_table_md = TABLES_DIR / "table_assignment_comparison.md"
    ppo_table_md = TABLES_DIR / "table_ppo_vs_heuristic.md"
    ablation_table_md = TABLES_DIR / "table_ablation.md"

    main_table_csv = TABLES_DIR / "table_main_results.csv"
    assignment_table_csv = TABLES_DIR / "table_assignment_comparison.csv"
    ppo_table_csv = TABLES_DIR / "table_ppo_vs_heuristic.csv"
    ablation_table_csv = TABLES_DIR / "table_ablation.csv"

    _write_csv(main_table_csv, compare_rows)
    _write_csv(assignment_table_csv, assignment_rows)
    _write_csv(ppo_table_csv, main_rows)
    _write_csv(ablation_table_csv, ablation_rows)

    main_table_md.write_text(
        _markdown_table(
            compare_rows,
            [
                ("metric", "metric"),
                ("delta mean±std", "delta_mean_std"),
            ],
        ),
        encoding="utf-8",
    )
    assignment_table_md.write_text(
        _markdown_table(
            assignment_rows,
            [
                ("profile", "profile"),
                ("num_uavs", "num_uavs"),
                ("assignment_rule", "assignment_rule"),
                ("latency mean±std", "average_latency_mean_std"),
                ("energy mean±std", "total_energy_mean_std"),
                ("fairness mean±std", "fairness_uav_load_mean_std"),
            ],
        ),
        encoding="utf-8",
    )
    ppo_table_md.write_text(
        _markdown_table(
            main_rows,
            [
                ("num_uavs", "num_uavs"),
                ("assignment_rule", "assignment_rule"),
                ("PPO latency mean±std", "ppo_average_latency_mean_std"),
                ("PPO energy mean±std", "ppo_total_energy_mean_std"),
                ("heuristic energy mean±std", "heuristic_total_energy_mean_std"),
                ("delta energy mean±std", "delta_total_energy_mean_std"),
            ],
        ),
        encoding="utf-8",
    )
    ablation_table_md.write_text(
        _markdown_table(
            ablation_rows,
            [
                ("variant", "variant"),
                ("latency mean±std", "average_latency_mean_std"),
                ("energy mean±std", "total_energy_mean_std"),
                ("delta energy mean±std", "delta_total_energy_mean_std"),
            ],
        ),
        encoding="utf-8",
    )

    return {
        "main_results_markdown": str(main_table_md),
        "main_results_csv": str(main_table_csv),
        "assignment_markdown": str(assignment_table_md),
        "assignment_csv": str(assignment_table_csv),
        "ppo_vs_heuristic_markdown": str(ppo_table_md),
        "ppo_vs_heuristic_csv": str(ppo_table_csv),
        "ablation_markdown": str(ablation_table_md),
        "ablation_csv": str(ablation_table_csv),
    }


def run_final_paper_package(
    *,
    seeds: list[int] | None = None,
    train_episodes: int = 240,
    eval_episodes: int = 32,
    device: str = "auto",
) -> dict[str, Any]:
    seeds = list(seeds or DEFAULT_SEEDS)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    compare_raw, compare_agg = _run_compare_ch4(seeds, episodes=eval_episodes)
    assignment_raw, assignment_agg = _run_assignment_multiseed(seeds, eval_episodes=eval_episodes)
    main_raw, main_agg, main_training_logs = _run_main_multiseed(
        seeds, eval_episodes=eval_episodes, train_episodes=train_episodes, device=device
    )
    ablation_raw, ablation_agg, ablation_training_logs = _run_ablation_multiseed(
        seeds, eval_episodes=eval_episodes, train_episodes=train_episodes, device=device
    )
    per_uav_diag = _collect_per_uav_diagnostics(main_raw)

    training_return_path = FIGURES_DIR / "final_training_return_curve.png"
    training_energy_path = FIGURES_DIR / "final_training_energy_curve.png"
    training_overview_path = FIGURES_DIR / "final_training_behavior_overview.png"
    ppo_vs_heuristic_path = FIGURES_DIR / "final_ppo_vs_heuristic.png"
    assignment_path = FIGURES_DIR / "final_assignment_comparison.png"
    energy_breakdown_path = FIGURES_DIR / "final_energy_breakdown.png"
    compare_delta_path = FIGURES_DIR / "final_compare_ch4_delta.png"
    per_uav_path = FIGURES_DIR / "final_per_uav_diagnostics.png"
    ablation_path = FIGURES_DIR / "final_ablation_energy.png"
    summary_panel_path = FIGURES_DIR / "final_comparison_bars.png"
    training_logs = {
        "main": ablation_training_logs["main"],
        "with_reward_shaping": ablation_training_logs["with_reward_shaping"],
        "no_movement_budget": ablation_training_logs["no_movement_budget"],
    }
    _plot_training_metric_curves(
        training_logs,
        metric="team_return",
        ylabel="团队回报",
        title="本文方法（PPO）训练回报（3 个种子的均值 ± 标准差）",
        output_path=training_return_path,
    )
    _plot_training_metric_curves(
        training_logs,
        metric="total_energy",
        ylabel="总能耗",
        title="本文方法（PPO）训练能耗（3 个种子的均值 ± 标准差）",
        output_path=training_energy_path,
    )
    _plot_training_behavior_overview(training_logs, output_path=training_overview_path)
    _plot_ppo_vs_heuristic(main_agg, ppo_vs_heuristic_path)
    _plot_assignment_comparison(main_agg, assignment_agg, assignment_path)
    _plot_energy_breakdown(main_agg, energy_breakdown_path)
    _plot_compare_ch4_delta(compare_agg, compare_delta_path)
    _plot_per_uav_diagnostics(per_uav_diag, per_uav_path)
    _plot_ablation_energy(ablation_agg, ablation_path)
    _plot_summary_panel(main_agg, assignment_agg, ablation_agg, summary_panel_path)

    tables = _write_tables(
        compare_rows=compare_agg,
        assignment_rows=assignment_agg,
        main_rows=main_agg,
        ablation_rows=ablation_agg,
    )

    package_manifest = {
        "final_main_config": {**FINAL_MAIN_CONFIG, "train_episodes": int(train_episodes)},
        "seeds": seeds,
        "train_episodes": int(train_episodes),
        "eval_episodes": eval_episodes,
        "device_request": device,
        "one_click_commands": [
            "python -m venv .venv",
            ".\\.venv\\Scripts\\python.exe -m pip install -r 第四章/requirements.txt",
            ".\\.venv\\Scripts\\python.exe 第三章/run_experiment.py --episodes 1 --compare-ch4 --seed 42",
            f".\\.venv\\Scripts\\python.exe 第四章/run_finalize_paper.py --seeds 42 52 62 --eval-episodes {int(eval_episodes)} --train-episodes {int(train_episodes)} --device auto",
        ],
        "result_directories": {
            "stage5": str(RESULTS_DIR / "paper_stage5"),
            "stage6": str(FINAL_DIR),
            "root_results": str(RESULTS_DIR),
        },
        "tables": tables,
        "figures": {
            "training_return_curve": str(training_return_path),
            "training_energy_curve": str(training_energy_path),
            "training_behavior_overview": str(training_overview_path),
            "ppo_vs_heuristic": str(ppo_vs_heuristic_path),
            "assignment_comparison": str(assignment_path),
            "energy_breakdown": str(energy_breakdown_path),
            "compare_ch4_delta": str(compare_delta_path),
            "per_uav_diagnostics": str(per_uav_path),
            "ablation_energy": str(ablation_path),
            "summary_panel": str(summary_panel_path),
        },
    }

    write_json(FINAL_DIR / "compare_ch4_multiseed_raw.json", {"rows": compare_raw})
    write_json(FINAL_DIR / "compare_ch4_multiseed_summary.json", {"rows": compare_agg})
    write_json(FINAL_DIR / "assignment_multiseed_raw.json", {"rows": assignment_raw})
    write_json(FINAL_DIR / "assignment_multiseed_summary.json", {"rows": assignment_agg})
    write_json(FINAL_DIR / "ppo_vs_heuristic_multiseed_raw.json", {"rows": main_raw})
    write_json(FINAL_DIR / "ppo_vs_heuristic_multiseed_summary.json", {"rows": main_agg})
    write_json(FINAL_DIR / "ablation_multiseed_raw.json", {"rows": ablation_raw})
    write_json(FINAL_DIR / "ablation_multiseed_summary.json", {"rows": ablation_agg})
    write_json(FINAL_DIR / "reproducibility_package.json", package_manifest)

    return {
        "final_main_config": {**FINAL_MAIN_CONFIG, "train_episodes": int(train_episodes)},
        "seeds": seeds,
        "train_episodes": int(train_episodes),
        "device_request": device,
        "compare_ch4_summary_path": str(FINAL_DIR / "compare_ch4_multiseed_summary.json"),
        "assignment_summary_path": str(FINAL_DIR / "assignment_multiseed_summary.json"),
        "ppo_vs_heuristic_summary_path": str(FINAL_DIR / "ppo_vs_heuristic_multiseed_summary.json"),
        "ablation_summary_path": str(FINAL_DIR / "ablation_multiseed_summary.json"),
        "tables": tables,
        "figures": {
            "training_return_curve": str(training_return_path),
            "training_energy_curve": str(training_energy_path),
            "training_behavior_overview": str(training_overview_path),
            "ppo_vs_heuristic": str(ppo_vs_heuristic_path),
            "assignment_comparison": str(assignment_path),
            "energy_breakdown": str(energy_breakdown_path),
            "compare_ch4_delta": str(compare_delta_path),
            "per_uav_diagnostics": str(per_uav_path),
            "ablation_energy": str(ablation_path),
            "summary_panel": str(summary_panel_path),
        },
        "package_manifest_path": str(FINAL_DIR / "reproducibility_package.json"),
    }
