"""第四章 stage-5 论文实验模块。

该模块负责组织论文阶段的调参、敏感性实验、与第三章对比和图表生成流程，
用于产出论文撰写所需的实验矩阵、汇总表和可视化结果。
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path
from typing import Any

from common.uav_mec.logging_utils import write_json

from ..experiments import run_sensitive_experiment
from .config import build_marl_config
from .eval import run_marl_evaluation
from .train import run_marl_training


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
PAPER_DIR = RESULTS_DIR / "paper_stage5"
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
CHAPTER3_DIR = WORKSPACE_ROOT / "第三章"

MAIN_NUM_UAVS = 2
MAIN_ASSIGNMENT_RULE = "nearest_uav"
DEFAULT_TUNING_SEEDS = [42, 52, 62]
TUNING_EVAL_OFFSET = 100
FINAL_MAIN_CANDIDATE_NAME = "freeze_noshaping_240"

TUNING_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "base_e12",
        "description": "baseline PPO, short training budget",
        "overrides": {
            "train_episodes": 12,
            "actor_lr": 3.0e-4,
            "critic_lr": 1.0e-3,
            "ppo_clip_eps": 0.20,
            "entropy_coef": 0.010,
            "value_loss_coef": 0.50,
            "reward_energy_weight": 1.20,
            "reward_action_magnitude_weight": 0.15,
        },
    },
    {
        "name": "energy_e30",
        "description": "lower exploration and stronger energy regularization",
        "overrides": {
            "train_episodes": 30,
            "actor_lr": 2.5e-4,
            "critic_lr": 8.0e-4,
            "ppo_clip_eps": 0.18,
            "entropy_coef": 0.008,
            "value_loss_coef": 0.60,
            "reward_energy_weight": 1.50,
            "reward_action_magnitude_weight": 0.20,
        },
    },
    {
        "name": "energy_e60",
        "description": "same energy-focused setting with a longer budget",
        "overrides": {
            "train_episodes": 60,
            "actor_lr": 2.5e-4,
            "critic_lr": 8.0e-4,
            "ppo_clip_eps": 0.18,
            "entropy_coef": 0.008,
            "value_loss_coef": 0.60,
            "reward_energy_weight": 1.50,
            "reward_action_magnitude_weight": 0.20,
        },
    },
    {
        "name": "stable_e30",
        "description": "tighter clip with lower entropy and stronger value fit",
        "overrides": {
            "train_episodes": 30,
            "actor_lr": 2.0e-4,
            "critic_lr": 6.0e-4,
            "ppo_clip_eps": 0.15,
            "entropy_coef": 0.005,
            "value_loss_coef": 0.70,
            "reward_energy_weight": 1.70,
            "reward_action_magnitude_weight": 0.22,
        },
    },
    {
        "name": "cons_lowlr",
        "description": "higher training budget with steadier conservative PPO updates",
        "overrides": {
            "train_episodes": 120,
            "actor_lr": 1.0e-4,
            "critic_lr": 6.0e-4,
            "ppo_clip_eps": 0.12,
            "entropy_coef": 0.0008,
            "value_loss_coef": 0.75,
            "reward_energy_weight": 4.50,
            "reward_action_magnitude_weight": 0.65,
            "action_std_init": 0.10,
            "action_std_min": 0.015,
            "action_std_decay": 0.983,
        },
    },
    {
        "name": "freeze_energy2_240",
        "description": "ultra-low exploration with stronger energy and action regularization",
        "overrides": {
            "train_episodes": 240,
            "actor_lr": 8.0e-5,
            "critic_lr": 5.0e-4,
            "ppo_clip_eps": 0.10,
            "entropy_coef": 0.0003,
            "value_loss_coef": 0.82,
            "reward_energy_weight": 2.00,
            "reward_action_magnitude_weight": 1.00,
            "action_std_init": 0.04,
            "action_std_min": 0.005,
            "action_std_decay": 0.984,
        },
    },
    {
        "name": "freeze_noshaping_240",
        "description": "ultra-low exploration with movement budget and no explicit energy/action shaping",
        "overrides": {
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
        },
    },
]


def get_tuning_candidate(name: str) -> dict[str, Any]:
    for candidate in TUNING_CANDIDATES:
        if candidate["name"] == name:
            return candidate
    raise KeyError(f"Unknown tuning candidate: {name}")


def get_candidate_overrides(name: str) -> dict[str, Any]:
    return dict(get_tuning_candidate(name)["overrides"])


def _normalize_seeds(seeds: list[int] | None) -> list[int]:
    resolved = list(seeds or DEFAULT_TUNING_SEEDS)
    unique: list[int] = []
    for seed in resolved:
        normalized = int(seed)
        if normalized not in unique:
            unique.append(normalized)
    if not unique:
        raise ValueError("At least one tuning seed is required.")
    return unique


def _load_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            "matplotlib is required for stage-5 paper plots. Install dependencies from 第四章/requirements.txt first."
        ) from exc
    return plt


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _metric_delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    if left.get(key) is None or right.get(key) is None:
        return None
    return float(left[key]) - float(right[key])


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


def _format_mean_std(mean: float | None, std: float | None, digits: int = 4) -> str:
    if mean is None or std is None:
        return "null"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def _metric_sort_value(value: float | None, *, prefer_high: bool) -> tuple[int, float]:
    if value is None:
        return (1, 0.0)
    return (0, -float(value) if prefer_high else float(value))


def _select_best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _value(row: dict[str, Any], *, mean_key: str, raw_key: str) -> float | None:
        if mean_key in row:
            return row.get(mean_key)
        return row.get(raw_key)

    return min(
        rows,
        key=lambda row: (
            _metric_sort_value(_value(row, mean_key="completion_rate_mean", raw_key="completion_rate"), prefer_high=True),
            _metric_sort_value(_value(row, mean_key="total_energy_mean", raw_key="total_energy"), prefer_high=False),
            _metric_sort_value(_value(row, mean_key="average_latency_mean", raw_key="average_latency"), prefer_high=False),
            _metric_sort_value(row.get("completion_rate_std", 0.0), prefer_high=False),
        ),
    )


def _build_eval_overrides(train_payload: dict[str, Any], *, output_tag: str, device: str) -> dict[str, Any]:
    config = train_payload["config"]
    return {
        "output_tag": output_tag,
        "use_movement_budget": bool(config.get("use_movement_budget", True)),
        "device": device,
    }


def _run_train_eval(
    *,
    seed: int,
    eval_seed: int,
    eval_episodes: int,
    num_uavs: int,
    assignment_rule: str,
    output_tag: str,
    overrides: dict[str, Any],
    device: str = "auto",
) -> dict[str, Any]:
    merged = dict(overrides)
    train_episodes = int(merged.pop("train_episodes"))
    merged["output_tag"] = output_tag
    merged["device"] = device
    train_payload = run_marl_training(
        seed=seed,
        train_episodes=train_episodes,
        num_uavs=num_uavs,
        assignment_rule=assignment_rule,
        overrides=merged,
    )
    eval_payload = run_marl_evaluation(
        seed=eval_seed,
        eval_episodes=eval_episodes,
        num_uavs=num_uavs,
        assignment_rule=assignment_rule,
        model_path=train_payload["checkpoint_path"],
        overrides=_build_eval_overrides(train_payload, output_tag=output_tag, device=device),
    )
    return {
        "train": train_payload,
        "eval": eval_payload,
    }


def _summarize_eval_result(
    *,
    label: str,
    train_eval: dict[str, Any],
) -> dict[str, Any]:
    config = train_eval["train"]["config"]
    marl_metrics = train_eval["eval"]["marl_metrics"]
    heuristic_metrics = train_eval["eval"]["heuristic_metrics"]
    return {
        "label": label,
        "output_tag": config["output_tag"],
        "num_uavs": config["num_uavs"],
        "assignment_rule": config["assignment_rule"],
        "train_episodes": config["train_episodes"],
        "actor_lr": config["actor_lr"],
        "critic_lr": config["critic_lr"],
        "clip_ratio": config["ppo_clip_eps"],
        "entropy_coef": config["entropy_coef"],
        "value_coef": config["value_loss_coef"],
        "reward_energy_weight": config["reward_energy_weight"],
        "reward_action_magnitude_weight": config["reward_action_magnitude_weight"],
        "use_movement_budget": config["use_movement_budget"],
        "completion_rate": _float_or_none(marl_metrics["completion_rate"]),
        "average_latency": _float_or_none(marl_metrics["average_latency"]),
        "total_energy": _float_or_none(marl_metrics["total_energy"]),
        "cache_hit_rate": _float_or_none(marl_metrics["cache_hit_rate"]),
        "fairness_uav_load": _float_or_none(marl_metrics["fairness_uav_load"]),
        "heuristic_completion_rate": _float_or_none(heuristic_metrics["completion_rate"]),
        "heuristic_average_latency": _float_or_none(heuristic_metrics["average_latency"]),
        "heuristic_total_energy": _float_or_none(heuristic_metrics["total_energy"]),
        "heuristic_cache_hit_rate": _float_or_none(heuristic_metrics["cache_hit_rate"]),
        "heuristic_fairness_uav_load": _float_or_none(heuristic_metrics["fairness_uav_load"]),
        "delta_completion_rate": _metric_delta(marl_metrics, heuristic_metrics, "completion_rate"),
        "delta_average_latency": _metric_delta(marl_metrics, heuristic_metrics, "average_latency"),
        "delta_total_energy": _metric_delta(marl_metrics, heuristic_metrics, "total_energy"),
        "checkpoint_path": train_eval["train"]["checkpoint_path"],
        "train_log_path": train_eval["train"]["train_log_path"],
        "eval_path": train_eval["eval"]["eval_path"],
    }


def _append_metric_summary(entry: dict[str, Any], group: list[dict[str, Any]], field: str) -> None:
    values = [float(item[field]) for item in group if item.get(field) is not None]
    if not values:
        entry[f"{field}_mean"] = None
        entry[f"{field}_std"] = None
        entry[f"{field}_mean_std"] = "null"
        return
    mean, std = _metric_stats(values)
    entry[f"{field}_mean"] = mean
    entry[f"{field}_std"] = std
    entry[f"{field}_mean_std"] = _format_mean_std(mean, std)


def _aggregate_tuning_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        grouped.setdefault(str(row["label"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    metric_fields = [
        "completion_rate",
        "average_latency",
        "total_energy",
        "cache_hit_rate",
        "fairness_uav_load",
        "heuristic_completion_rate",
        "heuristic_average_latency",
        "heuristic_total_energy",
        "heuristic_cache_hit_rate",
        "heuristic_fairness_uav_load",
        "delta_completion_rate",
        "delta_average_latency",
        "delta_total_energy",
    ]
    for label, group in grouped.items():
        template = group[0]
        entry = {
            "label": label,
            "description": template["description"],
            "num_uavs": template["num_uavs"],
            "assignment_rule": template["assignment_rule"],
            "train_episodes": template["train_episodes"],
            "actor_lr": template["actor_lr"],
            "critic_lr": template["critic_lr"],
            "clip_ratio": template["clip_ratio"],
            "entropy_coef": template["entropy_coef"],
            "value_coef": template["value_coef"],
            "reward_energy_weight": template["reward_energy_weight"],
            "reward_action_magnitude_weight": template["reward_action_magnitude_weight"],
            "use_movement_budget": template["use_movement_budget"],
            "num_seeds": len(group),
            "tuning_seeds": ",".join(str(int(item["tuning_seed"])) for item in group),
        }
        for field in metric_fields:
            _append_metric_summary(entry, group, field)
        summary_rows.append(entry)
    summary_rows.sort(key=lambda item: str(item["label"]))
    return summary_rows


def _aggregate_rows(
    raw_rows: list[dict[str, Any]],
    *,
    group_keys: list[str],
    metric_fields: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in raw_rows:
        key = tuple(row[group_key] for group_key in group_keys)
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, group in grouped.items():
        template = group[0]
        entry = {group_key: value for group_key, value in zip(group_keys, key)}
        for field, value in template.items():
            if field in entry or field in metric_fields:
                continue
            entry[field] = value
        entry["num_seeds"] = len(group)
        for field in metric_fields:
            _append_metric_summary(entry, group, field)
        summary_rows.append(entry)
    summary_rows.sort(key=lambda item: tuple(str(item[group_key]) for group_key in group_keys))
    return summary_rows


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


def _run_tuning(
    *,
    tuning_seeds: list[int],
    eval_episodes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tuning_raw_rows: list[dict[str, Any]] = []
    for candidate in TUNING_CANDIDATES:
        for tuning_seed in tuning_seeds:
            eval_seed = tuning_seed + TUNING_EVAL_OFFSET
            output_tag = f"tune_{candidate['name']}_s{tuning_seed}"
            train_eval = _run_train_eval(
                seed=tuning_seed,
                eval_seed=eval_seed,
                eval_episodes=eval_episodes,
                num_uavs=MAIN_NUM_UAVS,
                assignment_rule=MAIN_ASSIGNMENT_RULE,
                output_tag=output_tag,
                overrides=candidate["overrides"],
            )
            row = _summarize_eval_result(label=candidate["name"], train_eval=train_eval)
            row["description"] = candidate["description"]
            row["tuning_seed"] = tuning_seed
            row["eval_seed"] = eval_seed
            tuning_raw_rows.append(row)
    tuning_summary_rows = _aggregate_tuning_rows(tuning_raw_rows)
    selected_row = _select_best_candidate(tuning_summary_rows)
    return tuning_raw_rows, tuning_summary_rows, selected_row


def _select_named_candidate(rows: list[dict[str, Any]], *, name: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("label")) == name:
            return row
    raise KeyError(f"Named tuning candidate not found in summary rows: {name}")


def _plot_training_curves(
    *,
    training_logs: dict[str, list[list[dict[str, Any]]]],
    output_path: Path,
) -> None:
    plt = _load_matplotlib()
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = {
        "main": "#2E86AB",
        "with_reward_shaping": "#E67E22",
        "no_movement_budget": "#C0392B",
    }
    label_map = {
        "main": "main",
        "with_reward_shaping": "with_reward_shaping",
        "no_movement_budget": "no_movement_budget",
    }

    for variant, logs in training_logs.items():
        episodes, return_mean, return_std = _aggregate_training_curve(logs, "team_return")
        _, energy_mean, energy_std = _aggregate_training_curve(logs, "total_energy")
        color = colors.get(variant, "#555555")
        label = label_map.get(variant, variant)
        axes[0].plot(episodes, return_mean, label=label, color=color, linewidth=2.0)
        axes[0].fill_between(
            episodes,
            [value - delta for value, delta in zip(return_mean, return_std)],
            [value + delta for value, delta in zip(return_mean, return_std)],
            color=color,
            alpha=0.14,
        )
        axes[1].plot(episodes, energy_mean, label=label, color=color, linewidth=2.0)
        axes[1].fill_between(
            episodes,
            [value - delta for value, delta in zip(energy_mean, energy_std)],
            [value + delta for value, delta in zip(energy_mean, energy_std)],
            color=color,
            alpha=0.14,
        )

    axes[0].set_ylabel("Team Return")
    axes[0].set_title("PPO Training Curves (mean +/- std)")
    _style_axis(axes[0])
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Total Energy")
    _style_axis(axes[1])
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_assignment_rules(rows: list[dict[str, Any]], output_path: Path) -> None:
    plt = _load_matplotlib()
    labels = [f"u{row['num_uavs']}-{row['assignment_rule']}" for row in rows]
    energy = [row["total_energy_mean"] for row in rows]
    energy_std = [row["total_energy_std"] for row in rows]
    fairness = [row["fairness_uav_load_mean"] for row in rows]
    fairness_std = [row["fairness_uav_load_std"] for row in rows]
    positions = list(range(len(labels)))

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(positions, energy, yerr=energy_std, color="#4C78A8", capsize=4)
    axes[0].set_ylabel("Total Energy")
    axes[0].set_title("Assignment Rule Comparison (Sensitive Profile)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(positions, fairness, yerr=fairness_std, color="#F58518", capsize=4)
    axes[1].set_ylabel("Fairness")
    axes[1].set_xticks(positions, labels, rotation=20)
    axes[1].grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_main_comparison(rows: list[dict[str, Any]], output_path: Path) -> None:
    plt = _load_matplotlib()
    labels = [f"u{row['num_uavs']}-ppo" for row in rows] + [f"u{row['num_uavs']}-heuristic" for row in rows]
    energy = [row["total_energy_mean"] for row in rows] + [row["heuristic_total_energy_mean"] for row in rows]
    energy_std = [row["total_energy_std"] for row in rows] + [row["heuristic_total_energy_std"] for row in rows]
    latency = [row["average_latency_mean"] for row in rows] + [row["heuristic_average_latency_mean"] for row in rows]
    latency_std = [row["average_latency_std"] for row in rows] + [row["heuristic_average_latency_std"] for row in rows]
    positions = list(range(len(labels)))

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(positions, energy, yerr=energy_std, color=["#54A24B"] * len(rows) + ["#E45756"] * len(rows), capsize=4)
    axes[0].set_ylabel("Total Energy")
    axes[0].set_title("PPO vs Heuristic")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(positions, latency, yerr=latency_std, color=["#54A24B"] * len(rows) + ["#E45756"] * len(rows), capsize=4)
    axes[1].set_ylabel("Average Latency")
    axes[1].set_xticks(positions, labels, rotation=20)
    axes[1].grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _make_markdown_summary(
    *,
    final_config: dict[str, Any],
    selected_tuning_row: dict[str, Any],
    tuning_seeds: list[int],
    chapter_compare_rows: list[dict[str, Any]],
    assignment_rows: list[dict[str, Any]],
    main_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# 第五阶段实验汇总",
        "",
        "## 最终 PPO 主配置",
        "",
        f"- selected_candidate: `{selected_tuning_row['label']}`",
        f"- tuning_seeds: `{', '.join(str(seed) for seed in tuning_seeds)}`",
        f"- train_episodes: `{final_config['train_episodes']}`",
        f"- actor_lr / critic_lr: `{final_config['actor_lr']}` / `{final_config['critic_lr']}`",
        f"- clip_ratio: `{final_config['ppo_clip_eps']}`",
        f"- entropy_coef: `{final_config['entropy_coef']}`",
        f"- value_coef: `{final_config['value_loss_coef']}`",
        f"- reward_energy_weight: `{final_config['reward_energy_weight']}`",
        f"- reward_action_magnitude_weight: `{final_config['reward_action_magnitude_weight']}`",
        f"- use_movement_budget: `{final_config['use_movement_budget']}`",
        "",
        "## A. Chapter3 vs Chapter4(NUM_UAVS=1)",
        "",
        "| metric | delta mean±std |",
        "| --- | ---: |",
    ]
    for row in chapter_compare_rows:
        lines.append(f"| {row['metric']} | {row['delta_mean_std']} |")

    lines.extend(
        [
            "",
            "## B. nearest_uav vs least_loaded_uav",
            "",
            "| setting | completion_rate mean±std | average_latency mean±std | total_energy mean±std | fairness_uav_load mean±std |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in assignment_rows:
        lines.append(
            f"| u{row['num_uavs']} {row['assignment_rule']} | {row['completion_rate_mean_std']} | {row['average_latency_mean_std']} | {row['total_energy_mean_std']} | {row['fairness_uav_load_mean_std']} |"
        )

    lines.extend(
        [
            "",
            "## C. PPO vs heuristic",
            "",
            "| setting | PPO completion mean±std | PPO latency mean±std | PPO energy mean±std | heuristic energy mean±std | delta energy mean±std |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in main_rows:
        lines.append(
            f"| u{row['num_uavs']} {row['assignment_rule']} | {row['completion_rate_mean_std']} | {row['average_latency_mean_std']} | {row['total_energy_mean_std']} | {row['heuristic_total_energy_mean_std']} | {row['delta_total_energy_mean_std']} |"
        )

    lines.extend(
        [
            "",
            "## D. 最小消融",
            "",
            "| variant | completion_rate mean±std | average_latency mean±std | total_energy mean±std | delta energy vs heuristic mean±std |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in ablation_rows:
        lines.append(
            f"| {row['label']} | {row['completion_rate_mean_std']} | {row['average_latency_mean_std']} | {row['total_energy_mean_std']} | {row['delta_total_energy_mean_std']} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_paper_experiments(*, seed: int = 42, eval_seed: int = 142, eval_episodes: int = 32, device: str = "auto") -> dict[str, Any]:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    tuning_raw_rows: list[dict[str, Any]] = []
    tuning_runs: dict[str, dict[str, Any]] = {}
    for candidate in TUNING_CANDIDATES:
        output_tag = f"tune_{candidate['name']}_s{seed}"
        train_eval = _run_train_eval(
            seed=seed,
            eval_seed=eval_seed,
            eval_episodes=eval_episodes,
            num_uavs=MAIN_NUM_UAVS,
            assignment_rule=MAIN_ASSIGNMENT_RULE,
            output_tag=output_tag,
            overrides=candidate["overrides"],
            device=device,
        )
        row = _summarize_eval_result(label=candidate["name"], train_eval=train_eval)
        row["description"] = candidate["description"]
        row["tuning_seed"] = seed
        row["eval_seed"] = eval_seed
        tuning_raw_rows.append(row)
        tuning_runs[candidate["name"]] = train_eval
    tuning_summary_rows = _aggregate_tuning_rows(tuning_raw_rows)
    best_tuning_row = _select_best_candidate(tuning_summary_rows)
    selected_tuning_row = _select_named_candidate(tuning_summary_rows, name=FINAL_MAIN_CANDIDATE_NAME)

    final_overrides = get_candidate_overrides(str(selected_tuning_row["label"]))
    final_config = build_marl_config(
        {
            **final_overrides,
            "seed": seed,
            "num_uavs": MAIN_NUM_UAVS,
            "assignment_rule": MAIN_ASSIGNMENT_RULE,
            "output_tag": f"paper_{selected_tuning_row['label']}_u{MAIN_NUM_UAVS}",
            "device": device,
        }
    ).to_dict()

    main_raw_rows: list[dict[str, Any]] = []
    main_training_logs: dict[str, list[list[dict[str, Any]]]] = {"main": []}
    for num_uavs in (2, 3):
        output_tag = f"paper_main_s{seed}_u{num_uavs}"
        train_eval = _run_train_eval(
            seed=seed,
            eval_seed=eval_seed,
            eval_episodes=eval_episodes,
            num_uavs=num_uavs,
            assignment_rule=MAIN_ASSIGNMENT_RULE,
            output_tag=output_tag,
            overrides=final_overrides,
            device=device,
        )
        row = _summarize_eval_result(label=f"paper_main_u{num_uavs}", train_eval=train_eval)
        row["description"] = "stage5 main experiment"
        row["seed"] = seed
        row["eval_seed"] = eval_seed
        main_raw_rows.append(row)
        if num_uavs == MAIN_NUM_UAVS:
            main_training_logs["main"].append(train_eval["train"]["training_log"])
    main_summary_rows = _aggregate_rows(
        main_raw_rows,
        group_keys=["num_uavs", "assignment_rule", "label"],
        metric_fields=[
            "completion_rate",
            "average_latency",
            "total_energy",
            "cache_hit_rate",
            "fairness_uav_load",
            "heuristic_completion_rate",
            "heuristic_average_latency",
            "heuristic_total_energy",
            "heuristic_cache_hit_rate",
            "heuristic_fairness_uav_load",
            "delta_completion_rate",
            "delta_average_latency",
            "delta_total_energy",
        ],
    )

    ablation_raw_rows: list[dict[str, Any]] = []
    ablation_training_logs: dict[str, list[list[dict[str, Any]]]] = {
        "with_reward_shaping": [],
        "no_movement_budget": [],
    }
    ablation_settings = [
        {
            "label": "with_reward_shaping",
            "description": "reward_energy_weight=2.0 and reward_action_magnitude_weight=1.0",
            "overrides": {
                **final_overrides,
                "reward_energy_weight": 2.0,
                "reward_action_magnitude_weight": 1.0,
            },
        },
        {
            "label": "no_movement_budget",
            "description": "use_movement_budget=False with all other settings fixed",
            "overrides": {
                **final_overrides,
                "use_movement_budget": False,
            },
        },
    ]
    for ablation in ablation_settings:
        output_tag = f"paper_{ablation['label']}_s{seed}"
        train_eval = _run_train_eval(
            seed=seed,
            eval_seed=eval_seed,
            eval_episodes=eval_episodes,
            num_uavs=MAIN_NUM_UAVS,
            assignment_rule=MAIN_ASSIGNMENT_RULE,
            output_tag=output_tag,
            overrides=ablation["overrides"],
            device=device,
        )
        row = _summarize_eval_result(label=ablation["label"], train_eval=train_eval)
        row["description"] = ablation["description"]
        row["seed"] = seed
        row["eval_seed"] = eval_seed
        ablation_raw_rows.append(row)
        ablation_training_logs[str(ablation["label"])].append(train_eval["train"]["training_log"])
    ablation_summary_rows = _aggregate_rows(
        ablation_raw_rows,
        group_keys=["label", "description"],
        metric_fields=[
            "completion_rate",
            "average_latency",
            "total_energy",
            "cache_hit_rate",
            "fairness_uav_load",
            "heuristic_completion_rate",
            "heuristic_average_latency",
            "heuristic_total_energy",
            "heuristic_cache_hit_rate",
            "heuristic_fairness_uav_load",
            "delta_completion_rate",
            "delta_average_latency",
            "delta_total_energy",
        ],
    )

    assignment_raw_rows: list[dict[str, Any]] = []
    for num_uavs in (2, 3):
        for assignment_rule in ("nearest_uav", "least_loaded_uav"):
            result = run_sensitive_experiment(
                seed=seed,
                episodes=eval_episodes,
                num_uavs=num_uavs,
                assignment_rule=assignment_rule,
            )
            metrics = result["averaged_metrics"]
            assignment_raw_rows.append(
                {
                    "profile": "sensitive",
                    "num_uavs": num_uavs,
                    "assignment_rule": assignment_rule,
                    "seed": seed,
                    "completion_rate": _float_or_none(metrics["completion_rate"]),
                    "average_latency": _float_or_none(metrics["average_latency"]),
                    "total_energy": _float_or_none(metrics["total_energy"]),
                    "cache_hit_rate": _float_or_none(metrics["cache_hit_rate"]),
                    "fairness_uav_load": _float_or_none(metrics["fairness_uav_load"]),
                    "result_path": str(RESULTS_DIR / f"experiment_sensitive_u{num_uavs}_{assignment_rule}.json"),
                }
            )
    assignment_summary_rows = _aggregate_rows(
        assignment_raw_rows,
        group_keys=["profile", "num_uavs", "assignment_rule"],
        metric_fields=[
            "completion_rate",
            "average_latency",
            "total_energy",
            "cache_hit_rate",
            "fairness_uav_load",
        ],
    )

    if str(CHAPTER3_DIR) not in sys.path:
        sys.path.insert(0, str(CHAPTER3_DIR))
    from chapter3.experiments import compare_with_chapter4

    chapter_compare = compare_with_chapter4(seed=seed, episodes=1)
    chapter_compare_raw_rows = [
        {
            "seed": seed,
            "metric": metric,
            "delta": payload["delta"],
        }
        for metric, payload in chapter_compare["comparison"].items()
    ]
    chapter_compare_summary_rows = _aggregate_rows(
        chapter_compare_raw_rows,
        group_keys=["metric"],
        metric_fields=["delta"],
    )

    tuning_json = PAPER_DIR / "tuning_summary.json"
    tuning_csv = PAPER_DIR / "tuning_summary.csv"
    main_json = PAPER_DIR / "main_experiment_matrix.json"
    main_csv = PAPER_DIR / "main_experiment_matrix.csv"
    ablation_json = PAPER_DIR / "ablation_summary.json"
    ablation_csv = PAPER_DIR / "ablation_summary.csv"
    assignment_json = PAPER_DIR / "assignment_rule_matrix.json"
    assignment_csv = PAPER_DIR / "assignment_rule_matrix.csv"
    compare_json = PAPER_DIR / "chapter3_vs_chapter4_num_uavs1.json"
    compare_csv = PAPER_DIR / "chapter3_vs_chapter4_num_uavs1.csv"
    config_notes_path = PAPER_DIR / "config_notes.json"
    summary_path = PAPER_DIR / "paper_summary.md"
    training_curve_path = PAPER_DIR / "ppo_training_curves.png"
    assignment_plot_path = PAPER_DIR / "assignment_rule_comparison.png"
    main_plot_path = PAPER_DIR / "ppo_vs_heuristic.png"

    write_json(
        tuning_json,
        {
            "selected_candidate": selected_tuning_row["label"],
            "best_observed_candidate": best_tuning_row["label"],
            "tuning_seed": seed,
            "eval_seed": eval_seed,
            "rows": tuning_summary_rows,
            "raw_rows": tuning_raw_rows,
        },
    )
    write_json(main_json, {"rows": main_summary_rows, "raw_rows": main_raw_rows})
    write_json(ablation_json, {"rows": ablation_summary_rows, "raw_rows": ablation_raw_rows})
    write_json(assignment_json, {"rows": assignment_summary_rows, "raw_rows": assignment_raw_rows})
    write_json(compare_json, {"comparison": chapter_compare["comparison"], "rows": chapter_compare_summary_rows, "raw_rows": chapter_compare_raw_rows})
    _write_csv(tuning_csv, tuning_summary_rows)
    _write_csv(main_csv, main_summary_rows)
    _write_csv(ablation_csv, ablation_summary_rows)
    _write_csv(assignment_csv, assignment_summary_rows)
    _write_csv(compare_csv, chapter_compare_summary_rows)

    config_notes = {
        "tuning_protocol": {
            "tuning_seed": seed,
            "eval_seed": eval_seed,
            "eval_episodes": eval_episodes,
            "device_request": device,
            "selection_rule": "highest completion_rate, then lowest total_energy, then lowest average_latency",
            "main_setting": {
                "num_uavs": MAIN_NUM_UAVS,
                "assignment_rule": MAIN_ASSIGNMENT_RULE,
            },
            "candidates": TUNING_CANDIDATES,
        },
        "selected_final_config": final_config,
        "main_matrix_notes": {
            "chapter3_vs_chapter4": "NUM_UAVS=1 compare-ch4 check",
            "assignment_rule_profile": "sensitive",
            "ppo_vs_heuristic_settings": ["u2 nearest_uav", "u3 nearest_uav"],
        },
        "ablations": {
            "with_reward_shaping": "reward_energy_weight=2.0 and reward_action_magnitude_weight=1.0",
            "no_movement_budget": "use_movement_budget=False with all other main PPO settings fixed",
            "no_centralized_critic": "not included in this stage to keep a single algorithm implementation",
        },
    }
    write_json(config_notes_path, config_notes)

    summary_text = _make_markdown_summary(
        final_config=final_config,
        selected_tuning_row=selected_tuning_row,
        tuning_seeds=[seed],
        chapter_compare_rows=chapter_compare_summary_rows,
        assignment_rows=assignment_summary_rows,
        main_rows=main_summary_rows,
        ablation_rows=ablation_summary_rows,
    )
    summary_path.write_text(summary_text, encoding="utf-8")

    _plot_training_curves(
        training_logs={
            "main": main_training_logs["main"],
            "with_reward_shaping": ablation_training_logs["with_reward_shaping"],
            "no_movement_budget": ablation_training_logs["no_movement_budget"],
        },
        output_path=training_curve_path,
    )
    _plot_assignment_rules(assignment_summary_rows, assignment_plot_path)
    _plot_main_comparison(main_summary_rows, main_plot_path)

    payload = {
        "selected_final_config": final_config,
        "selected_tuning_candidate": selected_tuning_row["label"],
        "best_observed_tuning_candidate": best_tuning_row["label"],
        "tuning_seed": seed,
        "eval_seed": eval_seed,
        "device_request": device,
        "compare_ch4_summary_path": str(compare_json),
        "tuning_summary_path": str(tuning_json),
        "main_matrix_path": str(main_json),
        "ablation_summary_path": str(ablation_json),
        "assignment_matrix_path": str(assignment_json),
        "config_notes_path": str(config_notes_path),
        "paper_summary_path": str(summary_path),
        "plot_paths": {
            "training_curves": str(training_curve_path),
            "assignment_rule_comparison": str(assignment_plot_path),
            "ppo_vs_heuristic": str(main_plot_path),
        },
        "main_rows": main_summary_rows,
        "ablation_rows": ablation_summary_rows,
    }
    write_json(PAPER_DIR / "paper_experiments_summary.json", payload)
    return payload
