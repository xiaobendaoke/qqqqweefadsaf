from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path
from typing import Any

from common.uav_mec.logging_utils import write_json

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
    "train_episodes": 30,
    "actor_lr": 2.5e-4,
    "critic_lr": 8.0e-4,
    "ppo_clip_eps": 0.18,
    "entropy_coef": 0.008,
    "value_loss_coef": 0.60,
    "reward_energy_weight": 1.50,
    "reward_action_magnitude_weight": 0.20,
    "use_movement_budget": True,
}
MAIN_SETTINGS = [
    {"num_uavs": 2, "assignment_rule": "nearest_uav"},
    {"num_uavs": 3, "assignment_rule": "nearest_uav"},
]
ABLATION_SETTINGS = [
    {
        "label": "main",
        "description": "fixed energy_e30 PPO configuration",
        "overrides": dict(FINAL_MAIN_CONFIG),
    },
    {
        "label": "no_energy_shaped_reward",
        "description": "reward_energy_weight=0 and reward_action_magnitude_weight=0",
        "overrides": {
            **FINAL_MAIN_CONFIG,
            "reward_energy_weight": 0.0,
            "reward_action_magnitude_weight": 0.0,
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


def _load_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for stage-6 final plots. Install dependencies from 第四章/requirements.txt first."
        ) from exc
    return plt


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
            values = [float(item[metric]) for item in group]
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
) -> dict[str, Any]:
    train_episodes = int(overrides["train_episodes"])
    train_payload = run_marl_training(
        seed=seed,
        train_episodes=train_episodes,
        num_uavs=num_uavs,
        assignment_rule=assignment_rule,
        overrides={**overrides, "output_tag": output_tag},
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
        },
    )
    return {"train": train_payload, "eval": eval_payload}


def _run_compare_ch4(seeds: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _ensure_chapter3_import()
    from chapter3.experiments import compare_with_chapter4

    raw_rows: list[dict[str, Any]] = []
    for seed in seeds:
        comparison = compare_with_chapter4(seed=seed, episodes=1)["comparison"]
        for metric, payload in comparison.items():
            raw_rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "delta": 0.0 if payload["delta"] is None else float(payload["delta"]),
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
                        "completion_rate": float(metrics["completion_rate"]),
                        "average_latency": float(metrics["average_latency"]),
                        "total_energy": float(metrics["total_energy"]),
                        "cache_hit_rate": float(metrics["cache_hit_rate"]),
                        "fairness_uav_load": float(metrics["fairness_uav_load"]),
                    }
                )
    aggregated = _aggregate_rows(
        raw_rows,
        group_keys=["profile", "num_uavs", "assignment_rule"],
        metrics=["completion_rate", "average_latency", "total_energy", "cache_hit_rate", "fairness_uav_load"],
    )
    return raw_rows, aggregated


def _run_main_multiseed(seeds: list[int], *, eval_episodes: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[list[dict[str, Any]]]]]:
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
                overrides=dict(FINAL_MAIN_CONFIG),
            )
            marl_metrics = train_eval["eval"]["marl_metrics"]
            heuristic_metrics = train_eval["eval"]["heuristic_metrics"]
            raw_rows.append(
                {
                    "seed": seed,
                    "eval_seed": eval_seed,
                    "num_uavs": num_uavs,
                    "assignment_rule": assignment_rule,
                    "ppo_completion_rate": float(marl_metrics["completion_rate"]),
                    "ppo_average_latency": float(marl_metrics["average_latency"]),
                    "ppo_total_energy": float(marl_metrics["total_energy"]),
                    "ppo_cache_hit_rate": float(marl_metrics["cache_hit_rate"]),
                    "ppo_fairness_uav_load": float(marl_metrics["fairness_uav_load"]),
                    "heuristic_completion_rate": float(heuristic_metrics["completion_rate"]),
                    "heuristic_average_latency": float(heuristic_metrics["average_latency"]),
                    "heuristic_total_energy": float(heuristic_metrics["total_energy"]),
                    "heuristic_cache_hit_rate": float(heuristic_metrics["cache_hit_rate"]),
                    "heuristic_fairness_uav_load": float(heuristic_metrics["fairness_uav_load"]),
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
            "ppo_completion_rate",
            "ppo_average_latency",
            "ppo_total_energy",
            "ppo_cache_hit_rate",
            "ppo_fairness_uav_load",
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
    return raw_rows, aggregated, training_logs


def _run_ablation_multiseed(
    seeds: list[int],
    *,
    eval_episodes: int,
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
                overrides=dict(ablation["overrides"]),
            )
            marl_metrics = train_eval["eval"]["marl_metrics"]
            heuristic_metrics = train_eval["eval"]["heuristic_metrics"]
            raw_rows.append(
                {
                    "seed": seed,
                    "eval_seed": eval_seed,
                    "variant": label,
                    "description": str(ablation["description"]),
                    "completion_rate": float(marl_metrics["completion_rate"]),
                    "average_latency": float(marl_metrics["average_latency"]),
                    "total_energy": float(marl_metrics["total_energy"]),
                    "cache_hit_rate": float(marl_metrics["cache_hit_rate"]),
                    "fairness_uav_load": float(marl_metrics["fairness_uav_load"]),
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
            "completion_rate",
            "average_latency",
            "total_energy",
            "cache_hit_rate",
            "fairness_uav_load",
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


def _plot_training_curve(training_logs: dict[str, list[list[dict[str, Any]]]], output_path: Path) -> None:
    plt = _load_matplotlib()
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = {
        "main": "#2E86AB",
        "no_energy_shaped_reward": "#E67E22",
        "no_movement_budget": "#C0392B",
    }
    for variant, logs in training_logs.items():
        episodes, mean_return, std_return = _aggregate_training_curve(logs, "team_return")
        _, mean_energy, std_energy = _aggregate_training_curve(logs, "total_energy")
        color = colors.get(variant, "#555555")
        axes[0].plot(episodes, mean_return, label=variant, color=color, linewidth=2.0)
        axes[0].fill_between(
            episodes,
            [value - delta for value, delta in zip(mean_return, std_return)],
            [value + delta for value, delta in zip(mean_return, std_return)],
            color=color,
            alpha=0.18,
        )
        axes[1].plot(episodes, mean_energy, label=variant, color=color, linewidth=2.0)
        axes[1].fill_between(
            episodes,
            [value - delta for value, delta in zip(mean_energy, std_energy)],
            [value + delta for value, delta in zip(mean_energy, std_energy)],
            color=color,
            alpha=0.18,
        )
    axes[0].set_ylabel("Team Return")
    axes[0].set_title("Final PPO Training Curves (mean ± std over 3 seeds)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Total Energy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_comparison_bars(
    main_rows: list[dict[str, Any]],
    assignment_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    plt = _load_matplotlib()
    figure, axes = plt.subplots(3, 1, figsize=(11, 12))

    main_labels = [f"u{row['num_uavs']}" for row in main_rows]
    positions = list(range(len(main_labels)))
    width = 0.35
    axes[0].bar(
        [index - width / 2 for index in positions],
        [row["ppo_total_energy_mean"] for row in main_rows],
        width=width,
        yerr=[row["ppo_total_energy_std"] for row in main_rows],
        label="PPO",
        color="#2E86AB",
        capsize=4,
    )
    axes[0].bar(
        [index + width / 2 for index in positions],
        [row["heuristic_total_energy_mean"] for row in main_rows],
        width=width,
        yerr=[row["heuristic_total_energy_std"] for row in main_rows],
        label="heuristic",
        color="#E74C3C",
        capsize=4,
    )
    axes[0].set_title("PPO vs heuristic energy")
    axes[0].set_ylabel("Total Energy")
    axes[0].set_xticks(positions, main_labels)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    assignment_labels = [f"u{row['num_uavs']}-{row['assignment_rule']}" for row in assignment_rows]
    axes[1].bar(
        list(range(len(assignment_labels))),
        [row["fairness_uav_load_mean"] for row in assignment_rows],
        yerr=[row["fairness_uav_load_std"] for row in assignment_rows],
        color="#27AE60",
        capsize=4,
    )
    axes[1].set_title("Assignment fairness (sensitive profile)")
    axes[1].set_ylabel("Fairness")
    axes[1].set_xticks(list(range(len(assignment_labels))), assignment_labels, rotation=20)
    axes[1].grid(axis="y", alpha=0.3)

    ablation_labels = [row["variant"] for row in ablation_rows]
    axes[2].bar(
        list(range(len(ablation_labels))),
        [row["total_energy_mean"] for row in ablation_rows],
        yerr=[row["total_energy_std"] for row in ablation_rows],
        color=["#2E86AB", "#F39C12", "#C0392B"],
        capsize=4,
    )
    axes[2].set_title("Ablation energy on u2 nearest_uav")
    axes[2].set_ylabel("Total Energy")
    axes[2].set_xticks(list(range(len(ablation_labels))), ablation_labels, rotation=15)
    axes[2].grid(axis="y", alpha=0.3)

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


def run_final_paper_package(*, seeds: list[int] | None = None, eval_episodes: int = 4) -> dict[str, Any]:
    seeds = list(seeds or DEFAULT_SEEDS)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    compare_raw, compare_agg = _run_compare_ch4(seeds)
    assignment_raw, assignment_agg = _run_assignment_multiseed(seeds, eval_episodes=eval_episodes)
    main_raw, main_agg, main_training_logs = _run_main_multiseed(seeds, eval_episodes=eval_episodes)
    ablation_raw, ablation_agg, ablation_training_logs = _run_ablation_multiseed(seeds, eval_episodes=eval_episodes)

    training_curve_path = FIGURES_DIR / "final_training_curve_mean_std.png"
    comparison_bar_path = FIGURES_DIR / "final_comparison_bars.png"
    _plot_training_curve(
        {
            "main": ablation_training_logs["main"],
            "no_energy_shaped_reward": ablation_training_logs["no_energy_shaped_reward"],
            "no_movement_budget": ablation_training_logs["no_movement_budget"],
        },
        training_curve_path,
    )
    _plot_comparison_bars(main_agg, assignment_agg, ablation_agg, comparison_bar_path)

    tables = _write_tables(
        compare_rows=compare_agg,
        assignment_rows=assignment_agg,
        main_rows=main_agg,
        ablation_rows=ablation_agg,
    )

    package_manifest = {
        "final_main_config": FINAL_MAIN_CONFIG,
        "seeds": seeds,
        "eval_episodes": eval_episodes,
        "one_click_commands": [
            "python -m venv .venv",
            ".\\.venv\\Scripts\\python.exe -m pip install -r 第四章/requirements.txt",
            ".\\.venv\\Scripts\\python.exe 第三章/run_experiment.py --episodes 1 --compare-ch4 --seed 42",
            ".\\.venv\\Scripts\\python.exe 第四章/run_finalize_paper.py --seeds 42 52 62 --eval-episodes 4",
        ],
        "result_directories": {
            "stage5": str(RESULTS_DIR / "paper_stage5"),
            "stage6": str(FINAL_DIR),
            "root_results": str(RESULTS_DIR),
        },
        "tables": tables,
        "figures": {
            "training_curve": str(training_curve_path),
            "comparison_bars": str(comparison_bar_path),
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
        "final_main_config": FINAL_MAIN_CONFIG,
        "seeds": seeds,
        "compare_ch4_summary_path": str(FINAL_DIR / "compare_ch4_multiseed_summary.json"),
        "assignment_summary_path": str(FINAL_DIR / "assignment_multiseed_summary.json"),
        "ppo_vs_heuristic_summary_path": str(FINAL_DIR / "ppo_vs_heuristic_multiseed_summary.json"),
        "ablation_summary_path": str(FINAL_DIR / "ablation_multiseed_summary.json"),
        "tables": tables,
        "figures": {
            "training_curve": str(training_curve_path),
            "comparison_bars": str(comparison_bar_path),
        },
        "package_manifest_path": str(FINAL_DIR / "reproducibility_package.json"),
    }
