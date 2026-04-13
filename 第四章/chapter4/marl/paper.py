from __future__ import annotations

import csv
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
]


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


def _metric_sort_value(value: float | None, *, prefer_high: bool) -> tuple[int, float]:
    if value is None:
        return (1, 0.0)
    return (0, -float(value) if prefer_high else float(value))


def _select_best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        rows,
        key=lambda row: (
            _metric_sort_value(row["completion_rate"], prefer_high=True),
            _metric_sort_value(row["total_energy"], prefer_high=False),
            _metric_sort_value(row["average_latency"], prefer_high=False),
        ),
    )


def _build_eval_overrides(train_payload: dict[str, Any], *, output_tag: str) -> dict[str, Any]:
    config = train_payload["config"]
    return {
        "output_tag": output_tag,
        "use_movement_budget": bool(config.get("use_movement_budget", True)),
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
) -> dict[str, Any]:
    merged = dict(overrides)
    train_episodes = int(merged.pop("train_episodes"))
    merged["output_tag"] = output_tag
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
        overrides=_build_eval_overrides(train_payload, output_tag=output_tag),
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


def _run_tuning(
    *,
    seed: int,
    eval_seed: int,
    eval_episodes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    tuning_rows: list[dict[str, Any]] = []
    tuning_runs: dict[str, Any] = {}
    for candidate in TUNING_CANDIDATES:
        output_tag = f"tune_{candidate['name']}"
        train_eval = _run_train_eval(
            seed=seed,
            eval_seed=eval_seed,
            eval_episodes=eval_episodes,
            num_uavs=MAIN_NUM_UAVS,
            assignment_rule=MAIN_ASSIGNMENT_RULE,
            output_tag=output_tag,
            overrides=candidate["overrides"],
        )
        row = _summarize_eval_result(label=candidate["name"], train_eval=train_eval)
        row["description"] = candidate["description"]
        tuning_rows.append(row)
        tuning_runs[candidate["name"]] = train_eval
    selected_row = _select_best_candidate(tuning_rows)
    return tuning_rows, selected_row, tuning_runs[selected_row["label"]]


def _plot_training_curves(
    *,
    main_train_log: list[dict[str, Any]],
    ablation_logs: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    plt = _load_matplotlib()
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    def plot_series(logs: list[dict[str, Any]], *, label: str, key: str, axis: Any) -> None:
        axis.plot([entry["episode"] for entry in logs], [entry[key] for entry in logs], label=label, linewidth=2.0)

    plot_series(main_train_log, label="main", key="team_return", axis=axes[0])
    plot_series(main_train_log, label="main", key="total_energy", axis=axes[1])
    for label, logs in ablation_logs.items():
        plot_series(logs, label=label, key="team_return", axis=axes[0])
        plot_series(logs, label=label, key="total_energy", axis=axes[1])

    axes[0].set_ylabel("Team Return")
    axes[0].set_title("PPO Training Curves")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Total Energy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_assignment_rules(rows: list[dict[str, Any]], output_path: Path) -> None:
    plt = _load_matplotlib()
    labels = [f"u{row['num_uavs']}-{row['assignment_rule']}" for row in rows]
    energy = [row["total_energy"] for row in rows]
    fairness = [row["fairness_uav_load"] if row["fairness_uav_load"] is not None else 0.0 for row in rows]
    positions = list(range(len(labels)))

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(positions, energy, color="#4C78A8")
    axes[0].set_ylabel("Total Energy")
    axes[0].set_title("Assignment Rule Comparison (Sensitive Profile)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(positions, fairness, color="#F58518")
    axes[1].set_ylabel("Fairness")
    axes[1].set_xticks(positions, labels, rotation=20)
    axes[1].grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_main_comparison(rows: list[dict[str, Any]], output_path: Path) -> None:
    plt = _load_matplotlib()
    labels = [f"u{row['num_uavs']}-ppo" for row in rows] + [f"u{row['num_uavs']}-heuristic" for row in rows]
    energy = [row["total_energy"] for row in rows] + [row["heuristic_total_energy"] for row in rows]
    latency = [row["average_latency"] for row in rows] + [row["heuristic_average_latency"] for row in rows]
    positions = list(range(len(labels)))

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(positions, energy, color=["#54A24B"] * len(rows) + ["#E45756"] * len(rows))
    axes[0].set_ylabel("Total Energy")
    axes[0].set_title("PPO vs Heuristic")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(positions, latency, color=["#54A24B"] * len(rows) + ["#E45756"] * len(rows))
    axes[1].set_ylabel("Average Latency")
    axes[1].set_xticks(positions, labels, rotation=20)
    axes[1].grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _make_markdown_summary(
    *,
    final_config: dict[str, Any],
    chapter_compare: dict[str, Any],
    assignment_rows: list[dict[str, Any]],
    main_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
) -> str:
    comparison = chapter_compare["comparison"]
    lines = [
        "# 第五阶段实验汇总",
        "",
        "## 最终 PPO 主配置",
        "",
        f"- output_tag: `{final_config['output_tag']}`",
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
        "| metric | delta |",
        "| --- | ---: |",
    ]
    for metric, payload in comparison.items():
        lines.append(f"| {metric} | {payload['delta']} |")

    lines.extend(
        [
            "",
            "## B. nearest_uav vs least_loaded_uav",
            "",
            "| setting | completion_rate | average_latency | total_energy | fairness_uav_load |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in assignment_rows:
        lines.append(
            f"| u{row['num_uavs']} {row['assignment_rule']} | {row['completion_rate']} | {row['average_latency']} | {row['total_energy']} | {row['fairness_uav_load']} |"
        )

    lines.extend(
        [
            "",
            "## C. PPO vs heuristic",
            "",
            "| setting | PPO completion | PPO latency | PPO energy | heuristic energy | delta energy |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in main_rows:
        lines.append(
            f"| u{row['num_uavs']} {row['assignment_rule']} | {row['completion_rate']} | {row['average_latency']} | {row['total_energy']} | {row['heuristic_total_energy']} | {row['delta_total_energy']} |"
        )

    lines.extend(
        [
            "",
            "## D. 最小消融",
            "",
            "| variant | completion_rate | average_latency | total_energy | delta energy vs heuristic |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in ablation_rows:
        lines.append(
            f"| {row['label']} | {row['completion_rate']} | {row['average_latency']} | {row['total_energy']} | {row['delta_total_energy']} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_paper_experiments(*, seed: int = 42, eval_seed: int = 142, eval_episodes: int = 4) -> dict[str, Any]:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    tuning_rows, selected_tuning_row, _ = _run_tuning(seed=seed, eval_seed=eval_seed, eval_episodes=eval_episodes)

    final_overrides = {
        "train_episodes": int(selected_tuning_row["train_episodes"]),
        "actor_lr": float(selected_tuning_row["actor_lr"]),
        "critic_lr": float(selected_tuning_row["critic_lr"]),
        "ppo_clip_eps": float(selected_tuning_row["clip_ratio"]),
        "entropy_coef": float(selected_tuning_row["entropy_coef"]),
        "value_loss_coef": float(selected_tuning_row["value_coef"]),
        "reward_energy_weight": float(selected_tuning_row["reward_energy_weight"]),
        "reward_action_magnitude_weight": float(selected_tuning_row["reward_action_magnitude_weight"]),
        "use_movement_budget": bool(selected_tuning_row["use_movement_budget"]),
    }

    main_u2 = _run_train_eval(
        seed=seed,
        eval_seed=eval_seed,
        eval_episodes=eval_episodes,
        num_uavs=2,
        assignment_rule=MAIN_ASSIGNMENT_RULE,
        output_tag="paper_main_u2",
        overrides=final_overrides,
    )
    main_u3 = _run_train_eval(
        seed=seed,
        eval_seed=eval_seed,
        eval_episodes=eval_episodes,
        num_uavs=3,
        assignment_rule=MAIN_ASSIGNMENT_RULE,
        output_tag="paper_main_u3",
        overrides=final_overrides,
    )
    main_rows = [
        _summarize_eval_result(label="paper_main_u2", train_eval=main_u2),
        _summarize_eval_result(label="paper_main_u3", train_eval=main_u3),
    ]

    no_energy = _run_train_eval(
        seed=seed,
        eval_seed=eval_seed,
        eval_episodes=eval_episodes,
        num_uavs=MAIN_NUM_UAVS,
        assignment_rule=MAIN_ASSIGNMENT_RULE,
        output_tag="ablation_no_energy",
        overrides={
            **final_overrides,
            "reward_energy_weight": 0.0,
            "reward_action_magnitude_weight": 0.0,
        },
    )
    no_budget = _run_train_eval(
        seed=seed,
        eval_seed=eval_seed,
        eval_episodes=eval_episodes,
        num_uavs=MAIN_NUM_UAVS,
        assignment_rule=MAIN_ASSIGNMENT_RULE,
        output_tag="ablation_no_budget",
        overrides={
            **final_overrides,
            "use_movement_budget": False,
        },
    )
    ablation_rows = [
        _summarize_eval_result(label="no_energy_shaped_reward", train_eval=no_energy),
        _summarize_eval_result(label="no_movement_budget", train_eval=no_budget),
    ]

    assignment_rows: list[dict[str, Any]] = []
    for num_uavs in (2, 3):
        for assignment_rule in ("nearest_uav", "least_loaded_uav"):
            result = run_sensitive_experiment(
                seed=seed,
                episodes=eval_episodes,
                num_uavs=num_uavs,
                assignment_rule=assignment_rule,
            )
            metrics = result["averaged_metrics"]
            assignment_rows.append(
                {
                    "profile": "sensitive",
                    "num_uavs": num_uavs,
                    "assignment_rule": assignment_rule,
                    "completion_rate": _float_or_none(metrics["completion_rate"]),
                    "average_latency": _float_or_none(metrics["average_latency"]),
                    "total_energy": _float_or_none(metrics["total_energy"]),
                    "cache_hit_rate": _float_or_none(metrics["cache_hit_rate"]),
                    "fairness_uav_load": _float_or_none(metrics["fairness_uav_load"]),
                    "result_path": str(RESULTS_DIR / f"experiment_sensitive_u{num_uavs}_{assignment_rule}.json"),
                }
            )

    if str(CHAPTER3_DIR) not in sys.path:
        sys.path.insert(0, str(CHAPTER3_DIR))
    from chapter3.experiments import compare_with_chapter4

    chapter_compare = compare_with_chapter4(seed=seed, episodes=1)
    chapter_compare_path = PAPER_DIR / "chapter3_vs_chapter4_num_uavs1.json"
    write_json(chapter_compare_path, chapter_compare)

    tuning_json = PAPER_DIR / "tuning_summary.json"
    tuning_csv = PAPER_DIR / "tuning_summary.csv"
    main_json = PAPER_DIR / "main_experiment_matrix.json"
    main_csv = PAPER_DIR / "main_experiment_matrix.csv"
    ablation_json = PAPER_DIR / "ablation_summary.json"
    ablation_csv = PAPER_DIR / "ablation_summary.csv"
    assignment_json = PAPER_DIR / "assignment_rule_matrix.json"
    assignment_csv = PAPER_DIR / "assignment_rule_matrix.csv"
    config_notes_path = PAPER_DIR / "config_notes.json"
    summary_path = PAPER_DIR / "paper_summary.md"
    training_curve_path = PAPER_DIR / "ppo_training_curves.png"
    assignment_plot_path = PAPER_DIR / "assignment_rule_comparison.png"
    main_plot_path = PAPER_DIR / "ppo_vs_heuristic.png"

    final_config = build_marl_config(
        {
            **final_overrides,
            "seed": seed,
            "num_uavs": MAIN_NUM_UAVS,
            "assignment_rule": MAIN_ASSIGNMENT_RULE,
            "output_tag": "paper_main_u2",
        }
    ).to_dict()

    write_json(tuning_json, {"selected_candidate": selected_tuning_row["label"], "rows": tuning_rows})
    write_json(main_json, {"rows": main_rows})
    write_json(ablation_json, {"rows": ablation_rows})
    write_json(assignment_json, {"rows": assignment_rows})
    _write_csv(tuning_csv, tuning_rows)
    _write_csv(main_csv, main_rows)
    _write_csv(ablation_csv, ablation_rows)
    _write_csv(assignment_csv, assignment_rows)

    config_notes = {
        "tuning_protocol": {
            "seed": seed,
            "eval_seed": eval_seed,
            "eval_episodes": eval_episodes,
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
            "no_energy_shaped_reward": "reward_energy_weight=0.0 and reward_action_magnitude_weight=0.0",
            "no_movement_budget": "use_movement_budget=False with all other main PPO settings fixed",
            "no_centralized_critic": "not included in this stage to keep a single algorithm implementation",
        },
    }
    write_json(config_notes_path, config_notes)

    summary_text = _make_markdown_summary(
        final_config=final_config,
        chapter_compare=chapter_compare,
        assignment_rows=assignment_rows,
        main_rows=main_rows,
        ablation_rows=ablation_rows,
    )
    summary_path.write_text(summary_text, encoding="utf-8")

    _plot_training_curves(
        main_train_log=main_u2["train"]["training_log"],
        ablation_logs={
            "no_energy_shaped_reward": no_energy["train"]["training_log"],
            "no_movement_budget": no_budget["train"]["training_log"],
        },
        output_path=training_curve_path,
    )
    _plot_assignment_rules(assignment_rows, assignment_plot_path)
    _plot_main_comparison(main_rows, main_plot_path)

    payload = {
        "selected_final_config": final_config,
        "selected_tuning_candidate": selected_tuning_row["label"],
        "chapter3_vs_chapter4_path": str(chapter_compare_path),
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
        "main_rows": main_rows,
        "ablation_rows": ablation_rows,
    }
    write_json(PAPER_DIR / "paper_experiments_summary.json", payload)
    return payload
