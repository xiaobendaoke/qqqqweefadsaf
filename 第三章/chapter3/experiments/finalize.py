"""第三章图表打包模块。

该模块负责运行第三章四种单 UAV 策略的统一实验，
并输出论文展示常用的对比图、轨迹拼图和与第四章的一致性验证图。
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from common.uav_mec.logging_utils import write_json
from common.uav_mec.plot_i18n import ENERGY_COMPONENT_LABEL_CN, configure_matplotlib_for_chinese, metric_label, policy_label

from .experiment import compare_with_chapter4, run_experiment


CHAPTER3_RESULTS = Path(__file__).resolve().parents[2] / "results"
FINAL_DIR = CHAPTER3_RESULTS / "paper_chapter3"
TABLES_DIR = FINAL_DIR / "tables"
FIGURES_DIR = FINAL_DIR / "figures"

POLICY_STYLES = {
    "heuristic": {"label": policy_label("heuristic"), "color": "#2E86AB"},
    "mpc": {"label": policy_label("mpc"), "color": "#E67E22"},
    "fixed_point": {"label": policy_label("fixed_point"), "color": "#4CAF50"},
    "fixed_patrol": {"label": policy_label("fixed_patrol"), "color": "#C0392B"},
}
ENERGY_COMPONENTS = [
    ("uav_move_energy", ENERGY_COMPONENT_LABEL_CN["uav_move_energy"]),
    ("uav_compute_energy", ENERGY_COMPONENT_LABEL_CN["uav_compute_energy"]),
    ("ue_local_energy", ENERGY_COMPONENT_LABEL_CN["ue_local_energy"]),
    ("ue_uplink_energy", ENERGY_COMPONENT_LABEL_CN["ue_uplink_energy"]),
    ("bs_compute_energy", ENERGY_COMPONENT_LABEL_CN["bs_compute_energy"]),
    ("relay_fetch_energy", ENERGY_COMPONENT_LABEL_CN["relay_fetch_energy"]),
    ("bs_fetch_tx_energy", ENERGY_COMPONENT_LABEL_CN["bs_fetch_tx_energy"]),
]


def _load_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for Chapter 3 figure generation. "
            "Use the project virtual environment, for example "
            "`.venv\\Scripts\\python.exe 第三章/run_finalize_chapter3.py ...`."
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


def _format_metric(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "null"
    return f"{float(value):.{digits}f}"


def _metric_stats(values: list[float]) -> tuple[float, float]:
    mean = float(statistics.fmean(values))
    std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    return mean, std


def _style_axis(axis: Any) -> None:
    axis.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _episode_metric_stats(result: dict[str, Any], metric: str) -> tuple[float, float]:
    values = [
        float(item["metrics"][metric])
        for item in result["episode_summaries"]
        if item["metrics"].get(metric) is not None
    ]
    return _metric_stats(values)


def _episode_metric_values(result: dict[str, Any], metric: str) -> list[float]:
    return [
        float(item["metrics"][metric])
        for item in result["episode_summaries"]
        if item["metrics"].get(metric) is not None
    ]


def _aggregate_step_metric(result: dict[str, Any], metric: str) -> tuple[list[int], list[float], list[float]]:
    episode_logs = result["episode_logs"]
    if not episode_logs:
        return [], [], []
    steps = [int(item["step"]) for item in episode_logs[0]["step_signals"]]
    means: list[float] = []
    stds: list[float] = []
    for step_index in range(len(steps)):
        values = [
            float(episode_log["step_signals"][step_index][metric])
            for episode_log in episode_logs
            if episode_log["step_signals"][step_index].get(metric) is not None
        ]
        mean, std = _metric_stats(values)
        means.append(mean)
        stds.append(std)
    return steps, means, stds


def _load_trajectory_payload(result: dict[str, Any]) -> dict[str, Any]:
    if not result["trajectory_exports"]:
        raise RuntimeError("Trajectory export is missing; Chapter 3 figure package expects export_trajectory=True.")
    trajectory_json = result["trajectory_exports"][0]["json"]
    return json.loads(Path(trajectory_json).read_text(encoding="utf-8"))


def _draw_trajectory(axis: Any, *, payload: dict[str, Any]) -> None:
    area = payload["area"]
    axis.set_xlim(0.0, area["width"])
    axis.set_ylim(0.0, area["height"])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X 坐标 (m)")
    axis.set_ylabel("Y 坐标 (m)")
    axis.grid(alpha=0.22, linestyle="--", linewidth=0.6)

    for user_trace in payload["user_paths"]:
        samples = user_trace["samples"]
        xs = [sample["x"] for sample in samples]
        ys = [sample["y"] for sample in samples]
        axis.plot(xs, ys, color="#C8C8C8", linewidth=0.8, alpha=0.55, linestyle="--")

    uav_path = payload["uav_path"]
    uav_xs = [sample["x"] for sample in uav_path]
    uav_ys = [sample["y"] for sample in uav_path]
    axis.plot(uav_xs, uav_ys, color="#005F73", linewidth=2.2, marker="o", markersize=3.8)
    axis.scatter(uav_xs[0], uav_ys[0], color="#2A9D8F", s=42, marker="s")
    axis.scatter(uav_xs[-1], uav_ys[-1], color="#D62828", s=54, marker="*")

    stride = max(1, len(uav_path) // 7)
    for index, sample in enumerate(uav_path):
        if index % stride != 0 and index != len(uav_path) - 1:
            continue
        axis.annotate(
            f"{int(sample['step'])}",
            (sample["x"], sample["y"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color="#202020",
        )

    if len(uav_path) >= 2:
        axis.annotate(
            "",
            xy=(uav_xs[-1], uav_ys[-1]),
            xytext=(uav_xs[-2], uav_ys[-2]),
            arrowprops={"arrowstyle": "->", "color": "#005F73", "lw": 1.4},
        )


def _plot_policy_metrics(results_by_policy: dict[str, dict[str, Any]], output_path: Path) -> None:
    plt = _load_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(11.6, 7.8))
    policies = list(POLICY_STYLES.keys())
    labels = [POLICY_STYLES[policy]["label"] for policy in policies]
    colors = [POLICY_STYLES[policy]["color"] for policy in policies]
    positions = list(range(len(policies)))
    panels = [
        ("completion_rate", "任务完成率", "比例"),
        ("average_latency", "平均时延", "时延"),
        ("total_energy", "总能耗", "能耗"),
        ("cache_hit_rate", "缓存命中率", "比例"),
    ]

    for axis, (metric, title, ylabel) in zip(axes.flat, panels):
        means: list[float] = []
        stds: list[float] = []
        for policy in policies:
            mean, std = _episode_metric_stats(results_by_policy[policy], metric)
            means.append(mean)
            stds.append(std)
        axis.bar(positions, means, yerr=stds, color=colors, capsize=4, width=0.65)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, labels, rotation=12)
        if metric in {"completion_rate", "cache_hit_rate"}:
            axis.set_ylim(0.0, 1.05)
        _style_axis(axis)

    figure.suptitle("第三章策略对比", y=0.98)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_step_curves(results_by_policy: dict[str, dict[str, Any]], output_path: Path) -> None:
    plt = _load_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(11.6, 7.8), sharex=True)
    panels = [
        ("step_completion_ratio", "逐步任务完成占比", "比例"),
        ("step_average_latency", "逐步平均时延", "时延"),
        ("step_total_energy", "逐步总能耗", "能耗"),
        ("step_cache_hit_ratio", "逐步缓存命中占比", "比例"),
    ]

    for axis, (metric, title, ylabel) in zip(axes.flat, panels):
        for policy, result in results_by_policy.items():
            steps, means, stds = _aggregate_step_metric(result, metric)
            color = POLICY_STYLES[policy]["color"]
            axis.plot(steps, means, color=color, linewidth=2.0, label=POLICY_STYLES[policy]["label"])
            axis.fill_between(
                steps,
                [value - delta for value, delta in zip(means, stds)],
                [value + delta for value, delta in zip(means, stds)],
                color=color,
                alpha=0.12,
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("步骤")
        if metric in {"step_completion_ratio", "step_cache_hit_ratio"}:
            axis.set_ylim(0.0, 1.05)
        _style_axis(axis)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.suptitle("第三章逐步行为曲线（均值 ± 标准差）", y=0.98)
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_energy_breakdown(results_by_policy: dict[str, dict[str, Any]], output_path: Path) -> None:
    plt = _load_matplotlib()
    figure, axis = plt.subplots(figsize=(11.6, 7.0))
    policies = list(POLICY_STYLES.keys())
    labels = [POLICY_STYLES[policy]["label"] for policy in policies]
    positions = list(range(len(policies)))
    bottoms = [0.0] * len(policies)
    palette = ["#0F4C5C", "#2E8B57", "#7FB069", "#F4A259", "#BC4749", "#8E5A9B"]

    for component_index, (metric, component_label) in enumerate(ENERGY_COMPONENTS):
        means = [
            _metric_stats(_episode_metric_values(results_by_policy[policy], metric))[0]
            for policy in policies
        ]
        axis.bar(
            positions,
            means,
            bottom=bottoms,
            color=palette[component_index % len(palette)],
            width=0.68,
            label=component_label,
            alpha=0.92,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, means)]

    total_means = [
        _metric_stats(_episode_metric_values(results_by_policy[policy], "total_energy"))[0]
        for policy in policies
    ]
    for position, total in zip(positions, total_means):
        axis.text(position, total, f"{total:.1f}", ha="center", va="bottom", fontsize=9, color="#222222")

    axis.set_xticks(positions, labels, rotation=12)
    axis.set_ylabel("能耗 (J)")
    axis.set_title("第三章各策略能耗分解")
    _style_axis(axis)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_trajectory_grid(results_by_policy: dict[str, dict[str, Any]], output_path: Path) -> None:
    plt = _load_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(11.6, 10.4))
    for axis, policy in zip(axes.flat, POLICY_STYLES.keys()):
        payload = _load_trajectory_payload(results_by_policy[policy])
        _draw_trajectory(axis, payload=payload)
        summary = payload["summary_metrics"]
        axis.set_title(
            f"{POLICY_STYLES[policy]['label']}\n"
            f"完成率={_format_metric(summary.get('completion_rate'), digits=3)}  "
            f"总能耗={_format_metric(summary.get('total_energy'), digits=3)}"
        )

    figure.suptitle("第三章单无人机轨迹对比", y=0.98)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_compare_delta(compare_result: dict[str, Any], output_path: Path) -> None:
    plt = _load_matplotlib()
    rows = [
        {"metric": metric, "delta": payload["delta"]}
        for metric, payload in compare_result["comparison"].items()
        if payload["delta"] is not None
    ]
    labels = [metric_label(str(row["metric"])) for row in rows]
    values = [float(row["delta"]) for row in rows]
    colors = ["#7A7A7A" if math.isclose(value, 0.0, abs_tol=1e-12) else "#E76F51" for value in values]
    max_abs = max((abs(value) for value in values), default=0.0)
    all_zero = bool(values) and all(math.isclose(value, 0.0, abs_tol=1e-12) for value in values)
    if max_abs <= 1.0e-12:
        max_abs = 1.0e-6

    figure, axis = plt.subplots(figsize=(9.6, 6.4))
    positions = list(range(len(labels)))
    axis.barh(positions, values, color=colors, alpha=0.9)
    axis.scatter([0.0] * len(positions), positions, color="#444444", s=18, zorder=3)
    axis.axvline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    axis.set_yticks(positions, labels)
    axis.set_xlabel("第三章 - 第四章 指标差值")
    axis.set_xlim(-1.15 * max_abs, 1.15 * max_abs)
    axis.set_title("第三章与第四章（无人机数量=1）差值校验")
    if all_zero:
        axis.text(
            0.98,
            0.04,
            "所有对比指标完全一致（差值 = 0）。",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#444444",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F3F4F6", "edgecolor": "#D1D5DB"},
        )
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _summary_rows(results_by_policy: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy, result in results_by_policy.items():
        metrics = result["averaged_metrics"]
        rows.append(
            {
                "policy": policy,
                "policy_label": POLICY_STYLES[policy]["label"],
                "completion_rate": float(metrics["completion_rate"]),
                "average_latency": float(metrics["average_latency"]),
                "total_energy": float(metrics["total_energy"]),
                "cache_hit_rate": float(metrics["cache_hit_rate"]),
                "deadline_violation_rate": float(metrics["deadline_violation_rate"]),
                "reliability_violation_rate": float(metrics["reliability_violation_rate"]),
                "average_latency_completed": float(metrics["average_latency_completed"]),
                "latency_per_generated_task": float(metrics["latency_per_generated_task"]),
                "trajectory_json": result["trajectory_exports"][0]["json"] if result["trajectory_exports"] else "",
                "trajectory_png": result["trajectory_exports"][0]["png"] if result["trajectory_exports"] else "",
                **{
                    metric: float(metrics[metric])
                    for metric, _label in ENERGY_COMPONENTS
                },
            }
        )
    return rows


def _write_tables(rows: list[dict[str, Any]], *, output_prefix: str) -> dict[str, str]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / f"{output_prefix}_policy_summary.csv"
    md_path = TABLES_DIR / f"{output_prefix}_policy_summary.md"
    energy_md_path = TABLES_DIR / f"{output_prefix}_energy_breakdown.md"
    _write_csv(csv_path, rows)
    lines = [
        "| policy | completion_rate | average_latency | total_energy | cache_hit_rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy_label']} | {row['completion_rate']:.4f} | {row['average_latency']:.4f} | "
            f"{row['total_energy']:.4f} | {row['cache_hit_rate']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    energy_lines = [
        "| policy | uav_move | uav_compute | ue_local | ue_uplink | bs_compute | relay_fetch | bs_fetch_tx |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        energy_lines.append(
            f"| {row['policy_label']} | {row['uav_move_energy']:.4f} | {row['uav_compute_energy']:.4f} | "
            f"{row['ue_local_energy']:.4f} | {row['ue_uplink_energy']:.4f} | "
            f"{row['bs_compute_energy']:.4f} | {row['relay_fetch_energy']:.4f} | {row['bs_fetch_tx_energy']:.4f} |"
        )
    energy_md_path.write_text("\n".join(energy_lines) + "\n", encoding="utf-8")
    return {
        "policy_summary_csv": str(csv_path),
        "policy_summary_markdown": str(md_path),
        "energy_breakdown_markdown": str(energy_md_path),
    }


def run_chapter3_figure_package(
    *,
    seed: int = 42,
    episodes: int = 8,
    hard: bool = False,
    steps_per_episode: int = 20,
    compare_episodes: int | None = None,
) -> dict[str, Any]:
    """运行第三章四策略实验并生成常用图表结果包。"""
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    compare_episodes = int(compare_episodes if compare_episodes is not None else episodes)
    profile = "hard" if hard else "default"
    output_prefix = f"{profile}_s{int(steps_per_episode)}"

    results_by_policy: dict[str, dict[str, Any]] = {}
    for policy in POLICY_STYLES:
        results_by_policy[policy] = run_experiment(
            seed=seed,
            episodes=episodes,
            hard=hard,
            policy=policy,
            export_trajectory=True,
            steps_per_episode=steps_per_episode,
        )

    compare_result = compare_with_chapter4(
        seed=seed,
        episodes=compare_episodes,
        hard=hard,
        steps_per_episode=steps_per_episode,
    )

    metrics_path = FIGURES_DIR / f"{output_prefix}_policy_metrics.png"
    step_curves_path = FIGURES_DIR / f"{output_prefix}_step_curves.png"
    trajectories_path = FIGURES_DIR / f"{output_prefix}_trajectory_grid.png"
    delta_path = FIGURES_DIR / f"{output_prefix}_compare_ch4_delta.png"
    energy_path = FIGURES_DIR / f"{output_prefix}_energy_breakdown.png"

    _plot_policy_metrics(results_by_policy, metrics_path)
    _plot_step_curves(results_by_policy, step_curves_path)
    _plot_trajectory_grid(results_by_policy, trajectories_path)
    _plot_compare_delta(compare_result, delta_path)
    _plot_energy_breakdown(results_by_policy, energy_path)

    summary_rows = _summary_rows(results_by_policy)
    tables = _write_tables(summary_rows, output_prefix=output_prefix)
    summary_path = FINAL_DIR / f"{output_prefix}_figure_package.json"
    payload = {
        "seed": seed,
        "episodes": episodes,
        "compare_episodes": compare_episodes,
        "profile": profile,
        "steps_per_episode": int(steps_per_episode),
        "policy_summary": summary_rows,
        "compare_ch4": compare_result,
        "figures": {
            "policy_metrics": str(metrics_path),
            "step_curves": str(step_curves_path),
            "trajectory_grid": str(trajectories_path),
            "compare_ch4_delta": str(delta_path),
            "energy_breakdown": str(energy_path),
        },
        "tables": tables,
    }
    write_json(summary_path, payload)
    payload["summary_path"] = str(summary_path)
    return payload
