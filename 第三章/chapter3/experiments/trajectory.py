from __future__ import annotations

from pathlib import Path
from typing import Any

from common.uav_mec.logging_utils import write_json


class EpisodeTrajectoryRecorder:
    def __init__(self, *, env: Any, episode_index: int, seed: int, policy: str, profile: str) -> None:
        self.env = env
        self.episode_index = episode_index
        self.seed = seed
        self.policy = policy
        self.profile = profile
        self.area = {
            "width": float(env.config.area_width),
            "height": float(env.config.area_height),
        }
        self.uav_path: list[dict[str, float]] = []
        self.user_paths: dict[int, list[dict[str, float]]] = {user.user_id: [] for user in env.users}
        self.step_metrics: list[dict[str, float | None]] = []
        self._capture(step_index=0)

    def _capture(self, *, step_index: int) -> None:
        if self.env.uavs:
            uav = self.env.uavs[0]
            self.uav_path.append(
                {
                    "step": float(step_index),
                    "x": float(uav.position[0]),
                    "y": float(uav.position[1]),
                    "energy_ratio": float(uav.energy_ratio),
                }
            )
        for user in self.env.users:
            self.user_paths.setdefault(user.user_id, []).append(
                {
                    "step": float(step_index),
                    "x": float(user.position[0]),
                    "y": float(user.position[1]),
                }
            )

    def record_step(self, *, step_index: int, metrics: dict[str, float | None]) -> None:
        self._capture(step_index=step_index)
        self.step_metrics.append(
            {
                "step": float(step_index),
                "completion_rate": metrics.get("completion_rate"),
                "average_latency": metrics.get("average_latency"),
                "total_energy": metrics.get("total_energy"),
                "cache_hit_rate": metrics.get("cache_hit_rate"),
            }
        )

    def build_payload(self, *, summary_metrics: dict[str, float | None]) -> dict[str, Any]:
        return {
            "chapter": "chapter3",
            "profile": self.profile,
            "policy": self.policy,
            "episode_index": self.episode_index,
            "seed": self.seed,
            "area": self.area,
            "uav_path": self.uav_path,
            "user_paths": [
                {"user_id": user_id, "samples": samples} for user_id, samples in sorted(self.user_paths.items())
            ],
            "step_metrics": self.step_metrics,
            "summary_metrics": summary_metrics,
        }


def export_trajectory_artifacts(
    *,
    recorder: EpisodeTrajectoryRecorder,
    summary_metrics: dict[str, float | None],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"trajectory_{recorder.profile}_{recorder.policy}_seed{recorder.seed}_ep{recorder.episode_index}"
    json_path = output_dir / f"{stem}.json"
    png_path = output_dir / f"{stem}.png"

    payload = recorder.build_payload(summary_metrics=summary_metrics)
    write_json(json_path, payload)
    _plot_trajectory(payload=payload, output_path=png_path)
    return {
        "json": str(json_path),
        "png": str(png_path),
    }


def _plot_trajectory(*, payload: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    area = payload["area"]
    uav_path = payload["uav_path"]
    user_paths = payload["user_paths"]
    summary_metrics = payload["summary_metrics"]

    figure, axis = plt.subplots(figsize=(8.0, 7.4), dpi=180)
    axis.set_xlim(0.0, area["width"])
    axis.set_ylim(0.0, area["height"])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X Position (m)")
    axis.set_ylabel("Y Position (m)")
    axis.set_title(
        "Chapter 3 UAV Trajectory\n"
        f"policy={payload['policy']}  seed={payload['seed']}  "
        f"completion={_format_metric(summary_metrics.get('completion_rate'))}  "
        f"energy={_format_metric(summary_metrics.get('total_energy'))}"
    )
    axis.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    for user_trace in user_paths:
        samples = user_trace["samples"]
        xs = [sample["x"] for sample in samples]
        ys = [sample["y"] for sample in samples]
        axis.plot(xs, ys, color="#c7c7c7", linewidth=0.8, alpha=0.55, linestyle="--")
        axis.scatter(xs[0], ys[0], color="#8c8c8c", s=14, alpha=0.9)
        axis.scatter(xs[-1], ys[-1], color="#595959", s=14, alpha=0.9)

    uav_xs = [sample["x"] for sample in uav_path]
    uav_ys = [sample["y"] for sample in uav_path]
    axis.plot(uav_xs, uav_ys, color="#005f73", linewidth=2.2, marker="o", markersize=4.5, label="UAV path")
    axis.scatter(uav_xs[0], uav_ys[0], color="#2a9d8f", s=60, marker="s", label="start")
    axis.scatter(uav_xs[-1], uav_ys[-1], color="#d62828", s=70, marker="*", label="end")

    label_stride = max(1, len(uav_path) // 8)
    for index, sample in enumerate(uav_path):
        if index % label_stride != 0 and index != len(uav_path) - 1:
            continue
        axis.annotate(
            f"{int(sample['step'])}",
            (sample["x"], sample["y"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color="#0a0a0a",
        )

    if len(uav_path) >= 2:
        axis.annotate(
            "",
            xy=(uav_xs[-1], uav_ys[-1]),
            xytext=(uav_xs[-2], uav_ys[-2]),
            arrowprops={"arrowstyle": "->", "color": "#005f73", "lw": 1.5},
        )

    axis.legend(loc="best", frameon=True)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "null"
    return f"{float(value):.3f}"
