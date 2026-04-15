from __future__ import annotations

from dataclasses import dataclass, field

from .task import Task


def _jain(values: list[float]) -> float | None:
    if not values:
        return None
    denom = len(values) * sum(v * v for v in values)
    if denom <= 0:
        return 0.0
    return (sum(values) ** 2) / denom


def _sum_energy_breakdown(breakdown: dict[str, float]) -> float:
    return float(sum(float(value) for value in breakdown.values()))


@dataclass(slots=True)
class MetricTracker:
    num_uavs: int = 1
    user_generated: dict[int, int] = field(default_factory=dict)
    user_completed: dict[int, int] = field(default_factory=dict)
    uav_loads: dict[int, int] = field(default_factory=dict)
    total_generated: int = 0
    total_completed: int = 0
    total_latency: float = 0.0
    total_energy: float = 0.0
    total_cache_hits: int = 0
    deadline_violations: int = 0
    reliability_violations: int = 0
    uav_move_energy: float = 0.0
    uav_compute_energy: float = 0.0
    ue_local_energy: float = 0.0
    ue_uplink_energy: float = 0.0
    bs_compute_energy: float = 0.0
    relay_fetch_energy: float = 0.0
    last_step_metrics: dict[str, float | None] = field(default_factory=dict)

    def record_generated(self, tasks: list[Task]) -> None:
        for task in tasks:
            self.total_generated += 1
            self.user_generated[task.user_id] = self.user_generated.get(task.user_id, 0) + 1

    def record_assignment(self, task: Task) -> None:
        if task.execution_target in {"uav", "collaborator"} and task.assigned_uav_id is not None:
            self.uav_loads[task.assigned_uav_id] = self.uav_loads.get(task.assigned_uav_id, 0) + 1

    def record_step(
        self,
        *,
        generated_tasks: list[Task],
        finalized_tasks: list[Task],
        pending_tasks: list[Task],
        energy_breakdown: dict[str, float],
    ) -> dict[str, float | None]:
        self.record_generated(generated_tasks)
        for task in finalized_tasks:
            self.total_latency += float(task.total_latency)
            if task.cache_hit:
                self.total_cache_hits += 1
            if task.completed:
                self.total_completed += 1
                self.user_completed[task.user_id] = self.user_completed.get(task.user_id, 0) + 1
            else:
                if task.total_latency > task.slack:
                    self.deadline_violations += 1
                if task.success_probability < task.required_reliability:
                    self.reliability_violations += 1

        self.uav_move_energy += float(energy_breakdown.get("uav_move_energy", 0.0))
        self.uav_compute_energy += float(energy_breakdown.get("uav_compute_energy", 0.0))
        self.ue_local_energy += float(energy_breakdown.get("ue_local_energy", 0.0))
        self.ue_uplink_energy += float(energy_breakdown.get("ue_uplink_energy", 0.0))
        self.bs_compute_energy += float(energy_breakdown.get("bs_compute_energy", 0.0))
        self.relay_fetch_energy += float(energy_breakdown.get("relay_fetch_energy", 0.0))
        self.total_energy += _sum_energy_breakdown(energy_breakdown)

        generated_count = len(generated_tasks)
        finalized_count = len(finalized_tasks)
        completed_count = sum(1 for task in finalized_tasks if task.completed)
        step_latency = sum(float(task.total_latency) for task in finalized_tasks) / finalized_count if finalized_count else 0.0
        step_cache_hits = sum(1 for task in finalized_tasks if task.cache_hit)
        step_deadline_violations = sum(1 for task in finalized_tasks if task.total_latency > task.slack)
        step_reliability_violations = sum(1 for task in finalized_tasks if task.success_probability < task.required_reliability)
        self.last_step_metrics = {
            "generated_tasks": float(generated_count),
            "finalized_tasks": float(finalized_count),
            "completed_tasks": float(completed_count),
            "pending_tasks": float(len(pending_tasks)),
            "average_latency": step_latency,
            "cache_hit_rate": (step_cache_hits / finalized_count) if finalized_count else 0.0,
            "completion_rate": (completed_count / finalized_count) if finalized_count else 0.0,
            "deadline_violation_rate": (step_deadline_violations / finalized_count) if finalized_count else 0.0,
            "reliability_violation_rate": (step_reliability_violations / finalized_count) if finalized_count else 0.0,
            "total_energy": _sum_energy_breakdown(energy_breakdown),
            "uav_move_energy": float(energy_breakdown.get("uav_move_energy", 0.0)),
            "uav_compute_energy": float(energy_breakdown.get("uav_compute_energy", 0.0)),
            "ue_local_energy": float(energy_breakdown.get("ue_local_energy", 0.0)),
            "ue_uplink_energy": float(energy_breakdown.get("ue_uplink_energy", 0.0)),
            "bs_compute_energy": float(energy_breakdown.get("bs_compute_energy", 0.0)),
            "relay_fetch_energy": float(energy_breakdown.get("relay_fetch_energy", 0.0)),
        }
        return self.snapshot()

    def step_snapshot(self) -> dict[str, float | None]:
        return dict(self.last_step_metrics)

    def energy_breakdown_snapshot(self) -> dict[str, float]:
        return {
            "uav_move_energy": self.uav_move_energy,
            "uav_compute_energy": self.uav_compute_energy,
            "ue_local_energy": self.ue_local_energy,
            "ue_uplink_energy": self.ue_uplink_energy,
            "bs_compute_energy": self.bs_compute_energy,
            "relay_fetch_energy": self.relay_fetch_energy,
        }

    def snapshot(self) -> dict[str, float | None]:
        completion_rate = self.total_completed / self.total_generated if self.total_generated else 0.0
        average_latency = self.total_latency / self.total_generated if self.total_generated else 0.0
        cache_hit_rate = self.total_cache_hits / self.total_generated if self.total_generated else 0.0
        deadline_violation_rate = self.deadline_violations / self.total_generated if self.total_generated else 0.0
        reliability_violation_rate = self.reliability_violations / self.total_generated if self.total_generated else 0.0
        user_ratios = [
            self.user_completed.get(user_id, 0) / total if total else 0.0
            for user_id, total in self.user_generated.items()
        ]
        uav_load_values = [float(self.uav_loads.get(uav_id, 0)) for uav_id in range(self.num_uavs)]
        return {
            "completion_rate": completion_rate,
            "average_latency": average_latency,
            "total_energy": self.total_energy,
            "cache_hit_rate": cache_hit_rate,
            "fairness_user_completion": _jain(user_ratios),
            "fairness_uav_load": _jain(uav_load_values) if self.num_uavs > 1 else None,
            "deadline_violation_rate": deadline_violation_rate,
            "reliability_violation_rate": reliability_violation_rate,
            **self.energy_breakdown_snapshot(),
        }
