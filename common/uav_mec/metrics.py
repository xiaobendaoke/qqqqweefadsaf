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

    def record_tasks(self, tasks: list[Task], *, energy_spent: float) -> dict[str, float | None]:
        for task in tasks:
            self.total_generated += 1
            self.user_generated[task.user_id] = self.user_generated.get(task.user_id, 0) + 1
            self.total_latency += task.total_latency
            if task.cache_hit:
                self.total_cache_hits += 1
            if task.execution_target == "uav" and task.assigned_uav_id is not None:
                self.uav_loads[task.assigned_uav_id] = self.uav_loads.get(task.assigned_uav_id, 0) + 1
            if task.completed:
                self.total_completed += 1
                self.user_completed[task.user_id] = self.user_completed.get(task.user_id, 0) + 1
            else:
                if task.total_latency > task.slack:
                    self.deadline_violations += 1
                if task.success_probability < task.required_reliability:
                    self.reliability_violations += 1
        self.total_energy += energy_spent
        return self.snapshot()

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
        }
