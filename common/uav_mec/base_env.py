from __future__ import annotations

import random
from typing import Any

from .config import SystemConfig
from .core.action import action_schema, normalize_actions
from .core.observation import build_observations
from .core.state import build_uav_state, observation_schema, uav_state_schema
from .entities import BaseStation, ServiceCatalog, UAVNode, UserEquipment
from .metrics import MetricTracker
from .scheduler import ComputeQueue, TDMAQueue
from .simulation.episode_log import build_per_uav_metrics, episode_log_schema
from .simulation.engine import run_step


class BaseEnv:
    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.service_catalog = ServiceCatalog.from_config(config)
        self.bs = BaseStation.from_config(config)
        self.users: list[UserEquipment] = []
        self.uavs: list[UAVNode] = []
        self.tdma_queue = TDMAQueue()
        self.compute_queue = ComputeQueue()
        self.metrics = MetricTracker(num_uavs=config.num_uavs)
        self.current_step = 0
        self.pending_tasks = []
        self.completed_tasks = []
        self.expired_tasks = []
        self.step_metrics_history: list[dict[str, Any]] = []
        self.energy_breakdown_history: list[dict[str, float]] = []
        self.queue_breakdown_history: list[dict[str, Any]] = []
        self.cache_event_history: list[dict[str, Any]] = []
        self.task_lifecycle_history: list[dict[str, int]] = []

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng = random.Random(seed)
        self.current_step = 0
        self.tdma_queue.reset()
        self.compute_queue.reset()
        self.metrics = MetricTracker(num_uavs=self.config.num_uavs)
        self.users = [UserEquipment.random_init(idx, self.config, self.rng) for idx in range(self.config.num_users)]
        self.uavs = [UAVNode.random_init(idx, self.config, self.rng) for idx in range(self.config.num_uavs)]
        self.pending_tasks = []
        self.completed_tasks = []
        self.expired_tasks = []
        self.step_metrics_history = []
        self.energy_breakdown_history = []
        self.queue_breakdown_history = []
        self.cache_event_history = []
        self.task_lifecycle_history = []
        observations = build_observations(
            uavs=self.uavs,
            users=self.users,
            pending_tasks=[],
            config=self.config,
            current_time=0.0,
        )
        return {
            "observations": observations,
            "info": {
                "time_step": self.current_step,
                "chapter_name": self.config.chapter_name,
                "agent_ids": self.get_agent_ids(),
            },
        }

    def step(self, actions: list[list[float]] | list[tuple[float, float]]) -> dict[str, Any]:
        normalized_actions = normalize_actions(actions, num_agents=self.config.num_uavs)
        execution = run_step(
            config=self.config,
            users=self.users,
            uavs=self.uavs,
            bs=self.bs,
            service_catalog=self.service_catalog,
            tdma_queue=self.tdma_queue,
            compute_queue=self.compute_queue,
            metrics=self.metrics,
            current_step=self.current_step,
            pending_tasks=self.pending_tasks,
            completed_tasks=self.completed_tasks,
            expired_tasks=self.expired_tasks,
            actions=normalized_actions,
            rng=self.rng,
        )
        self.current_step += 1
        observations = build_observations(
            uavs=self.uavs,
            users=self.users,
            pending_tasks=execution.pending_tasks,
            config=self.config,
            current_time=self.current_step * self.config.time_slot_duration,
        )
        metrics_snapshot = self.metrics.snapshot()
        step_metrics = self.metrics.step_snapshot()
        self.step_metrics_history.append({"step": self.current_step, **step_metrics})
        self.energy_breakdown_history.append({"step": self.current_step, **execution.energy_breakdown})
        self.queue_breakdown_history.append(
            {
                "step": self.current_step,
                "uavs": [
                    {
                        "uav_id": uav.uav_id,
                        "tx_queue_length": uav.current_tx_queue_length,
                        "tx_queue_delay": uav.current_tx_queue_delay,
                        "compute_queue_length": uav.current_compute_queue_length,
                        "compute_queue_delay": uav.current_compute_queue_delay,
                        "backlog_load": uav.current_backlog_load,
                        "coverage_load": uav.current_coverage_load,
                    }
                    for uav in self.uavs
                ],
            }
        )
        self.cache_event_history.extend({"step": self.current_step, **event} for event in execution.cache_events)
        self.task_lifecycle_history.append({"step": self.current_step, **execution.task_lifecycle_counts})
        terminated = self.current_step >= self.config.steps_per_episode
        return {
            "observations": observations,
            "rewards": [float(step_metrics["completion_rate"] - 0.05 * step_metrics["average_latency"]) for _ in self.uavs],
            "terminated": terminated,
            "truncated": False,
            "metrics": metrics_snapshot,
            "step_metrics": step_metrics,
            "info": {
                "time_step": self.current_step,
                "agent_ids": self.get_agent_ids(),
                "num_generated_tasks": execution.task_lifecycle_counts["new_tasks"],
                "num_completed_tasks": execution.task_lifecycle_counts["completed_tasks"],
                "num_expired_tasks": execution.task_lifecycle_counts["expired_tasks"],
                "num_pending_tasks": execution.task_lifecycle_counts["pending_tasks"],
                "num_cache_hits": sum(1 for task in execution.finalized_tasks if task.cache_hit),
                "num_deadline_violations": sum(1 for task in execution.finalized_tasks if task.total_latency > task.slack),
                "num_reliability_violations": sum(1 for task in execution.finalized_tasks if task.success_probability < task.required_reliability),
            },
        }

    def get_global_state(self) -> list[float]:
        state = []
        for uav in self.uavs:
            state.extend([float(uav.position[0]), float(uav.position[1]), float(uav.energy_ratio)])
        return state

    def get_num_agents(self) -> int:
        return self.config.num_uavs

    def get_agent_ids(self) -> list[str]:
        return [f"uav_{index}" for index in range(self.config.num_uavs)]

    def get_action_schema(self) -> dict[str, object]:
        return action_schema(num_agents=self.config.num_uavs, agent_ids=self.get_agent_ids(), config=self.config)

    def export_episode_summary(self) -> dict[str, Any]:
        return {"chapter_name": self.config.chapter_name, "config": self.config.to_dict(), "metrics": self.metrics.snapshot()}

    def get_uav_states(self) -> list[dict[str, Any]]:
        return [build_uav_state(uav=uav, all_uavs=self.uavs, config=self.config) for uav in self.uavs]

    def get_observation_schema(self) -> dict[str, object]:
        return observation_schema(self.config)

    def get_uav_state_schema(self) -> dict[str, object]:
        return uav_state_schema(self.config)

    def get_episode_log_schema(self) -> dict[str, object]:
        return episode_log_schema(self.config, self.get_agent_ids())

    def export_episode_log(self, *, episode_index: int, seed: int) -> dict[str, Any]:
        return {
            "chapter_name": self.config.chapter_name,
            "episode_index": episode_index,
            "seed": seed,
            "num_uavs": self.config.num_uavs,
            "num_users": self.config.num_users,
            "assignment_rule": self.config.assignment_rule,
            "global_metrics": self.metrics.snapshot(),
            "per_uav_metrics": build_per_uav_metrics(self.uavs),
            "action_schema": self.get_action_schema(),
            "observation_schema": self.get_observation_schema(),
            "uav_state_schema": self.get_uav_state_schema(),
            "step_metrics": self.step_metrics_history,
            "energy_breakdown": {
                "episode_totals": self.metrics.energy_breakdown_snapshot(),
                "step_breakdown": self.energy_breakdown_history,
            },
            "queue_breakdown": self.queue_breakdown_history,
            "cache_events": self.cache_event_history,
            "task_lifecycle_counts": self.task_lifecycle_history,
        }
