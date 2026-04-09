from __future__ import annotations

import random
from typing import Any

from .config import SystemConfig
from .core.action import action_schema, normalize_actions
from .core.observation import build_observations
from .core.state import build_uav_state, observation_schema, uav_state_schema
from .entities import BaseStation, ServiceCatalog, UAVNode, UserEquipment
from .metrics import MetricTracker
from .scheduler.tdma import TDMAQueue
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
        self.metrics = MetricTracker(num_uavs=config.num_uavs)
        self.current_step = 0

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng = random.Random(seed)
        self.current_step = 0
        self.tdma_queue.reset()
        self.metrics = MetricTracker(num_uavs=self.config.num_uavs)
        self.users = [UserEquipment.random_init(idx, self.config, self.rng) for idx in range(self.config.num_users)]
        self.uavs = [UAVNode.random_init(idx, self.config, self.rng) for idx in range(self.config.num_uavs)]
        observations = build_observations(uavs=self.uavs, users=self.users, pending_tasks=[], config=self.config)
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
            metrics=self.metrics,
            current_step=self.current_step,
            actions=normalized_actions,
            rng=self.rng,
        )
        self.current_step += 1
        observations = build_observations(
            uavs=self.uavs,
            users=self.users,
            pending_tasks=execution.generated_tasks,
            config=self.config,
        )
        metrics_snapshot = self.metrics.snapshot()
        terminated = self.current_step >= self.config.steps_per_episode
        return {
            "observations": observations,
            "rewards": [float(metrics_snapshot["completion_rate"] - 0.05 * metrics_snapshot["average_latency"]) for _ in self.uavs],
            "terminated": terminated,
            "truncated": False,
            "metrics": metrics_snapshot,
            "info": {
                "time_step": self.current_step,
                "agent_ids": self.get_agent_ids(),
                "num_generated_tasks": len(execution.generated_tasks),
                "num_completed_tasks": sum(1 for task in execution.generated_tasks if task.completed),
                "num_cache_hits": sum(1 for task in execution.generated_tasks if task.cache_hit),
                "num_deadline_violations": sum(1 for task in execution.generated_tasks if task.total_latency > task.slack),
                "num_reliability_violations": sum(
                    1 for task in execution.generated_tasks if task.success_probability < task.required_reliability
                ),
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
        }
