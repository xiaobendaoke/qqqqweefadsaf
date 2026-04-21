"""MARL rollout 缓冲区模块。

该模块同时保留：

- legacy mobility-only PPO 轨迹缓存
- hybrid joint-action MAPPO/PPO 轨迹缓存

其中 hybrid buffer 会显式存储联合动作、分支 log-prob、action masks、
以及中心化 critic 所需的 joint observation summaries。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FloatArray = np.ndarray
IntArray = np.ndarray


@dataclass(slots=True)
class RolloutBatch:
    """legacy PPO 更新阶段使用的定型批数据。"""

    states: FloatArray
    observations: FloatArray
    actions: FloatArray
    log_probs: FloatArray
    rewards: FloatArray
    dones: FloatArray
    values: FloatArray
    advantages: FloatArray
    returns: FloatArray

    @property
    def team_advantages(self) -> FloatArray:
        return self.advantages

    @property
    def flat_observations(self) -> FloatArray:
        return self.observations.reshape(-1, self.observations.shape[-1])

    @property
    def flat_actions(self) -> FloatArray:
        return self.actions.reshape(-1, self.actions.shape[-1])

    @property
    def flat_log_probs(self) -> FloatArray:
        return self.log_probs.reshape(-1)

    @property
    def flat_advantages(self) -> FloatArray:
        num_agents = self.observations.shape[1] if self.observations.ndim >= 2 else 0
        return np.repeat(self.advantages, num_agents).astype(np.float32)


class RolloutBuffer:
    """legacy 轨迹缓存；供 mobility-only baseline 继续复用。"""

    def __init__(self) -> None:
        self.states: list[list[float]] = []
        self.observations: list[list[list[float]]] = []
        self.actions: list[list[list[float]]] = []
        self.log_probs: list[list[float]] = []
        self.rewards: list[float] = []
        self.dones: list[float] = []
        self.values: list[float] = []

    def add(
        self,
        *,
        state: list[float],
        observations: list[list[float]],
        actions: list[list[float]],
        log_probs: list[float],
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        self.states.append([float(item) for item in state])
        self.observations.append([[float(item) for item in row] for row in observations])
        self.actions.append([[float(item) for item in row] for row in actions])
        self.log_probs.append([float(item) for item in log_probs])
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

    def finalize(self, *, gamma: float, gae_lambda: float, last_value: float) -> RolloutBatch:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        next_value = float(last_value)
        for index in range(len(rewards) - 1, -1, -1):
            mask = 1.0 - float(dones[index])
            delta = float(rewards[index]) + gamma * next_value * mask - float(values[index])
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[index] = gae
            next_value = float(values[index])
        returns = advantages + values
        if len(advantages) > 1:
            mean_adv = float(np.mean(advantages))
            std_adv = float(np.std(advantages))
            advantages = (advantages - mean_adv) / (std_adv + 1e-8)
        return RolloutBatch(
            states=np.asarray(self.states, dtype=np.float32),
            observations=np.asarray(self.observations, dtype=np.float32),
            actions=np.asarray(self.actions, dtype=np.float32),
            log_probs=np.asarray(self.log_probs, dtype=np.float32),
            rewards=rewards,
            dones=dones,
            values=values,
            advantages=advantages.astype(np.float32),
            returns=returns.astype(np.float32),
        )


@dataclass(slots=True)
class HybridRolloutBatch:
    """hybrid MAPPO/PPO 更新阶段使用的联合动作批数据。"""

    critic_agent_summaries: FloatArray
    critic_global_summaries: FloatArray
    observations: FloatArray
    task_slot_features: FloatArray
    offload_candidate_features: FloatArray
    cache_candidate_features: FloatArray
    offload_candidate_ids: IntArray
    mobility_masks: FloatArray
    task_slot_masks: FloatArray
    offloading_candidate_masks: FloatArray
    offloading_defer_masks: FloatArray
    cache_service_masks: FloatArray
    mobility_actions: FloatArray
    offloading_option_indices: IntArray
    offloading_plan_ids: IntArray
    cache_actions: FloatArray
    mobility_log_probs: FloatArray
    offloading_log_probs: FloatArray
    cache_log_probs: FloatArray
    rewards: FloatArray
    dones: FloatArray
    values: FloatArray
    advantages: FloatArray
    returns: FloatArray

    @property
    def joint_log_probs(self) -> FloatArray:
        return self.mobility_log_probs + self.offloading_log_probs + self.cache_log_probs

    @property
    def team_advantages(self) -> FloatArray:
        return self.advantages

    @property
    def critic_inputs(self) -> FloatArray:
        if self.critic_global_summaries.shape[-1] <= 0:
            return self.critic_agent_summaries.reshape(self.critic_agent_summaries.shape[0], -1)
        return np.concatenate(
            [
                self.critic_agent_summaries.reshape(self.critic_agent_summaries.shape[0], -1),
                self.critic_global_summaries,
            ],
            axis=-1,
        ).astype(np.float32)


class HybridRolloutBuffer:
    """按时间顺序收集 joint-action 轨迹，并在 episode 结束后计算 GAE。"""

    def __init__(self) -> None:
        self.critic_agent_summaries: list[list[list[float]]] = []
        self.critic_global_summaries: list[list[float]] = []
        self.observations: list[list[list[float]]] = []
        self.task_slot_features: list[list[list[list[float]]]] = []
        self.offload_candidate_features: list[list[list[list[list[float]]]]] = []
        self.cache_candidate_features: list[list[list[list[float]]]] = []
        self.offload_candidate_ids: list[list[list[list[int]]]] = []
        self.mobility_masks: list[list[list[float]]] = []
        self.task_slot_masks: list[list[list[float]]] = []
        self.offloading_candidate_masks: list[list[list[list[float]]]] = []
        self.offloading_defer_masks: list[list[list[float]]] = []
        self.cache_service_masks: list[list[list[float]]] = []
        self.mobility_actions: list[list[list[float]]] = []
        self.offloading_option_indices: list[list[list[int]]] = []
        self.offloading_plan_ids: list[list[list[int]]] = []
        self.cache_actions: list[list[list[float]]] = []
        self.mobility_log_probs: list[list[float]] = []
        self.offloading_log_probs: list[list[float]] = []
        self.cache_log_probs: list[list[float]] = []
        self.rewards: list[float] = []
        self.dones: list[float] = []
        self.values: list[float] = []

    def add(
        self,
        *,
        critic_agent_summaries: list[list[float]],
        critic_global_summary: list[float],
        observations: list[list[float]],
        task_slot_features: list[list[list[float]]],
        offload_candidate_features: list[list[list[list[float]]]],
        cache_candidate_features: list[list[list[float]]],
        offload_candidate_ids: list[list[list[int]]],
        action_masks: dict[str, list],
        mobility_actions: list[list[float]],
        offloading_option_indices: list[list[int]],
        offloading_plan_ids: list[list[int]],
        cache_actions: list[list[float]],
        mobility_log_probs: list[float],
        offloading_log_probs: list[float],
        cache_log_probs: list[float],
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        self.critic_agent_summaries.append([[float(item) for item in row] for row in critic_agent_summaries])
        self.critic_global_summaries.append([float(item) for item in critic_global_summary])
        self.observations.append([[float(item) for item in row] for row in observations])
        self.task_slot_features.append(
            [[[float(item) for item in slot] for slot in agent_slots] for agent_slots in task_slot_features]
        )
        self.offload_candidate_features.append(
            [
                [
                    [[float(item) for item in candidate] for candidate in slot_candidates]
                    for slot_candidates in agent_candidates
                ]
                for agent_candidates in offload_candidate_features
            ]
        )
        self.cache_candidate_features.append(
            [[[float(item) for item in cache_item] for cache_item in agent_cache] for agent_cache in cache_candidate_features]
        )
        self.offload_candidate_ids.append(
            [
                [[int(item) for item in candidate_ids] for candidate_ids in agent_candidate_ids]
                for agent_candidate_ids in offload_candidate_ids
            ]
        )
        self.mobility_masks.append([[float(item) for item in row] for row in action_masks["mobility_mask"]])
        self.task_slot_masks.append([[float(item) for item in row] for row in action_masks["task_slot_mask"]])
        self.offloading_candidate_masks.append(
            [
                [[float(item) for item in row] for row in candidate_mask_rows]
                for candidate_mask_rows in action_masks["offloading_candidate_mask"]
            ]
        )
        self.offloading_defer_masks.append(
            [[float(item) for item in row] for row in action_masks["offloading_defer_mask"]]
        )
        self.cache_service_masks.append([[float(item) for item in row] for row in action_masks["cache_service_mask"]])
        self.mobility_actions.append([[float(item) for item in row] for row in mobility_actions])
        self.offloading_option_indices.append([[int(item) for item in row] for row in offloading_option_indices])
        self.offloading_plan_ids.append([[int(item) for item in row] for row in offloading_plan_ids])
        self.cache_actions.append([[float(item) for item in row] for row in cache_actions])
        self.mobility_log_probs.append([float(item) for item in mobility_log_probs])
        self.offloading_log_probs.append([float(item) for item in offloading_log_probs])
        self.cache_log_probs.append([float(item) for item in cache_log_probs])
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

    def finalize(self, *, gamma: float, gae_lambda: float, last_value: float) -> HybridRolloutBatch:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        next_value = float(last_value)
        for index in range(len(rewards) - 1, -1, -1):
            mask = 1.0 - float(dones[index])
            delta = float(rewards[index]) + gamma * next_value * mask - float(values[index])
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[index] = gae
            next_value = float(values[index])
        returns = advantages + values
        if len(advantages) > 1:
            mean_adv = float(np.mean(advantages))
            std_adv = float(np.std(advantages))
            advantages = (advantages - mean_adv) / (std_adv + 1e-8)

        return HybridRolloutBatch(
            critic_agent_summaries=np.asarray(self.critic_agent_summaries, dtype=np.float32),
            critic_global_summaries=np.asarray(self.critic_global_summaries, dtype=np.float32),
            observations=np.asarray(self.observations, dtype=np.float32),
            task_slot_features=np.asarray(self.task_slot_features, dtype=np.float32),
            offload_candidate_features=np.asarray(self.offload_candidate_features, dtype=np.float32),
            cache_candidate_features=np.asarray(self.cache_candidate_features, dtype=np.float32),
            offload_candidate_ids=np.asarray(self.offload_candidate_ids, dtype=np.int64),
            mobility_masks=np.asarray(self.mobility_masks, dtype=np.float32),
            task_slot_masks=np.asarray(self.task_slot_masks, dtype=np.float32),
            offloading_candidate_masks=np.asarray(self.offloading_candidate_masks, dtype=np.float32),
            offloading_defer_masks=np.asarray(self.offloading_defer_masks, dtype=np.float32),
            cache_service_masks=np.asarray(self.cache_service_masks, dtype=np.float32),
            mobility_actions=np.asarray(self.mobility_actions, dtype=np.float32),
            offloading_option_indices=np.asarray(self.offloading_option_indices, dtype=np.int64),
            offloading_plan_ids=np.asarray(self.offloading_plan_ids, dtype=np.int64),
            cache_actions=np.asarray(self.cache_actions, dtype=np.float32),
            mobility_log_probs=np.asarray(self.mobility_log_probs, dtype=np.float32),
            offloading_log_probs=np.asarray(self.offloading_log_probs, dtype=np.float32),
            cache_log_probs=np.asarray(self.cache_log_probs, dtype=np.float32),
            rewards=rewards,
            dones=dones,
            values=values,
            advantages=advantages.astype(np.float32),
            returns=returns.astype(np.float32),
        )
