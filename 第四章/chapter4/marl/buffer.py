from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FloatArray = np.ndarray


@dataclass(slots=True)
class RolloutBatch:
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
