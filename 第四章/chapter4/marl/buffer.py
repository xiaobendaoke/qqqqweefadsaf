from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RolloutBatch:
    states: list[list[float]]
    observations: list[list[list[float]]]
    actions: list[list[list[float]]]
    log_probs: list[list[float]]
    rewards: list[float]
    dones: list[float]
    values: list[float]
    advantages: list[float]
    returns: list[float]

    @property
    def team_advantages(self) -> list[float]:
        return self.advantages

    @property
    def flat_observations(self) -> list[list[float]]:
        return [observation for step in self.observations for observation in step]

    @property
    def flat_actions(self) -> list[list[float]]:
        return [action for step in self.actions for action in step]

    @property
    def flat_advantages(self) -> list[float]:
        num_agents = len(self.observations[0]) if self.observations else 0
        return [advantage for advantage in self.advantages for _ in range(num_agents)]


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
        rewards = [float(item) for item in self.rewards]
        dones = [float(item) for item in self.dones]
        values = [float(item) for item in self.values]
        advantages = [0.0 for _ in rewards]
        gae = 0.0
        next_value = float(last_value)
        for index in range(len(rewards) - 1, -1, -1):
            mask = 1.0 - dones[index]
            delta = rewards[index] + gamma * next_value * mask - values[index]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[index] = gae
            next_value = values[index]
        returns = [advantage + value for advantage, value in zip(advantages, values)]
        if len(advantages) > 1:
            mean_adv = sum(advantages) / len(advantages)
            variance = sum((advantage - mean_adv) ** 2 for advantage in advantages) / len(advantages)
            std_adv = variance ** 0.5
            advantages = [(advantage - mean_adv) / (std_adv + 1e-8) for advantage in advantages]
        return RolloutBatch(
            states=self.states,
            observations=self.observations,
            actions=self.actions,
            log_probs=self.log_probs,
            rewards=rewards,
            dones=dones,
            values=values,
            advantages=advantages,
            returns=returns,
        )
