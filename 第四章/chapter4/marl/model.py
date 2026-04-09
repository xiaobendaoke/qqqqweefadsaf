from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


class SharedGaussianActor:
    def __init__(self, *, obs_dim: int, action_dim: int, rng: random.Random) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.weights = [[rng.uniform(-0.08, 0.08) for _ in range(action_dim)] for _ in range(obs_dim)]
        self.biases = [0.0 for _ in range(action_dim)]
        self.log_std = [math.log(0.35) for _ in range(action_dim)]

    def mean_action(self, observation: list[float]) -> list[float]:
        outputs: list[float] = []
        for action_idx in range(self.action_dim):
            linear = self.biases[action_idx]
            for obs_idx, obs_value in enumerate(observation):
                linear += obs_value * self.weights[obs_idx][action_idx]
            outputs.append(math.tanh(linear))
        return outputs

    def std_action(self, action_std_scale: float) -> list[float]:
        return [math.exp(log_std) * action_std_scale for log_std in self.log_std]

    def act(
        self,
        observations: list[list[float]],
        *,
        rng: random.Random,
        deterministic: bool,
        action_std_scale: float,
    ) -> tuple[list[list[float]], list[float]]:
        actions: list[list[float]] = []
        log_probs: list[float] = []
        for observation in observations:
            mean = self.mean_action(observation)
            stds = self.std_action(action_std_scale)
            if deterministic:
                action = mean[:]
            else:
                action = [
                    max(-1.0, min(1.0, mean[action_idx] + stds[action_idx] * rng.gauss(0.0, 1.0)))
                    for action_idx in range(self.action_dim)
                ]
            actions.append(action)
            log_probs.append(self.log_prob(observation, action, action_std_scale=action_std_scale))
        return actions, log_probs

    def log_prob(self, observation: list[float], action: list[float], *, action_std_scale: float) -> float:
        mean = self.mean_action(observation)
        stds = self.std_action(action_std_scale)
        total = 0.0
        for action_idx in range(self.action_dim):
            std = max(stds[action_idx], 1e-6)
            variance = std * std
            diff = action[action_idx] - mean[action_idx]
            total += -0.5 * ((diff * diff) / variance + math.log(2.0 * math.pi * variance))
        return total

    def update(
        self,
        *,
        observations: list[list[float]],
        actions: list[list[float]],
        advantages: list[float],
        learning_rate: float,
        entropy_coef: float,
        gradient_clip: float,
        action_std_scale: float,
    ) -> dict[str, float]:
        batch_size = max(1, len(observations))
        grad_weights = [[0.0 for _ in range(self.action_dim)] for _ in range(self.obs_dim)]
        grad_biases = [0.0 for _ in range(self.action_dim)]
        grad_log_std = [0.0 for _ in range(self.action_dim)]
        weighted_log_probs: list[float] = []
        entropies: list[float] = []

        for observation, action, advantage in zip(observations, actions, advantages):
            mean = self.mean_action(observation)
            stds = self.std_action(action_std_scale)
            weighted_log_probs.append(advantage * self.log_prob(observation, action, action_std_scale=action_std_scale))
            entropies.append(sum(math.log(max(std, 1e-6)) + 0.5 * math.log(2.0 * math.pi * math.e) for std in stds))
            for action_idx in range(self.action_dim):
                std = max(stds[action_idx], 1e-6)
                variance = std * std
                diff = action[action_idx] - mean[action_idx]
                dloss_dmean = -(advantage * diff / variance) / batch_size
                dloss_dlinear = dloss_dmean * (1.0 - mean[action_idx] * mean[action_idx])
                grad_biases[action_idx] += dloss_dlinear
                for obs_idx, obs_value in enumerate(observation):
                    grad_weights[obs_idx][action_idx] += obs_value * dloss_dlinear
                grad_log_std[action_idx] += -(advantage * (-1.0 + (diff * diff) / variance)) / batch_size

        for action_idx in range(self.action_dim):
            grad_log_std[action_idx] -= entropy_coef
            grad_biases[action_idx] = _clip(grad_biases[action_idx], gradient_clip)
            grad_log_std[action_idx] = _clip(grad_log_std[action_idx], gradient_clip)
            for obs_idx in range(self.obs_dim):
                grad_weights[obs_idx][action_idx] = _clip(grad_weights[obs_idx][action_idx], gradient_clip)
                self.weights[obs_idx][action_idx] -= learning_rate * grad_weights[obs_idx][action_idx]
            self.biases[action_idx] -= learning_rate * grad_biases[action_idx]
            self.log_std[action_idx] = max(
                math.log(0.05),
                min(math.log(1.0), self.log_std[action_idx] - learning_rate * grad_log_std[action_idx]),
            )

        return {
            "actor_loss": float(-_mean(weighted_log_probs) - entropy_coef * _mean(entropies)),
            "entropy": float(_mean(entropies)),
        }


class CentralValueCritic:
    def __init__(self, *, state_dim: int, rng: random.Random) -> None:
        self.state_dim = state_dim
        self.weights = [rng.uniform(-0.05, 0.05) for _ in range(state_dim)]
        self.bias = 0.0

    def value(self, state: list[float]) -> float:
        total = self.bias
        for index, value in enumerate(state):
            total += value * self.weights[index]
        return total

    def update(self, *, states: list[list[float]], returns: list[float], learning_rate: float, gradient_clip: float) -> dict[str, float]:
        batch_size = max(1, len(states))
        grad_weights = [0.0 for _ in range(self.state_dim)]
        grad_bias = 0.0
        errors: list[float] = []
        for state, target in zip(states, returns):
            value = self.value(state)
            error = value - target
            errors.append(error)
            grad_bias += error / batch_size
            for index, state_value in enumerate(state):
                grad_weights[index] += (error * state_value) / batch_size
        grad_bias = _clip(grad_bias, gradient_clip)
        for index in range(self.state_dim):
            grad_weights[index] = _clip(grad_weights[index], gradient_clip)
            self.weights[index] -= learning_rate * grad_weights[index]
        self.bias -= learning_rate * grad_bias
        return {"critic_loss": float(0.5 * _mean([error * error for error in errors]))}


class MinimalMultiAgentActorCritic:
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        seed: int,
        action_std_init: float,
        action_std_min: float,
        action_std_decay: float,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.state_dim = obs_dim * num_agents
        self.rng = random.Random(seed)
        self.actor = SharedGaussianActor(obs_dim=obs_dim, action_dim=action_dim, rng=self.rng)
        self.critic = CentralValueCritic(state_dim=self.state_dim, rng=self.rng)
        self.action_std_scale = float(action_std_init)
        self.action_std_min = float(action_std_min)
        self.action_std_decay = float(action_std_decay)

    def act(self, observations: list[list[float]], *, deterministic: bool) -> tuple[list[list[float]], list[float]]:
        return self.actor.act(
            observations,
            rng=self.rng,
            deterministic=deterministic,
            action_std_scale=self.action_std_scale,
        )

    def value(self, state: list[float]) -> float:
        return self.critic.value(state)

    def update(
        self,
        *,
        observations: list[list[float]],
        actions: list[list[float]],
        advantages: list[float],
        states: list[list[float]],
        returns: list[float],
        actor_lr: float,
        critic_lr: float,
        entropy_coef: float,
        gradient_clip: float,
    ) -> dict[str, float]:
        actor_stats = self.actor.update(
            observations=observations,
            actions=actions,
            advantages=advantages,
            learning_rate=actor_lr,
            entropy_coef=entropy_coef,
            gradient_clip=gradient_clip,
            action_std_scale=self.action_std_scale,
        )
        critic_stats = self.critic.update(
            states=states,
            returns=returns,
            learning_rate=critic_lr,
            gradient_clip=gradient_clip,
        )
        self.action_std_scale = max(self.action_std_min, self.action_std_scale * self.action_std_decay)
        return {
            **actor_stats,
            **critic_stats,
            "action_std_scale": float(self.action_std_scale),
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "num_agents": self.num_agents,
            "state_dim": self.state_dim,
            "action_std_scale": self.action_std_scale,
            "action_std_min": self.action_std_min,
            "action_std_decay": self.action_std_decay,
            "actor": {
                "weights": self.actor.weights,
                "biases": self.actor.biases,
                "log_std": self.actor.log_std,
            },
            "critic": {
                "weights": self.critic.weights,
                "bias": self.critic.bias,
            },
        }
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, *, seed: int) -> "MinimalMultiAgentActorCritic":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            obs_dim=int(payload["obs_dim"]),
            action_dim=int(payload["action_dim"]),
            num_agents=int(payload["num_agents"]),
            seed=seed,
            action_std_init=float(payload["action_std_scale"]),
            action_std_min=float(payload["action_std_min"]),
            action_std_decay=float(payload["action_std_decay"]),
        )
        model.actor.weights = payload["actor"]["weights"]
        model.actor.biases = payload["actor"]["biases"]
        model.actor.log_std = payload["actor"]["log_std"]
        model.critic.weights = payload["critic"]["weights"]
        model.critic.bias = float(payload["critic"]["bias"])
        model.action_std_scale = float(payload["action_std_scale"])
        return model

    def tensor_contract(self) -> dict[str, Any]:
        return {
            "observation_batch_shape": ["T", self.num_agents, self.obs_dim],
            "central_state_shape": ["T", self.state_dim],
            "action_batch_shape": ["T", self.num_agents, self.action_dim],
            "team_reward_shape": ["T"],
            "value_shape": ["T"],
            "advantage_shape": ["T"],
        }
