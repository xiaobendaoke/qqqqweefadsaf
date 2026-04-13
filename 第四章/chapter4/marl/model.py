from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .buffer import RolloutBatch
from .config import MinimalMARLConfig


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _cap_action_magnitude(values: list[float], *, limit: float = 1.0) -> list[float]:
    clipped = [_clip(value, limit) for value in values]
    norm = math.sqrt(sum(value * value for value in clipped))
    if norm <= limit or norm <= 1e-8:
        return clipped
    scale = limit / norm
    return [value * scale for value in clipped]


def _movement_budget(observation: list[float]) -> float:
    if len(observation) < 15:
        return 1.0
    start = len(observation) - 15
    rel_x = float(observation[start])
    rel_y = float(observation[start + 1])
    pending = max(0.0, float(observation[start + 2]))
    min_slack = min(1.0, max(0.0, float(observation[start + 3])))
    distance = min(1.0, math.sqrt(rel_x * rel_x + rel_y * rel_y))
    urgency = min(1.0, pending + (1.0 - min_slack))
    return min(1.0, max(distance, 0.10) + 0.35 * urgency)


def _layer_init(layer: nn.Linear, *, std: float = math.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class SharedActor(nn.Module):
    def __init__(self, *, obs_dim: int, action_dim: int, hidden_dim: int, action_std_init: float) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.mean_head = _layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(action_std_init)))

    def forward(self, observations: torch.Tensor) -> Normal:
        hidden = self.backbone(observations)
        mean = torch.tanh(self.mean_head(hidden))
        log_std = self.log_std.clamp(math.log(0.03), math.log(0.8))
        std = torch.exp(log_std).expand_as(mean)
        return Normal(mean, std)


class CentralCritic(nn.Module):
    def __init__(self, *, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            _layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states).squeeze(-1)


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
        use_movement_budget: bool = True,
        hidden_dim: int = 128,
        device: str = "cpu",
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.state_dim = obs_dim * num_agents
        self.action_std_min = float(action_std_min)
        self.action_std_decay = float(action_std_decay)
        self.use_movement_budget = bool(use_movement_budget)
        self.device = torch.device(device)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.actor = SharedActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            action_std_init=action_std_init,
        ).to(self.device)
        self.critic = CentralCritic(state_dim=self.state_dim, hidden_dim=hidden_dim).to(self.device)
        self.actor_optimizer: torch.optim.Optimizer | None = None
        self.critic_optimizer: torch.optim.Optimizer | None = None

    def configure_optimizers(self, *, actor_lr: float, critic_lr: float) -> None:
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def _apply_action_constraints(self, observations: list[list[float]], raw_actions: np.ndarray) -> list[list[float]]:
        constrained: list[list[float]] = []
        for observation, action in zip(observations, raw_actions.tolist()):
            limit = _movement_budget(observation) if self.use_movement_budget else 1.0
            constrained.append(_cap_action_magnitude(action, limit=limit))
        return constrained

    def act(self, observations: list[list[float]], *, deterministic: bool) -> tuple[list[list[float]], list[float]]:
        obs_tensor = torch.as_tensor(np.asarray(observations, dtype=np.float32), device=self.device)
        with torch.no_grad():
            dist = self.actor(obs_tensor)
            raw_actions = dist.mean if deterministic else dist.sample()
        actions = self._apply_action_constraints(observations, raw_actions.cpu().numpy())
        constrained_tensor = torch.as_tensor(np.asarray(actions, dtype=np.float32), device=self.device)
        with torch.no_grad():
            dist = self.actor(obs_tensor)
            log_probs = dist.log_prob(constrained_tensor).sum(dim=-1)
        return actions, [float(item) for item in log_probs.cpu().tolist()]

    def value(self, state: list[float]) -> float:
        state_tensor = torch.as_tensor(np.asarray([state], dtype=np.float32), device=self.device)
        with torch.no_grad():
            value = self.critic(state_tensor)
        return float(value.item())

    def update(self, *, batch: RolloutBatch, config: MinimalMARLConfig) -> dict[str, float]:
        if self.actor_optimizer is None or self.critic_optimizer is None:
            self.configure_optimizers(actor_lr=config.actor_lr, critic_lr=config.critic_lr)

        observations = torch.as_tensor(batch.flat_observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.flat_actions, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(batch.flat_log_probs, dtype=torch.float32, device=self.device)
        policy_advantages = torch.as_tensor(batch.flat_advantages, dtype=torch.float32, device=self.device)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(batch.values, dtype=torch.float32, device=self.device)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0

        for _ in range(config.ppo_epochs):
            dist = self.actor(observations)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
            ratio = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratio * policy_advantages
            clipped = torch.clamp(ratio, 1.0 - config.ppo_clip_eps, 1.0 + config.ppo_clip_eps) * policy_advantages
            actor_loss = -torch.min(unclipped, clipped).mean() - config.entropy_coef * entropy

            values = self.critic(states)
            value_delta = values - old_values
            clipped_values = old_values + value_delta.clamp(-config.value_clip_eps, config.value_clip_eps)
            critic_loss_unclipped = (values - returns).pow(2)
            critic_loss_clipped = (clipped_values - returns).pow(2)
            critic_loss = config.value_loss_coef * torch.max(critic_loss_unclipped, critic_loss_clipped).mean()

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), config.gradient_clip)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), config.gradient_clip)
            self.critic_optimizer.step()

            with torch.no_grad():
                total_actor_loss += float(actor_loss.item())
                total_critic_loss += float(critic_loss.item())
                total_entropy += float(entropy.item())

        current_std = float(torch.exp(self.actor.log_std).mean().item())
        target_std = max(self.action_std_min, current_std * self.action_std_decay)
        with torch.no_grad():
            self.actor.log_std.copy_(torch.log(torch.full_like(self.actor.log_std, target_std)))

        divisor = float(max(1, config.ppo_epochs))
        return {
            "actor_loss": total_actor_loss / divisor,
            "critic_loss": total_critic_loss / divisor,
            "entropy": total_entropy / divisor,
            "action_std_scale": target_std,
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "num_agents": self.num_agents,
            "state_dim": self.state_dim,
            "device": str(self.device),
            "actor_state": self.actor.state_dict(),
            "critic_state": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict() if self.actor_optimizer else None,
            "critic_optimizer": self.critic_optimizer.state_dict() if self.critic_optimizer else None,
            "action_std_min": self.action_std_min,
            "action_std_decay": self.action_std_decay,
            "use_movement_budget": self.use_movement_budget,
        }
        torch.save(payload, target)

    @classmethod
    def load(cls, path: str | Path, *, seed: int, device: str = "cpu") -> "MinimalMultiAgentActorCritic":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        log_std = payload["actor_state"]["log_std"]
        action_std_init = float(torch.exp(log_std).mean().item())
        model = cls(
            obs_dim=int(payload["obs_dim"]),
            action_dim=int(payload["action_dim"]),
            num_agents=int(payload["num_agents"]),
            seed=seed,
            action_std_init=action_std_init,
            action_std_min=float(payload["action_std_min"]),
            action_std_decay=float(payload["action_std_decay"]),
            use_movement_budget=bool(payload.get("use_movement_budget", True)),
            hidden_dim=int(payload["actor_state"]["backbone.0.weight"].shape[0]),
            device=device,
        )
        model.actor.load_state_dict(payload["actor_state"])
        model.critic.load_state_dict(payload["critic_state"])
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
