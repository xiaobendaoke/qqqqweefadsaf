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


LOG_STD_MIN = math.log(0.03)
LOG_STD_MAX = math.log(0.8)
EPS = 1.0e-6


def _layer_init(layer: nn.Linear, *, std: float = math.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _movement_budget(observation: list[float], *, max_user_blocks: int, user_feature_dim: int) -> float:
    block_width = max_user_blocks * user_feature_dim
    if block_width <= 0 or len(observation) < block_width:
        return 1.0
    user_values = observation[-block_width:]
    max_priority = 0.0
    max_distance = 0.0
    for offset in range(0, len(user_values), user_feature_dim):
        block = user_values[offset : offset + user_feature_dim]
        if len(block) < user_feature_dim:
            continue
        rel_x = float(block[0])
        rel_y = float(block[1])
        pending = max(0.0, float(block[2]))
        min_slack = min(1.0, max(0.0, float(block[3])))
        if pending <= 0.0:
            continue
        distance = min(1.0, math.hypot(rel_x, rel_y))
        priority = pending + (1.0 - min_slack)
        max_priority = max(max_priority, priority)
        max_distance = max(max_distance, distance)
    if max_priority <= 0.0:
        return 0.35
    return min(1.0, max(0.18, 0.35 + 0.30 * min(1.0, max_priority) + 0.35 * max_distance))


def _budget_tensor(observations: torch.Tensor, *, max_user_blocks: int, user_feature_dim: int) -> torch.Tensor:
    budgets = [
        _movement_budget(observation.tolist(), max_user_blocks=max_user_blocks, user_feature_dim=user_feature_dim)
        for observation in observations
    ]
    return torch.as_tensor(budgets, dtype=torch.float32, device=observations.device).unsqueeze(-1)


def _atanh(value: torch.Tensor) -> torch.Tensor:
    clamped = value.clamp(-1.0 + EPS, 1.0 - EPS)
    return 0.5 * (torch.log1p(clamped) - torch.log1p(-clamped))


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
        mean = self.mean_head(hidden)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
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
        max_user_blocks: int = 3,
        user_feature_dim: int = 5,
        device: str = "cpu",
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.state_dim = obs_dim * num_agents
        self.action_std_min = float(action_std_min)
        self.action_std_decay = float(action_std_decay)
        self.use_movement_budget = bool(use_movement_budget)
        self.max_user_blocks = int(max_user_blocks)
        self.user_feature_dim = int(user_feature_dim)
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

    def _action_and_log_prob(self, observations: torch.Tensor, *, raw_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor(observations)
        budgets = _budget_tensor(
            observations,
            max_user_blocks=self.max_user_blocks,
            user_feature_dim=self.user_feature_dim,
        ) if self.use_movement_budget else torch.ones((observations.shape[0], 1), dtype=torch.float32, device=observations.device)
        squashed = torch.tanh(raw_actions)
        actions = squashed * budgets
        log_prob = dist.log_prob(raw_actions).sum(dim=-1)
        log_prob -= torch.log(1.0 - squashed.pow(2) + EPS).sum(dim=-1)
        log_prob -= self.action_dim * torch.log(budgets.squeeze(-1) + EPS)
        return actions, log_prob

    def _raw_actions_from_constrained(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        budgets = _budget_tensor(
            observations,
            max_user_blocks=self.max_user_blocks,
            user_feature_dim=self.user_feature_dim,
        ) if self.use_movement_budget else torch.ones((observations.shape[0], 1), dtype=torch.float32, device=observations.device)
        normalized = (actions / (budgets + EPS)).clamp(-1.0 + EPS, 1.0 - EPS)
        raw_actions = _atanh(normalized)
        return raw_actions, budgets

    def act(self, observations: list[list[float]], *, deterministic: bool) -> tuple[list[list[float]], list[float]]:
        obs_tensor = torch.as_tensor(np.asarray(observations, dtype=np.float32), device=self.device)
        with torch.no_grad():
            dist = self.actor(obs_tensor)
            raw_actions = dist.mean if deterministic else dist.rsample()
            actions, log_probs = self._action_and_log_prob(obs_tensor, raw_actions=raw_actions)
        return actions.cpu().numpy().tolist(), [float(item) for item in log_probs.cpu().tolist()]

    def value(self, state: list[float]) -> float:
        state_tensor = torch.as_tensor(np.asarray([state], dtype=np.float32), device=self.device)
        with torch.no_grad():
            value = self.critic(state_tensor)
        return float(value.item())

    def _iter_minibatches(self, total_steps: int, minibatch_size: int) -> list[np.ndarray]:
        indices = np.arange(total_steps, dtype=np.int64)
        np.random.shuffle(indices)
        return [indices[start : start + minibatch_size] for start in range(0, total_steps, minibatch_size)]

    def update(self, *, batch: RolloutBatch, config: MinimalMARLConfig) -> dict[str, float]:
        if self.actor_optimizer is None or self.critic_optimizer is None:
            self.configure_optimizers(actor_lr=config.actor_lr, critic_lr=config.critic_lr)

        observations = torch.as_tensor(batch.observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(batch.log_probs, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.device)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(batch.values, dtype=torch.float32, device=self.device)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        update_steps = 0
        minibatch_size = max(1, min(config.minibatch_size, observations.shape[0]))

        for _ in range(config.ppo_epochs):
            for step_indices in self._iter_minibatches(observations.shape[0], minibatch_size):
                obs_mb = observations[step_indices].reshape(-1, self.obs_dim)
                act_mb = actions[step_indices].reshape(-1, self.action_dim)
                old_log_prob_mb = old_log_probs[step_indices].reshape(-1)
                adv_mb = advantages[step_indices].repeat_interleave(self.num_agents)
                states_mb = states[step_indices]
                returns_mb = returns[step_indices]
                old_values_mb = old_values[step_indices]

                raw_actions_mb, budgets_mb = self._raw_actions_from_constrained(obs_mb, act_mb)
                dist = self.actor(obs_mb)
                squashed = torch.tanh(raw_actions_mb)
                new_log_probs = dist.log_prob(raw_actions_mb).sum(dim=-1)
                new_log_probs -= torch.log(1.0 - squashed.pow(2) + EPS).sum(dim=-1)
                new_log_probs -= self.action_dim * torch.log(budgets_mb.squeeze(-1) + EPS)
                entropy = dist.entropy().sum(dim=-1).mean()
                ratio = torch.exp(new_log_probs - old_log_prob_mb)
                unclipped = ratio * adv_mb
                clipped = torch.clamp(ratio, 1.0 - config.ppo_clip_eps, 1.0 + config.ppo_clip_eps) * adv_mb
                actor_loss = -torch.min(unclipped, clipped).mean() - config.entropy_coef * entropy

                values = self.critic(states_mb)
                value_delta = values - old_values_mb
                clipped_values = old_values_mb + value_delta.clamp(-config.value_clip_eps, config.value_clip_eps)
                critic_loss_unclipped = (values - returns_mb).pow(2)
                critic_loss_clipped = (clipped_values - returns_mb).pow(2)
                critic_loss = config.value_loss_coef * torch.max(critic_loss_unclipped, critic_loss_clipped).mean()

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), config.gradient_clip)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), config.gradient_clip)
                self.critic_optimizer.step()

                total_actor_loss += float(actor_loss.item())
                total_critic_loss += float(critic_loss.item())
                total_entropy += float(entropy.item())
                update_steps += 1

        current_std = float(torch.exp(self.actor.log_std).mean().item())
        target_std = max(self.action_std_min, current_std * self.action_std_decay)
        with torch.no_grad():
            self.actor.log_std.copy_(torch.log(torch.full_like(self.actor.log_std, target_std)))

        divisor = float(max(1, update_steps))
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
            "max_user_blocks": self.max_user_blocks,
            "user_feature_dim": self.user_feature_dim,
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
            max_user_blocks=int(payload.get("max_user_blocks", 3)),
            user_feature_dim=int(payload.get("user_feature_dim", 5)),
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
