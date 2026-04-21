"""第四章共享策略与中心化 critic 模型模块。

该模块在保留 legacy mobility-only PPO 接口的同时，
为后续联合优化升级提供 joint actor / centralized critic 前向接口：

- mobility head: 连续高斯头
- offloading head: 对 K x C 候选做 masked categorical 选择
- cache head: 输出服务缓存 priority 分数
- critic: 支持读取 joint observation summaries

兼容性约束：
- 现有 trainer 继续走 `act()/value()/update()` 的 mobility-only 路径
- 新的 `forward_joint()` 只暴露模型前向契约，不要求本步同步改 trainer
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.distributions import Normal

from .buffer import RolloutBatch
from .config import MinimalMARLConfig
from .device import configure_torch_runtime
from .device import describe_runtime_device
from .device import normalize_device_request
from .device import resolve_device


LOG_STD_MIN = math.log(0.03)
LOG_STD_MAX = math.log(0.8)
CACHE_LOG_STD_MIN = math.log(0.05)
CACHE_LOG_STD_MAX = math.log(1.2)
MASKED_LOGIT = -1.0e9
DEFAULT_DEFER_PLAN_ID = -1
EPS = 1.0e-6


def _layer_init(layer: nn.Linear, *, std: float = math.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    """使用 PPO 常见的正交初始化稳定训练初期尺度。"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _movement_budget(observation: list[float], *, max_user_blocks: int, user_feature_dim: int) -> float:
    """依据观测中的 backlog 强度动态收缩或放宽动作幅度预算。"""
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
        return 0.20
    return min(1.0, max(0.18, 0.35 + 0.30 * min(1.0, max_priority) + 0.35 * max_distance))


def _budget_tensor(observations: torch.Tensor, *, max_user_blocks: int, user_feature_dim: int) -> torch.Tensor:
    """为每条观测批量生成移动预算张量。"""
    budgets = [
        _movement_budget(observation.tolist(), max_user_blocks=max_user_blocks, user_feature_dim=user_feature_dim)
        for observation in observations
    ]
    return torch.as_tensor(budgets, dtype=torch.float32, device=observations.device).unsqueeze(-1)


def _atanh(value: torch.Tensor) -> torch.Tensor:
    """对约束动作做稳定的反双曲正切变换。"""
    clamped = value.clamp(-1.0 + EPS, 1.0 - EPS)
    return 0.5 * (torch.log1p(clamped) - torch.log1p(-clamped))


def _to_tensor(value: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """把列表 / numpy / tensor 统一转为目标 device 上的 tensor。"""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)


def _with_optional_batch(tensor: torch.Tensor, *, expected_rank: int, name: str) -> tuple[torch.Tensor, bool]:
    """允许输入省略 batch 维；省略时自动补成 batch=1。"""
    squeeze_batch = False
    if tensor.dim() == expected_rank - 1:
        tensor = tensor.unsqueeze(0)
        squeeze_batch = True
    if tensor.dim() != expected_rank:
        raise ValueError(f"{name} must have rank {expected_rank - 1} or {expected_rank}, got shape {tuple(tensor.shape)}")
    return tensor, squeeze_batch


def _broadcast_tensor(
    value: Any,
    *,
    target_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """允许 mask / id 张量以共享前缀的形式广播到 joint batch。"""
    tensor = _to_tensor(value, dtype=dtype, device=device)
    while tensor.dim() < len(target_shape):
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != len(target_shape):
        raise ValueError(f"{name} must be broadcastable to shape {target_shape}, got {tuple(tensor.shape)}")
    expand_shape: list[int] = []
    for current, target in zip(tensor.shape, target_shape):
        if current == target:
            expand_shape.append(target)
            continue
        if current == 1:
            expand_shape.append(target)
            continue
        raise ValueError(f"{name} must be broadcastable to shape {target_shape}, got {tuple(tensor.shape)}")
    return tensor.expand(*expand_shape)


def _masked_logits(logits: torch.Tensor, mask: torch.Tensor, *, fallback_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """对 categorical logits 施加 mask；若一整行都不可选，则强制 fallback 可选。"""
    if logits.shape != mask.shape:
        raise ValueError(f"logits/mask shape mismatch: {tuple(logits.shape)} vs {tuple(mask.shape)}")
    fallback_index = int(fallback_index) % logits.shape[-1]
    mask_bool = mask > 0.5
    fallback_mask = torch.zeros_like(mask_bool)
    fallback_mask[..., fallback_index] = True
    valid_mask = torch.where(mask_bool.any(dim=-1, keepdim=True), mask_bool, fallback_mask)
    return logits.masked_fill(~valid_mask, MASKED_LOGIT), valid_mask.to(dtype=logits.dtype)


def _reshape_or_squeeze(tensor: torch.Tensor, *, batch_size: int, num_agents: int, squeeze_batch: bool) -> torch.Tensor:
    reshaped = tensor.reshape(batch_size, num_agents, *tensor.shape[1:])
    return reshaped.squeeze(0) if squeeze_batch else reshaped


class SharedActor(nn.Module):
    """共享 actor 骨干；legacy 只用 mobility，高阶 joint 接口复用同一骨干。"""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        action_std_init: float,
        cache_action_std_init: float,
        task_slot_dim: int,
        offloading_candidate_dim: int,
        cache_candidate_dim: int,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.task_slot_dim = int(task_slot_dim)
        self.offloading_candidate_dim = int(offloading_candidate_dim)
        self.cache_candidate_dim = int(cache_candidate_dim)
        self.hidden_dim = int(hidden_dim)

        self.backbone = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.mean_head = _layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(action_std_init)))
        self.cache_log_std = nn.Parameter(torch.tensor(math.log(cache_action_std_init), dtype=torch.float32))

        self.task_slot_encoder = nn.Sequential(
            _layer_init(nn.Linear(hidden_dim + task_slot_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.offloading_candidate_head = nn.Sequential(
            _layer_init(nn.Linear(hidden_dim + offloading_candidate_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, 1), std=0.01),
        )
        self.offloading_defer_head = _layer_init(nn.Linear(hidden_dim, 1), std=0.01)
        self.cache_priority_head = nn.Sequential(
            _layer_init(nn.Linear(hidden_dim + cache_candidate_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, 1), std=0.01),
        )

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        return self.backbone(observations)

    def _mobility_distribution(self, hidden: torch.Tensor) -> Normal:
        mean = self.mean_head(hidden)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mean)
        return Normal(mean, std)

    def _cache_distribution(self, cache_means: torch.Tensor) -> Normal:
        log_std = self.cache_log_std.clamp(CACHE_LOG_STD_MIN, CACHE_LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(cache_means)
        return Normal(cache_means, std)

    def forward(self, observations: torch.Tensor) -> Normal:
        """legacy mobility-only 前向：返回连续高斯分布。"""
        return self._mobility_distribution(self.encode(observations))

    def forward_joint_heads(
        self,
        *,
        flat_observations: torch.Tensor,
        task_slot_features: torch.Tensor,
        offload_candidate_features: torch.Tensor,
        cache_candidate_features: torch.Tensor,
    ) -> dict[str, torch.Tensor | Normal]:
        """joint actor 前向：返回 mobility 分布与 offloading/cache 原始 head 输出。"""
        if flat_observations.dim() != 2:
            raise ValueError(f"flat_observations must have shape [B*N, obs_dim], got {tuple(flat_observations.shape)}")
        if task_slot_features.dim() != 3:
            raise ValueError(f"task_slot_features must have shape [B*N, K, D_task], got {tuple(task_slot_features.shape)}")
        if offload_candidate_features.dim() != 4:
            raise ValueError(
                "offload_candidate_features must have shape [B*N, K, C, D_candidate], "
                f"got {tuple(offload_candidate_features.shape)}"
            )
        if cache_candidate_features.dim() != 3:
            raise ValueError(
                f"cache_candidate_features must have shape [B*N, S, D_cache], got {tuple(cache_candidate_features.shape)}"
            )

        hidden = self.encode(flat_observations)
        mobility_dist = self._mobility_distribution(hidden)

        task_slot_count = task_slot_features.shape[1]
        candidate_count = offload_candidate_features.shape[2]
        cache_service_count = cache_candidate_features.shape[1]

        repeated_hidden = hidden.unsqueeze(1).expand(-1, task_slot_count, -1)
        slot_hidden = self.task_slot_encoder(torch.cat([repeated_hidden, task_slot_features], dim=-1))

        repeated_slot_hidden = slot_hidden.unsqueeze(2).expand(-1, -1, candidate_count, -1)
        offloading_candidate_logits = self.offloading_candidate_head(
            torch.cat([repeated_slot_hidden, offload_candidate_features], dim=-1)
        ).squeeze(-1)
        offloading_defer_logits = self.offloading_defer_head(slot_hidden).squeeze(-1)

        repeated_cache_hidden = hidden.unsqueeze(1).expand(-1, cache_service_count, -1)
        cache_mean_scores = self.cache_priority_head(
            torch.cat([repeated_cache_hidden, cache_candidate_features], dim=-1)
        ).squeeze(-1)
        cache_dist = self._cache_distribution(cache_mean_scores)

        return {
            "context": hidden,
            "mobility_dist": mobility_dist,
            "offloading_candidate_logits": offloading_candidate_logits,
            "offloading_defer_logits": offloading_defer_logits,
            "cache_mean_scores": cache_mean_scores,
            "cache_dist": cache_dist,
        }


class CentralCritic(nn.Module):
    """读取中心化 critic 输入向量，输出 team value。"""

    def __init__(self, *, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.network = nn.Sequential(
            _layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, critic_inputs: torch.Tensor) -> torch.Tensor:
        return self.network(critic_inputs).squeeze(-1)


class MinimalMultiAgentActorCritic:
    """最小可运行的 shared-actor centralized-critic PPO/MAPPO 风格实现。"""

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
        cache_action_std_init: float = 0.35,
        cache_action_std_min: float = 0.10,
        cache_action_std_decay: float = 0.995,
        task_slot_dim: int = 11,
        offloading_candidate_dim: int = 12,
        cache_candidate_dim: int = 9,
        joint_agent_summary_dim: int | None = None,
        joint_global_summary_dim: int = 0,
        default_defer_plan_id: int = DEFAULT_DEFER_PLAN_ID,
        device: str = "auto",
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.num_agents = int(num_agents)
        self.state_dim = self.obs_dim * self.num_agents
        self.action_std_min = float(action_std_min)
        self.action_std_decay = float(action_std_decay)
        self.cache_action_std_min = float(cache_action_std_min)
        self.cache_action_std_decay = float(cache_action_std_decay)
        self.use_movement_budget = bool(use_movement_budget)
        self.max_user_blocks = int(max_user_blocks)
        self.user_feature_dim = int(user_feature_dim)
        self.task_slot_dim = int(task_slot_dim)
        self.offloading_candidate_dim = int(offloading_candidate_dim)
        self.cache_candidate_dim = int(cache_candidate_dim)
        self.joint_agent_summary_dim = int(joint_agent_summary_dim if joint_agent_summary_dim is not None else obs_dim)
        self.joint_global_summary_dim = int(joint_global_summary_dim)
        self.critic_input_dim = self.joint_agent_summary_dim * self.num_agents + self.joint_global_summary_dim
        self.default_defer_plan_id = int(default_defer_plan_id)
        self.requested_device = normalize_device_request(device)
        self.resolved_device = resolve_device(self.requested_device)
        configure_torch_runtime(self.resolved_device)
        self.device = torch.device(self.resolved_device)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        self.actor = SharedActor(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            action_std_init=action_std_init,
            cache_action_std_init=cache_action_std_init,
            task_slot_dim=self.task_slot_dim,
            offloading_candidate_dim=self.offloading_candidate_dim,
            cache_candidate_dim=self.cache_candidate_dim,
        ).to(self.device)
        self.critic = CentralCritic(input_dim=self.critic_input_dim, hidden_dim=hidden_dim).to(self.device)
        self.actor_optimizer: torch.optim.Optimizer | None = None
        self.critic_optimizer: torch.optim.Optimizer | None = None

    def configure_optimizers(self, *, actor_lr: float, critic_lr: float) -> None:
        """为 actor 和 critic 分别配置 Adam 优化器。"""
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def _continuous_action_statistics(
        self,
        *,
        raw_actions: torch.Tensor,
        dist: Normal,
        action_mask: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if action_mask is None:
            action_mask = torch.ones_like(raw_actions)
        else:
            action_mask = action_mask.to(device=raw_actions.device, dtype=torch.float32)
        if scale is None:
            scale_tensor = torch.ones_like(raw_actions)
        else:
            scale_tensor = scale.to(device=raw_actions.device, dtype=torch.float32)
        squashed = torch.tanh(raw_actions)
        actions = squashed * scale_tensor * action_mask
        mean_actions = torch.tanh(dist.mean) * scale_tensor * action_mask
        log_prob_terms = dist.log_prob(raw_actions) - torch.log(1.0 - squashed.pow(2) + EPS)
        log_prob = (log_prob_terms * action_mask).sum(dim=-1)
        if scale is not None:
            log_prob -= (torch.log(scale_tensor + EPS) * action_mask).sum(dim=-1)
        entropy = (dist.entropy() * action_mask).sum(dim=-1)
        return {
            "actions": actions,
            "mean_actions": mean_actions,
            "raw_actions": raw_actions,
            "log_prob": log_prob,
            "entropy": entropy,
        }

    def _mobility_action_statistics(
        self,
        observations: torch.Tensor,
        *,
        raw_actions: torch.Tensor,
        dist: Normal,
        mobility_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        budgets = (
            _budget_tensor(
                observations,
                max_user_blocks=self.max_user_blocks,
                user_feature_dim=self.user_feature_dim,
            )
            if self.use_movement_budget
            else torch.ones((observations.shape[0], 1), dtype=torch.float32, device=observations.device)
        )
        outputs = self._continuous_action_statistics(
            raw_actions=raw_actions,
            dist=dist,
            action_mask=mobility_mask,
            scale=budgets,
        )
        outputs["budgets"] = budgets.squeeze(-1)
        return outputs

    def _cache_action_statistics(
        self,
        *,
        raw_actions: torch.Tensor,
        dist: Normal,
        cache_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self._continuous_action_statistics(
            raw_actions=raw_actions,
            dist=dist,
            action_mask=cache_mask,
            scale=None,
        )

    def _action_and_log_prob(self, observations: torch.Tensor, *, raw_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """将高斯原始动作压缩到合法范围，并计算修正后的对数概率。"""
        dist = self.actor(observations)
        outputs = self._mobility_action_statistics(observations, raw_actions=raw_actions, dist=dist)
        return outputs["actions"], outputs["log_prob"]

    def _raw_actions_from_constrained(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """把已受预算约束的动作还原到高斯采样空间，用于 PPO 重算 log-prob。"""
        budgets = (
            _budget_tensor(
                observations,
                max_user_blocks=self.max_user_blocks,
                user_feature_dim=self.user_feature_dim,
            )
            if self.use_movement_budget
            else torch.ones((observations.shape[0], 1), dtype=torch.float32, device=observations.device)
        )
        normalized = (actions / (budgets + EPS)).clamp(-1.0 + EPS, 1.0 - EPS)
        raw_actions = _atanh(normalized)
        return raw_actions, budgets

    def _raw_cache_actions_from_constrained(self, cache_actions: torch.Tensor) -> torch.Tensor:
        normalized = cache_actions.clamp(-1.0 + EPS, 1.0 - EPS)
        return _atanh(normalized)

    def decay_exploration_stds(self) -> dict[str, float]:
        mobility_current = float(torch.exp(self.actor.log_std).mean().item())
        mobility_target = max(self.action_std_min, mobility_current * self.action_std_decay)
        cache_current = float(torch.exp(self.actor.cache_log_std).mean().item())
        cache_target = max(self.cache_action_std_min, cache_current * self.cache_action_std_decay)
        with torch.no_grad():
            self.actor.log_std.copy_(torch.log(torch.full_like(self.actor.log_std, mobility_target)))
            self.actor.cache_log_std.copy_(torch.log(torch.full_like(self.actor.cache_log_std, cache_target)))
        return {
            "action_std_scale": mobility_target,
            "cache_action_std_scale": cache_target,
        }

    def forward_legacy(self, observations: list[list[float]] | np.ndarray | torch.Tensor, *, deterministic: bool) -> dict[str, torch.Tensor]:
        """legacy mobility-only 前向接口；供当前 trainer / smoke test 继续复用。"""
        obs_tensor = _to_tensor(observations, dtype=torch.float32, device=self.device)
        obs_tensor, squeeze_batch = _with_optional_batch(obs_tensor, expected_rank=3, name="observations")
        if obs_tensor.shape[1] != self.num_agents or obs_tensor.shape[2] != self.obs_dim:
            raise ValueError(
                f"observations must have shape [N, {self.obs_dim}] or [B, {self.num_agents}, {self.obs_dim}], "
                f"got {tuple(obs_tensor.shape)}"
            )

        batch_size = int(obs_tensor.shape[0])
        flat_observations = obs_tensor.reshape(batch_size * self.num_agents, self.obs_dim)
        dist = self.actor(flat_observations)
        raw_actions = dist.mean if deterministic else dist.rsample()
        mobility = self._mobility_action_statistics(flat_observations, raw_actions=raw_actions, dist=dist)

        return {
            "actions": _reshape_or_squeeze(mobility["actions"], batch_size=batch_size, num_agents=self.num_agents, squeeze_batch=squeeze_batch),
            "mean_actions": _reshape_or_squeeze(
                mobility["mean_actions"], batch_size=batch_size, num_agents=self.num_agents, squeeze_batch=squeeze_batch
            ),
            "raw_actions": _reshape_or_squeeze(
                mobility["raw_actions"], batch_size=batch_size, num_agents=self.num_agents, squeeze_batch=squeeze_batch
            ),
            "budgets": _reshape_or_squeeze(mobility["budgets"].unsqueeze(-1), batch_size=batch_size, num_agents=self.num_agents, squeeze_batch=squeeze_batch).squeeze(-1),
            "log_prob": _reshape_or_squeeze(mobility["log_prob"].unsqueeze(-1), batch_size=batch_size, num_agents=self.num_agents, squeeze_batch=squeeze_batch).squeeze(-1),
            "entropy": _reshape_or_squeeze(mobility["entropy"].unsqueeze(-1), batch_size=batch_size, num_agents=self.num_agents, squeeze_batch=squeeze_batch).squeeze(-1),
        }

    def _build_joint_critic_input(
        self,
        *,
        agent_summaries: torch.Tensor,
        global_summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        agent_summaries, _ = _with_optional_batch(agent_summaries, expected_rank=3, name="agent_summaries")
        if agent_summaries.shape[1] != self.num_agents or agent_summaries.shape[2] != self.joint_agent_summary_dim:
            raise ValueError(
                "agent_summaries must have shape "
                f"[N, {self.joint_agent_summary_dim}] or [B, {self.num_agents}, {self.joint_agent_summary_dim}], "
                f"got {tuple(agent_summaries.shape)}"
            )
        critic_input = agent_summaries.reshape(agent_summaries.shape[0], self.num_agents * self.joint_agent_summary_dim)
        if self.joint_global_summary_dim <= 0:
            if global_summary is not None:
                raise ValueError("This model was built with joint_global_summary_dim=0, so global_summary must be omitted.")
            return critic_input

        if global_summary is None:
            raise ValueError(
                f"This model expects global_summary with trailing dim {self.joint_global_summary_dim}, but none was provided."
            )
        global_summary, _ = _with_optional_batch(global_summary, expected_rank=2, name="global_summary")
        if global_summary.shape[0] != critic_input.shape[0] or global_summary.shape[1] != self.joint_global_summary_dim:
            raise ValueError(
                "global_summary must have shape "
                f"[{critic_input.shape[0]}, {self.joint_global_summary_dim}] (batch may be implicit), "
                f"got {tuple(global_summary.shape)}"
            )
        return torch.cat([critic_input, global_summary], dim=-1)

    def joint_value(
        self,
        *,
        agent_summaries: list[list[float]] | np.ndarray | torch.Tensor,
        global_summary: list[float] | np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """新的 centralized critic 接口：读取 joint observation summaries。"""
        agent_summary_tensor = _to_tensor(agent_summaries, dtype=torch.float32, device=self.device)
        global_summary_tensor = None if global_summary is None else _to_tensor(global_summary, dtype=torch.float32, device=self.device)
        critic_input = self._build_joint_critic_input(agent_summaries=agent_summary_tensor, global_summary=global_summary_tensor)
        return self.critic(critic_input)

    def forward_joint(
        self,
        *,
        flat_observations: list[list[float]] | np.ndarray | torch.Tensor,
        task_slot_features: Any,
        offload_candidate_features: Any,
        cache_candidate_features: Any,
        offload_candidate_ids: Any | None = None,
        action_masks: dict[str, Any] | None = None,
        agent_summaries: Any | None = None,
        global_summary: Any | None = None,
        mobility_actions: Any | None = None,
        offloading_option_indices: Any | None = None,
        cache_actions: Any | None = None,
        deterministic: bool,
    ) -> dict[str, Any]:
        """joint actor / centralized critic 前向接口。

        输入格式：
        - `flat_observations`: `[N, obs_dim]` 或 `[B, N, obs_dim]`
        - `task_slot_features`: `[N, K, D_task]` 或 `[B, N, K, D_task]`
        - `offload_candidate_features`: `[N, K, C, D_candidate]` 或 `[B, N, K, C, D_candidate]`
        - `cache_candidate_features`: `[N, S, D_cache]` 或 `[B, N, S, D_cache]`
        - `offload_candidate_ids`: 可选，形状与 `[N, K, C]` / `[B, N, K, C]` 对齐；缺省时使用局部 index
        - `action_masks`: 可选 dict，支持：
          `mobility_mask`, `task_slot_mask`, `offloading_candidate_mask`, `offloading_defer_mask`, `cache_service_mask`
        - `agent_summaries`, `global_summary`: 可选 centralized critic 输入
        - `mobility_actions`, `offloading_option_indices`, `cache_actions`:
          可选，用于对给定 joint action 重算 log-prob（PPO update 阶段）

        输出格式：
        - `joint_action.mobility`: `[... , 2]`
        - `joint_action.offloading_plan_ids`: `[... , K]`
        - `joint_action.cache_priority_scores`: `[... , S]`
        - 以及各 head 的 logits / log_prob / entropy / critic_value
        """
        flat_observations_tensor = _to_tensor(flat_observations, dtype=torch.float32, device=self.device)
        flat_observations_tensor, squeeze_batch = _with_optional_batch(
            flat_observations_tensor, expected_rank=3, name="flat_observations"
        )
        if flat_observations_tensor.shape[1] != self.num_agents or flat_observations_tensor.shape[2] != self.obs_dim:
            raise ValueError(
                f"flat_observations must have shape [N, {self.obs_dim}] or [B, {self.num_agents}, {self.obs_dim}], "
                f"got {tuple(flat_observations_tensor.shape)}"
            )

        task_slot_tensor = _to_tensor(task_slot_features, dtype=torch.float32, device=self.device)
        task_slot_tensor, _ = _with_optional_batch(task_slot_tensor, expected_rank=4, name="task_slot_features")
        offload_candidate_tensor = _to_tensor(offload_candidate_features, dtype=torch.float32, device=self.device)
        offload_candidate_tensor, _ = _with_optional_batch(
            offload_candidate_tensor, expected_rank=5, name="offload_candidate_features"
        )
        cache_candidate_tensor = _to_tensor(cache_candidate_features, dtype=torch.float32, device=self.device)
        cache_candidate_tensor, _ = _with_optional_batch(cache_candidate_tensor, expected_rank=4, name="cache_candidate_features")

        batch_size = int(flat_observations_tensor.shape[0])
        task_slot_count = int(task_slot_tensor.shape[2])
        candidate_count = int(offload_candidate_tensor.shape[3])
        cache_service_count = int(cache_candidate_tensor.shape[2])

        expected_task_shape = (batch_size, self.num_agents, task_slot_count, self.task_slot_dim)
        expected_candidate_shape = (batch_size, self.num_agents, task_slot_count, candidate_count, self.offloading_candidate_dim)
        expected_cache_shape = (batch_size, self.num_agents, cache_service_count, self.cache_candidate_dim)
        if tuple(task_slot_tensor.shape) != expected_task_shape:
            raise ValueError(f"task_slot_features must have shape {expected_task_shape}, got {tuple(task_slot_tensor.shape)}")
        if tuple(offload_candidate_tensor.shape) != expected_candidate_shape:
            raise ValueError(
                f"offload_candidate_features must have shape {expected_candidate_shape}, got {tuple(offload_candidate_tensor.shape)}"
            )
        if tuple(cache_candidate_tensor.shape) != expected_cache_shape:
            raise ValueError(f"cache_candidate_features must have shape {expected_cache_shape}, got {tuple(cache_candidate_tensor.shape)}")

        flat_actor_obs = flat_observations_tensor.reshape(batch_size * self.num_agents, self.obs_dim)
        flat_task_slots = task_slot_tensor.reshape(batch_size * self.num_agents, task_slot_count, self.task_slot_dim)
        flat_candidate_features = offload_candidate_tensor.reshape(
            batch_size * self.num_agents,
            task_slot_count,
            candidate_count,
            self.offloading_candidate_dim,
        )
        flat_cache_features = cache_candidate_tensor.reshape(
            batch_size * self.num_agents,
            cache_service_count,
            self.cache_candidate_dim,
        )

        if offload_candidate_ids is None:
            candidate_id_base = torch.arange(candidate_count, dtype=torch.long, device=self.device).view(1, 1, 1, candidate_count)
            candidate_ids = candidate_id_base.expand(batch_size, self.num_agents, task_slot_count, candidate_count)
        else:
            candidate_ids = _broadcast_tensor(
                offload_candidate_ids,
                target_shape=(batch_size, self.num_agents, task_slot_count, candidate_count),
                dtype=torch.long,
                device=self.device,
                name="offload_candidate_ids",
            )
        flat_candidate_ids = candidate_ids.reshape(batch_size * self.num_agents, task_slot_count, candidate_count)

        mobility_mask = _broadcast_tensor(
            (action_masks or {}).get("mobility_mask", 1.0),
            target_shape=(batch_size, self.num_agents, self.action_dim),
            dtype=torch.float32,
            device=self.device,
            name="mobility_mask",
        ).reshape(batch_size * self.num_agents, self.action_dim)
        task_slot_mask = _broadcast_tensor(
            (action_masks or {}).get("task_slot_mask", 1.0),
            target_shape=(batch_size, self.num_agents, task_slot_count),
            dtype=torch.float32,
            device=self.device,
            name="task_slot_mask",
        ).reshape(batch_size * self.num_agents, task_slot_count)
        offloading_candidate_mask = _broadcast_tensor(
            (action_masks or {}).get("offloading_candidate_mask", 1.0),
            target_shape=(batch_size, self.num_agents, task_slot_count, candidate_count),
            dtype=torch.float32,
            device=self.device,
            name="offloading_candidate_mask",
        ).reshape(batch_size * self.num_agents, task_slot_count, candidate_count)
        offloading_defer_mask = _broadcast_tensor(
            (action_masks or {}).get("offloading_defer_mask", 1.0),
            target_shape=(batch_size, self.num_agents, task_slot_count),
            dtype=torch.float32,
            device=self.device,
            name="offloading_defer_mask",
        ).reshape(batch_size * self.num_agents, task_slot_count)
        cache_service_mask = _broadcast_tensor(
            (action_masks or {}).get("cache_service_mask", 1.0),
            target_shape=(batch_size, self.num_agents, cache_service_count),
            dtype=torch.float32,
            device=self.device,
            name="cache_service_mask",
        ).reshape(batch_size * self.num_agents, cache_service_count)

        raw_outputs = self.actor.forward_joint_heads(
            flat_observations=flat_actor_obs,
            task_slot_features=flat_task_slots,
            offload_candidate_features=flat_candidate_features,
            cache_candidate_features=flat_cache_features,
        )
        mobility_dist = raw_outputs["mobility_dist"]
        assert isinstance(mobility_dist, Normal)
        if mobility_actions is None:
            raw_mobility = mobility_dist.mean if deterministic else mobility_dist.rsample()
        else:
            provided_mobility_actions = _broadcast_tensor(
                mobility_actions,
                target_shape=(batch_size, self.num_agents, self.action_dim),
                dtype=torch.float32,
                device=self.device,
                name="mobility_actions",
            ).reshape(batch_size * self.num_agents, self.action_dim)
            raw_mobility, _ = self._raw_actions_from_constrained(flat_actor_obs, provided_mobility_actions)
        mobility = self._mobility_action_statistics(
            flat_actor_obs,
            raw_actions=raw_mobility,
            dist=mobility_dist,
            mobility_mask=mobility_mask,
        )

        candidate_logits = raw_outputs["offloading_candidate_logits"]
        defer_logits = raw_outputs["offloading_defer_logits"]
        assert isinstance(candidate_logits, torch.Tensor)
        assert isinstance(defer_logits, torch.Tensor)
        candidate_logits = candidate_logits * task_slot_mask.unsqueeze(-1)
        safe_defer_mask = torch.where(task_slot_mask > 0.5, offloading_defer_mask, torch.ones_like(offloading_defer_mask))
        offloading_logits = torch.cat([candidate_logits, defer_logits.unsqueeze(-1)], dim=-1)
        offloading_mask = torch.cat(
            [offloading_candidate_mask * task_slot_mask.unsqueeze(-1), safe_defer_mask.unsqueeze(-1)],
            dim=-1,
        )
        masked_offloading_logits, effective_offloading_mask = _masked_logits(
            offloading_logits,
            offloading_mask,
            fallback_index=candidate_count,
        )
        offloading_dist = Categorical(logits=masked_offloading_logits)
        if offloading_option_indices is None:
            offloading_choice = masked_offloading_logits.argmax(dim=-1) if deterministic else offloading_dist.sample()
        else:
            offloading_choice = _broadcast_tensor(
                offloading_option_indices,
                target_shape=(batch_size, self.num_agents, task_slot_count),
                dtype=torch.long,
                device=self.device,
                name="offloading_option_indices",
            ).reshape(batch_size * self.num_agents, task_slot_count)
        defer_column = torch.full(
            (batch_size * self.num_agents, task_slot_count, 1),
            fill_value=self.default_defer_plan_id,
            dtype=torch.long,
            device=self.device,
        )
        plan_id_lut = torch.cat([flat_candidate_ids, defer_column], dim=-1)
        selected_plan_ids = plan_id_lut.gather(-1, offloading_choice.unsqueeze(-1)).squeeze(-1)

        cache_dist = raw_outputs["cache_dist"]
        assert isinstance(cache_dist, Normal)
        if cache_actions is None:
            raw_cache = cache_dist.mean if deterministic else cache_dist.rsample()
        else:
            provided_cache_actions = _broadcast_tensor(
                cache_actions,
                target_shape=(batch_size, self.num_agents, cache_service_count),
                dtype=torch.float32,
                device=self.device,
                name="cache_actions",
            ).reshape(batch_size * self.num_agents, cache_service_count)
            raw_cache = self._raw_cache_actions_from_constrained(provided_cache_actions)
        cache = self._cache_action_statistics(
            raw_actions=raw_cache,
            dist=cache_dist,
            cache_mask=cache_service_mask,
        )

        critic_value: torch.Tensor | None = None
        critic_input: torch.Tensor | None = None
        if agent_summaries is not None or global_summary is not None or self.joint_agent_summary_dim == self.obs_dim:
            if agent_summaries is None and self.joint_agent_summary_dim == self.obs_dim:
                agent_summary_tensor = flat_observations_tensor
            elif agent_summaries is None:
                agent_summary_tensor = None
            else:
                agent_summary_tensor = _to_tensor(agent_summaries, dtype=torch.float32, device=self.device)
            global_summary_tensor = None if global_summary is None else _to_tensor(global_summary, dtype=torch.float32, device=self.device)
            if agent_summary_tensor is not None:
                critic_input = self._build_joint_critic_input(
                    agent_summaries=agent_summary_tensor,
                    global_summary=global_summary_tensor,
                )
                critic_value = self.critic(critic_input)

        return {
            "joint_action": {
                "mobility": _reshape_or_squeeze(
                    mobility["actions"], batch_size=batch_size, num_agents=self.num_agents, squeeze_batch=squeeze_batch
                ),
                "offloading_plan_ids": _reshape_or_squeeze(
                    selected_plan_ids.unsqueeze(-1),
                    batch_size=batch_size,
                    num_agents=self.num_agents,
                    squeeze_batch=squeeze_batch,
                ).squeeze(-1),
                "cache_priority_scores": _reshape_or_squeeze(
                    cache["actions"], batch_size=batch_size, num_agents=self.num_agents, squeeze_batch=squeeze_batch
                ),
            },
            "mobility": {
                "actions": _reshape_or_squeeze(
                    mobility["actions"], batch_size=batch_size, num_agents=self.num_agents, squeeze_batch=squeeze_batch
                ),
                "mean_actions": _reshape_or_squeeze(
                    mobility["mean_actions"], batch_size=batch_size, num_agents=self.num_agents, squeeze_batch=squeeze_batch
                ),
                "raw_actions": _reshape_or_squeeze(
                    mobility["raw_actions"], batch_size=batch_size, num_agents=self.num_agents, squeeze_batch=squeeze_batch
                ),
                "log_prob": _reshape_or_squeeze(
                    mobility["log_prob"].unsqueeze(-1),
                    batch_size=batch_size,
                    num_agents=self.num_agents,
                    squeeze_batch=squeeze_batch,
                ).squeeze(-1),
                "entropy": _reshape_or_squeeze(
                    mobility["entropy"].unsqueeze(-1),
                    batch_size=batch_size,
                    num_agents=self.num_agents,
                    squeeze_batch=squeeze_batch,
                ).squeeze(-1),
                "budgets": _reshape_or_squeeze(
                    mobility["budgets"].unsqueeze(-1),
                    batch_size=batch_size,
                    num_agents=self.num_agents,
                    squeeze_batch=squeeze_batch,
                ).squeeze(-1),
            },
            "offloading": {
                "option_logits": _reshape_or_squeeze(
                    masked_offloading_logits,
                    batch_size=batch_size,
                    num_agents=self.num_agents,
                    squeeze_batch=squeeze_batch,
                ),
                "option_mask": _reshape_or_squeeze(
                    effective_offloading_mask,
                    batch_size=batch_size,
                    num_agents=self.num_agents,
                    squeeze_batch=squeeze_batch,
                ),
                "selected_option_index": _reshape_or_squeeze(
                    offloading_choice.unsqueeze(-1),
                    batch_size=batch_size,
                    num_agents=self.num_agents,
                    squeeze_batch=squeeze_batch,
                ).squeeze(-1),
                "selected_plan_ids": _reshape_or_squeeze(
                    selected_plan_ids.unsqueeze(-1),
                    batch_size=batch_size,
                    num_agents=self.num_agents,
                    squeeze_batch=squeeze_batch,
                ).squeeze(-1),
                "log_prob": _reshape_or_squeeze(
                    offloading_dist.log_prob(offloading_choice).unsqueeze(-1),
                    batch_size=batch_size,
                    num_agents=self.num_agents,
                    squeeze_batch=squeeze_batch,
                ).squeeze(-1),
                "entropy": _reshape_or_squeeze(
                    offloading_dist.entropy().unsqueeze(-1),
                    batch_size=batch_size,
                    num_agents=self.num_agents,
                    squeeze_batch=squeeze_batch,
                ).squeeze(-1),
                "defer_option_index": candidate_count,
                "defer_plan_id": self.default_defer_plan_id,
            },
            "cache": {
                "priority_scores": _reshape_or_squeeze(
                    cache["actions"], batch_size=batch_size, num_agents=self.num_agents, squeeze_batch=squeeze_batch
                ),
                "mean_priority_scores": _reshape_or_squeeze(
                    cache["mean_actions"], batch_size=batch_size, num_agents=self.num_agents, squeeze_batch=squeeze_batch
                ),
                "raw_actions": _reshape_or_squeeze(
                    cache["raw_actions"], batch_size=batch_size, num_agents=self.num_agents, squeeze_batch=squeeze_batch
                ),
                "log_prob": _reshape_or_squeeze(
                    cache["log_prob"].unsqueeze(-1),
                    batch_size=batch_size,
                    num_agents=self.num_agents,
                    squeeze_batch=squeeze_batch,
                ).squeeze(-1),
                "entropy": _reshape_or_squeeze(
                    cache["entropy"].unsqueeze(-1),
                    batch_size=batch_size,
                    num_agents=self.num_agents,
                    squeeze_batch=squeeze_batch,
                ).squeeze(-1),
                "actions": _reshape_or_squeeze(
                    cache["actions"], batch_size=batch_size, num_agents=self.num_agents, squeeze_batch=squeeze_batch
                ),
                "service_mask": _reshape_or_squeeze(
                    cache_service_mask,
                    batch_size=batch_size,
                    num_agents=self.num_agents,
                    squeeze_batch=squeeze_batch,
                ),
            },
            "critic": {
                "value": None if critic_value is None else (critic_value.squeeze(0) if squeeze_batch else critic_value),
                "critic_input": None if critic_input is None else (critic_input.squeeze(0) if squeeze_batch else critic_input),
            },
        }

    def act(self, observations: list[list[float]], *, deterministic: bool) -> tuple[list[list[float]], list[float]]:
        """基于局部观测为所有 UAV 采样或输出确定性动作。"""
        with torch.no_grad():
            outputs = self.forward_legacy(observations, deterministic=deterministic)
        actions = outputs["actions"]
        log_probs = outputs["log_prob"]
        return actions.cpu().numpy().tolist(), [float(item) for item in log_probs.cpu().tolist()]

    def value(self, state: list[float]) -> float:
        """legacy value 接口；默认仍读取拼接后的 flat global state。"""
        state_tensor = torch.as_tensor(np.asarray([state], dtype=np.float32), device=self.device)
        if state_tensor.shape[1] != self.critic_input_dim:
            raise ValueError(
                f"value() expects critic input dim {self.critic_input_dim}, got {state_tensor.shape[1]}. "
                "If you upgraded the critic to joint summaries, call joint_value() instead."
            )
        with torch.no_grad():
            value = self.critic(state_tensor)
        return float(value.item())

    def _iter_minibatches(self, total_steps: int, minibatch_size: int) -> list[np.ndarray]:
        indices = np.arange(total_steps, dtype=np.int64)
        np.random.shuffle(indices)
        return [indices[start : start + minibatch_size] for start in range(0, total_steps, minibatch_size)]

    def update(self, *, batch: RolloutBatch, config: MinimalMARLConfig) -> dict[str, float]:
        """执行多轮 PPO 更新，并同步衰减策略动作方差。"""
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

        decay_stats = self.decay_exploration_stds()

        divisor = float(max(1, update_steps))
        return {
            "actor_loss": total_actor_loss / divisor,
            "critic_loss": total_critic_loss / divisor,
            "entropy": total_entropy / divisor,
            "action_std_scale": decay_stats["action_std_scale"],
            "cache_action_std_scale": decay_stats["cache_action_std_scale"],
        }

    def save(self, path: str | Path) -> None:
        """保存模型参数、优化器状态和关键张量契约。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "num_agents": self.num_agents,
            "state_dim": self.state_dim,
            "critic_input_dim": self.critic_input_dim,
            "joint_agent_summary_dim": self.joint_agent_summary_dim,
            "joint_global_summary_dim": self.joint_global_summary_dim,
            "task_slot_dim": self.task_slot_dim,
            "offloading_candidate_dim": self.offloading_candidate_dim,
            "cache_candidate_dim": self.cache_candidate_dim,
            "default_defer_plan_id": self.default_defer_plan_id,
            "device": self.resolved_device,
            "requested_device": self.requested_device,
            "actor_state": self.actor.state_dict(),
            "critic_state": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict() if self.actor_optimizer else None,
            "critic_optimizer": self.critic_optimizer.state_dict() if self.critic_optimizer else None,
            "action_std_min": self.action_std_min,
            "action_std_decay": self.action_std_decay,
            "cache_action_std_min": self.cache_action_std_min,
            "cache_action_std_decay": self.cache_action_std_decay,
            "use_movement_budget": self.use_movement_budget,
            "max_user_blocks": self.max_user_blocks,
            "user_feature_dim": self.user_feature_dim,
        }
        torch.save(payload, target)

    @classmethod
    def load(cls, path: str | Path, *, seed: int, device: str = "auto") -> "MinimalMultiAgentActorCritic":
        requested_device = normalize_device_request(device)
        resolved_device = resolve_device(requested_device)
        configure_torch_runtime(resolved_device)
        payload = torch.load(Path(path), map_location=resolved_device, weights_only=False)
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
            cache_action_std_init=float(torch.exp(payload["actor_state"].get("cache_log_std", torch.tensor(math.log(0.35)))).mean().item()),
            cache_action_std_min=float(payload.get("cache_action_std_min", 0.10)),
            cache_action_std_decay=float(payload.get("cache_action_std_decay", 0.995)),
            use_movement_budget=bool(payload.get("use_movement_budget", True)),
            hidden_dim=int(payload["actor_state"]["backbone.0.weight"].shape[0]),
            max_user_blocks=int(payload.get("max_user_blocks", 3)),
            user_feature_dim=int(payload.get("user_feature_dim", 5)),
            task_slot_dim=int(payload.get("task_slot_dim", 11)),
            offloading_candidate_dim=int(payload.get("offloading_candidate_dim", 12)),
            cache_candidate_dim=int(payload.get("cache_candidate_dim", 9)),
            joint_agent_summary_dim=int(payload.get("joint_agent_summary_dim", payload["obs_dim"])),
            joint_global_summary_dim=int(payload.get("joint_global_summary_dim", 0)),
            default_defer_plan_id=int(payload.get("default_defer_plan_id", DEFAULT_DEFER_PLAN_ID)),
            device=requested_device,
        )
        model.actor.load_state_dict(payload["actor_state"], strict=False)
        model.critic.load_state_dict(payload["critic_state"], strict=False)
        return model

    def runtime_device_info(self) -> dict[str, Any]:
        """导出当前模型实例实际使用的 torch 设备信息。"""
        return describe_runtime_device(self.requested_device, resolved_device=self.resolved_device)

    def tensor_contract(self) -> dict[str, Any]:
        """导出 legacy 与 joint 两条接口的张量契约。"""
        return {
            "observation_batch_shape": ["T", self.num_agents, self.obs_dim],
            "central_state_shape": ["T", self.critic_input_dim],
            "action_batch_shape": ["T", self.num_agents, self.action_dim],
            "team_reward_shape": ["T"],
            "value_shape": ["T"],
            "advantage_shape": ["T"],
            "joint_policy": {
                "flat_observation_shape": ["B", self.num_agents, self.obs_dim],
                "task_slot_feature_shape": ["B", self.num_agents, "K", self.task_slot_dim],
                "offloading_candidate_feature_shape": ["B", self.num_agents, "K", "C", self.offloading_candidate_dim],
                "cache_candidate_feature_shape": ["B", self.num_agents, "S", self.cache_candidate_dim],
                "offloading_plan_id_shape": ["B", self.num_agents, "K"],
                "cache_priority_shape": ["B", self.num_agents, "S"],
                "critic_agent_summary_shape": ["B", self.num_agents, self.joint_agent_summary_dim],
                "critic_global_summary_shape": None if self.joint_global_summary_dim <= 0 else ["B", self.joint_global_summary_dim],
            },
        }
