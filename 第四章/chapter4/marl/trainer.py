"""第四章 MARL 训练执行模块。

该模块同时支持两条训练路径：

- `hybrid_joint`: 主路径，使用 joint actor + centralized critic 的 hybrid MAPPO/PPO
- `legacy_mobility_only`: 兼容 baseline，保留旧 mobility-only PPO

主路径会直接基于结构化 observation 采样 mobility/offloading/caching 联合动作，
并在 joint-action 环境执行器上收集 rollout 与更新策略。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from common.uav_mec.core.observation import build_observations
from common.uav_mec.core.state import cache_candidate_feature_fields
from common.uav_mec.core.state import offloading_candidate_feature_fields
from common.uav_mec.core.state import task_slot_feature_fields
from common.uav_mec.logging_utils import write_json

from ..env import Chapter4Env
from .buffer import HybridRolloutBatch
from .buffer import HybridRolloutBuffer
from .buffer import RolloutBuffer
from .config import MinimalMARLConfig
from .model import MinimalMultiAgentActorCritic
from ..results_paths import trainer_results_dir


TASK_SLOT_FIELDS = tuple(task_slot_feature_fields())
OFFLOAD_CANDIDATE_FIELDS = tuple(offloading_candidate_feature_fields())
CACHE_CANDIDATE_FIELDS = tuple(cache_candidate_feature_fields())


def _mean_action_norm(actions: list[list[float]]) -> float:
    if not actions:
        return 0.0
    return float(sum((action[0] ** 2 + action[1] ** 2) ** 0.5 for action in actions) / len(actions))


def _fit_vector(values: list[float], target_dim: int) -> list[float]:
    if target_dim <= 0:
        return []
    clipped = [float(item) for item in values[:target_dim]]
    if len(clipped) < target_dim:
        clipped.extend([0.0] * (target_dim - len(clipped)))
    return clipped


def _numpy(value: Any, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if dtype is None:
        return np.asarray(array)
    return np.asarray(array, dtype=dtype)


def _scalar(value: Any) -> float:
    array = _numpy(value, dtype=np.float32)
    return float(array.reshape(-1)[0]) if array.size else 0.0


def _joint_action_slot_count(env: Chapter4Env) -> float:
    return float(max(1, env.config.num_uavs * env.config.task_arrival_max_per_step))


def _step_energy_reference(env: Chapter4Env) -> float:
    cfg = env.config
    max_uav_move = cfg.num_uavs * cfg.uav_move_energy_per_meter * cfg.uav_speed * cfg.time_slot_duration
    max_uav_compute = cfg.num_uavs * cfg.task_arrival_max_per_step * cfg.task_cpu_cycles_range[1] * cfg.uav_compute_energy_per_cycle
    max_ue_local = cfg.num_users * cfg.task_arrival_max_per_step * cfg.task_cpu_cycles_range[1] * cfg.ue_local_compute_energy_per_cycle
    max_ue_uplink = cfg.num_users * cfg.task_arrival_max_per_step * cfg.ue_uplink_power_w * cfg.time_slot_duration
    max_uav_backhaul = cfg.num_uavs * cfg.task_arrival_max_per_step * cfg.uav_tx_power_w * cfg.time_slot_duration
    max_bs_compute = cfg.num_users * cfg.task_arrival_max_per_step * cfg.task_cpu_cycles_range[1] * cfg.bs_compute_energy_per_cycle
    max_bs_backhaul = cfg.num_uavs * cfg.task_arrival_max_per_step * cfg.bs_backhaul_tx_power_w * cfg.time_slot_duration
    return max(
        1.0,
        float(max_uav_move + max_uav_compute + max_ue_local + max_ue_uplink + max_uav_backhaul + max_bs_compute + max_bs_backhaul),
    )


def _structured_observations(env: Chapter4Env) -> list[dict[str, object]]:
    current_time = env.current_step * env.config.time_slot_duration
    structured = build_observations(
        uavs=env.uavs,
        users=env.users,
        pending_tasks=env.pending_tasks,
        config=env.config,
        current_time=current_time,
        bs=env.bs,
        service_catalog=env.service_catalog,
        tdma_queue=env.tdma_queue,
        compute_queue=env.compute_queue,
        export_mode="structured_joint",
    )
    assert isinstance(structured, list)
    return structured


def _global_summary(env: Chapter4Env, structured: list[dict[str, object]], *, target_dim: int) -> list[float]:
    if target_dim <= 0:
        return []
    num_users_scale = max(1, env.config.num_users)
    backlog_scale = max(1, env.config.num_users * env.config.task_arrival_max_per_step)
    task_slot_mask = np.asarray([item["action_masks"]["task_slot_mask"] for item in structured], dtype=np.float32)
    candidate_mask = np.asarray([item["action_masks"]["offloading_candidate_mask"] for item in structured], dtype=np.float32)
    cache_mask = np.asarray([item["action_masks"]["cache_service_mask"] for item in structured], dtype=np.float32)
    raw_values = [
        float(env.current_step) / max(1, env.config.steps_per_episode),
        float(len(env.pending_tasks)) / backlog_scale,
        float(np.mean([uav.energy_ratio for uav in env.uavs])) if env.uavs else 0.0,
        float(np.mean([uav.current_tx_queue_length / num_users_scale for uav in env.uavs])) if env.uavs else 0.0,
        float(np.mean([uav.current_compute_queue_length / num_users_scale for uav in env.uavs])) if env.uavs else 0.0,
        float(np.mean([uav.current_backlog_load / backlog_scale for uav in env.uavs])) if env.uavs else 0.0,
        float(task_slot_mask.mean()) if task_slot_mask.size else 0.0,
        float(candidate_mask.mean()) if candidate_mask.size else 0.0,
        float(cache_mask.mean()) if cache_mask.size else 0.0,
    ]
    return _fit_vector(raw_values, target_dim)


def build_joint_policy_inputs(env: Chapter4Env, *, config: MinimalMARLConfig) -> dict[str, Any]:
    """从环境当前状态构造 joint actor / centralized critic 所需张量。"""
    structured = _structured_observations(env)
    flat_observations = np.asarray([item["flat_legacy"] for item in structured], dtype=np.float32)
    agent_summary_dim = int(config.joint_agent_summary_dim) if config.joint_agent_summary_dim > 0 else int(flat_observations.shape[-1])
    critic_agent_summaries = np.asarray(
        [_fit_vector(list(map(float, observation.tolist())), agent_summary_dim) for observation in flat_observations],
        dtype=np.float32,
    )
    critic_global_summary = np.asarray(
        _global_summary(env, structured, target_dim=config.joint_global_summary_dim),
        dtype=np.float32,
    )
    task_slot_features = np.asarray(
        [
            [
                [[float(slot[field]) for field in TASK_SLOT_FIELDS] for slot in item["task_slots"]]
                for item in structured
            ]
        ][0],
        dtype=np.float32,
    )
    offload_candidate_features = np.asarray(
        [
            [
                [
                    [float(candidate[field]) for field in OFFLOAD_CANDIDATE_FIELDS]
                    for candidate in slot_candidates
                ]
                for slot_candidates in item["offload_candidates"]
            ]
            for item in structured
        ],
        dtype=np.float32,
    )
    offload_candidate_ids = np.asarray(
        [
            [
                [int(candidate["candidate_id"]) for candidate in slot_candidates]
                for slot_candidates in item["offload_candidates"]
            ]
            for item in structured
        ],
        dtype=np.int64,
    )
    cache_candidate_features = np.asarray(
        [
            [
                [float(candidate[field]) for field in CACHE_CANDIDATE_FIELDS]
                for candidate in item["cache_candidates"]
            ]
            for item in structured
        ],
        dtype=np.float32,
    )
    action_masks = {
        "mobility_mask": np.asarray([item["action_masks"]["mobility_mask"] for item in structured], dtype=np.float32),
        "task_slot_mask": np.asarray([item["action_masks"]["task_slot_mask"] for item in structured], dtype=np.float32),
        "offloading_candidate_mask": np.asarray(
            [item["action_masks"]["offloading_candidate_mask"] for item in structured],
            dtype=np.float32,
        ),
        "offloading_defer_mask": np.asarray(
            [item["action_masks"]["offloading_defer_mask"] for item in structured],
            dtype=np.float32,
        ),
        "cache_service_mask": np.asarray(
            [item["action_masks"]["cache_service_mask"] for item in structured],
            dtype=np.float32,
        ),
    }
    return {
        "structured_observations": structured,
        "flat_observations": flat_observations,
        "critic_agent_summaries": critic_agent_summaries,
        "critic_global_summary": critic_global_summary,
        "task_slot_features": task_slot_features,
        "offload_candidate_features": offload_candidate_features,
        "offload_candidate_ids": offload_candidate_ids,
        "cache_candidate_features": cache_candidate_features,
        "action_masks": action_masks,
    }


def joint_action_payloads(policy_output: dict[str, Any]) -> list[dict[str, object]]:
    mobility = _numpy(policy_output["joint_action"]["mobility"], dtype=np.float32)
    offloading_plan_ids = _numpy(policy_output["joint_action"]["offloading_plan_ids"], dtype=np.int64)
    cache_priority_scores = _numpy(policy_output["joint_action"]["cache_priority_scores"], dtype=np.float32)
    payloads: list[dict[str, object]] = []
    for agent_index in range(mobility.shape[0]):
        payloads.append(
            {
                "mobility": {"dx": float(mobility[agent_index, 0]), "dy": float(mobility[agent_index, 1])},
                "offloading": {"task_slot_plan_ids": [int(item) for item in offloading_plan_ids[agent_index].tolist()]},
                "caching": {"service_priorities": [float(item) for item in cache_priority_scores[agent_index].tolist()]},
            }
        )
    return payloads


def _shape_joint_reward(
    *,
    config: MinimalMARLConfig,
    env: Chapter4Env,
    step_result: dict[str, Any],
    policy_output: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """面向联合系统目标的 reward：completion/latency/energy/cache/deadline/reliability/backlog。"""
    step_signals = step_result["step_signals"]
    info = step_result["info"]
    delta_energy = float(step_signals["step_total_energy"])
    completion_ratio = float(step_signals["step_completion_ratio"])
    cache_hit_ratio = float(step_signals["step_cache_hit_ratio"])
    backlog_ratio = float(step_signals["step_backlog_ratio"])
    deadline_violation_ratio = float(step_signals["step_deadline_violation_ratio"])
    reliability_violation_ratio = float(step_signals["step_reliability_violation_ratio"])
    expired_ratio = float(step_signals["step_expired_ratio"])
    latency = float(step_signals["step_average_latency"])
    mobility_actions = _numpy(policy_output["joint_action"]["mobility"], dtype=np.float32)
    action_magnitude = _mean_action_norm(mobility_actions.tolist())
    latency_norm = latency / max(env.config.task_slack_range[1], 1e-6)
    energy_norm = delta_energy / _step_energy_reference(env)
    action_feedback = info.get("action_feedback", {})
    slot_count = _joint_action_slot_count(env)
    invalid_action_ratio = float(action_feedback.get("invalid_plan_reject_count", 0)) / slot_count
    infeasible_action_ratio = float(action_feedback.get("infeasible_plan_reject_count", 0)) / slot_count
    reward = (
        config.reward_completion_weight * completion_ratio
        + config.reward_cache_hit_weight * cache_hit_ratio
        - config.reward_latency_weight * latency_norm
        - config.reward_energy_weight * energy_norm
        - config.reward_backlog_weight * backlog_ratio
        - config.reward_deadline_weight * deadline_violation_ratio
        - config.reward_reliability_weight * reliability_violation_ratio
        - config.reward_expired_weight * expired_ratio
        - config.reward_action_magnitude_weight * action_magnitude
        - config.reward_invalid_action_weight * invalid_action_ratio
        - config.reward_infeasible_action_weight * infeasible_action_ratio
    )
    feedback = action_feedback
    return float(reward), {
        "step_completion_ratio": completion_ratio,
        "step_cache_hit_ratio": cache_hit_ratio,
        "step_backlog_ratio": backlog_ratio,
        "step_deadline_violation_ratio": deadline_violation_ratio,
        "step_reliability_violation_ratio": reliability_violation_ratio,
        "step_expired_ratio": expired_ratio,
        "step_latency": latency,
        "step_latency_norm": latency_norm,
        "step_energy": delta_energy,
        "step_energy_norm": energy_norm,
        "step_action_magnitude": action_magnitude,
        "step_invalid_action_reject_ratio": invalid_action_ratio,
        "step_infeasible_action_reject_ratio": infeasible_action_ratio,
        "step_completed_tasks": float(step_signals["completed_tasks"]),
        "step_expired_tasks": float(step_signals["expired_tasks"]),
        "step_pending_tasks": float(step_signals["pending_tasks"]),
        "step_deadline_violations": float(step_signals["deadline_violations"]),
        "step_reliability_violations": float(step_signals["reliability_violations"]),
        "step_executed_plan_count": float(feedback.get("executed_plan_count", 0)),
        "step_cache_event_count": float(feedback.get("cache_event_count", 0)),
    }


def _clipped_policy_loss(ratio: torch.Tensor, advantage: torch.Tensor, clip_eps: float) -> torch.Tensor:
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    return -torch.min(unclipped, clipped).mean()


def hybrid_update(
    *,
    agent: MinimalMultiAgentActorCritic,
    batch: HybridRolloutBatch,
    config: MinimalMARLConfig,
) -> dict[str, float]:
    """执行 hybrid MAPPO/PPO 更新。"""
    if agent.actor_optimizer is None or agent.critic_optimizer is None:
        agent.configure_optimizers(actor_lr=config.actor_lr, critic_lr=config.critic_lr)

    observations = torch.as_tensor(batch.observations, dtype=torch.float32, device=agent.device)
    critic_agent_summaries = torch.as_tensor(batch.critic_agent_summaries, dtype=torch.float32, device=agent.device)
    critic_global_summaries = torch.as_tensor(batch.critic_global_summaries, dtype=torch.float32, device=agent.device)
    task_slot_features = torch.as_tensor(batch.task_slot_features, dtype=torch.float32, device=agent.device)
    offload_candidate_features = torch.as_tensor(batch.offload_candidate_features, dtype=torch.float32, device=agent.device)
    cache_candidate_features = torch.as_tensor(batch.cache_candidate_features, dtype=torch.float32, device=agent.device)
    offload_candidate_ids = torch.as_tensor(batch.offload_candidate_ids, dtype=torch.long, device=agent.device)
    mobility_masks = torch.as_tensor(batch.mobility_masks, dtype=torch.float32, device=agent.device)
    task_slot_masks = torch.as_tensor(batch.task_slot_masks, dtype=torch.float32, device=agent.device)
    offloading_candidate_masks = torch.as_tensor(batch.offloading_candidate_masks, dtype=torch.float32, device=agent.device)
    offloading_defer_masks = torch.as_tensor(batch.offloading_defer_masks, dtype=torch.float32, device=agent.device)
    cache_service_masks = torch.as_tensor(batch.cache_service_masks, dtype=torch.float32, device=agent.device)
    mobility_actions = torch.as_tensor(batch.mobility_actions, dtype=torch.float32, device=agent.device)
    offloading_option_indices = torch.as_tensor(batch.offloading_option_indices, dtype=torch.long, device=agent.device)
    cache_actions = torch.as_tensor(batch.cache_actions, dtype=torch.float32, device=agent.device)
    old_mobility_log_probs = torch.as_tensor(batch.mobility_log_probs, dtype=torch.float32, device=agent.device)
    old_offloading_log_probs = torch.as_tensor(batch.offloading_log_probs, dtype=torch.float32, device=agent.device)
    old_cache_log_probs = torch.as_tensor(batch.cache_log_probs, dtype=torch.float32, device=agent.device)
    advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=agent.device)
    returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=agent.device)
    old_values = torch.as_tensor(batch.values, dtype=torch.float32, device=agent.device)

    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_mobility_loss = 0.0
    total_offloading_loss = 0.0
    total_cache_loss = 0.0
    total_mobility_entropy = 0.0
    total_offloading_entropy = 0.0
    total_cache_entropy = 0.0
    total_joint_ratio = 0.0
    update_steps = 0
    minibatch_size = max(1, min(config.minibatch_size, observations.shape[0]))
    loss_weight_sum = max(1e-6, config.mobility_loss_coef + config.offloading_loss_coef + config.cache_loss_coef)

    indices = np.arange(observations.shape[0], dtype=np.int64)
    for _ in range(config.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, observations.shape[0], minibatch_size):
            step_indices = indices[start : start + minibatch_size]

            global_summary_mb: torch.Tensor | None = None
            if critic_global_summaries.shape[-1] > 0:
                global_summary_mb = critic_global_summaries[step_indices]

            outputs = agent.forward_joint(
                flat_observations=observations[step_indices],
                task_slot_features=task_slot_features[step_indices],
                offload_candidate_features=offload_candidate_features[step_indices],
                cache_candidate_features=cache_candidate_features[step_indices],
                offload_candidate_ids=offload_candidate_ids[step_indices],
                action_masks={
                    "mobility_mask": mobility_masks[step_indices],
                    "task_slot_mask": task_slot_masks[step_indices],
                    "offloading_candidate_mask": offloading_candidate_masks[step_indices],
                    "offloading_defer_mask": offloading_defer_masks[step_indices],
                    "cache_service_mask": cache_service_masks[step_indices],
                },
                agent_summaries=critic_agent_summaries[step_indices],
                global_summary=global_summary_mb,
                mobility_actions=mobility_actions[step_indices],
                offloading_option_indices=offloading_option_indices[step_indices],
                cache_actions=cache_actions[step_indices],
                deterministic=False,
            )

            new_mobility_log_probs = outputs["mobility"]["log_prob"]
            new_offloading_log_probs = outputs["offloading"]["log_prob"].sum(dim=-1)
            new_cache_log_probs = outputs["cache"]["log_prob"]
            mobility_entropy = outputs["mobility"]["entropy"]
            offloading_entropy = outputs["offloading"]["entropy"].sum(dim=-1)
            cache_entropy = outputs["cache"]["entropy"]
            values = outputs["critic"]["value"]
            assert isinstance(values, torch.Tensor)

            adv_mb = advantages[step_indices].unsqueeze(-1).expand(-1, agent.num_agents)
            old_mobility_mb = old_mobility_log_probs[step_indices]
            old_offloading_mb = old_offloading_log_probs[step_indices]
            old_cache_mb = old_cache_log_probs[step_indices]

            mobility_ratio = torch.exp(new_mobility_log_probs - old_mobility_mb)
            offloading_ratio = torch.exp(new_offloading_log_probs - old_offloading_mb)
            cache_ratio = torch.exp(new_cache_log_probs - old_cache_mb)
            joint_ratio = torch.exp(
                (new_mobility_log_probs + new_offloading_log_probs + new_cache_log_probs)
                - (old_mobility_mb + old_offloading_mb + old_cache_mb)
            )

            mobility_policy_loss = _clipped_policy_loss(mobility_ratio, adv_mb, config.ppo_clip_eps)
            offloading_policy_loss = _clipped_policy_loss(offloading_ratio, adv_mb, config.ppo_clip_eps)
            cache_policy_loss = _clipped_policy_loss(cache_ratio, adv_mb, config.ppo_clip_eps)
            entropy_bonus = (
                config.mobility_entropy_coef * mobility_entropy.mean()
                + config.offloading_entropy_coef * offloading_entropy.mean()
                + config.cache_entropy_coef * cache_entropy.mean()
            )
            actor_loss = (
                config.mobility_loss_coef * mobility_policy_loss
                + config.offloading_loss_coef * offloading_policy_loss
                + config.cache_loss_coef * cache_policy_loss
            ) / loss_weight_sum - entropy_bonus

            value_delta = values - old_values[step_indices]
            clipped_values = old_values[step_indices] + value_delta.clamp(-config.value_clip_eps, config.value_clip_eps)
            critic_loss_unclipped = (values - returns[step_indices]).pow(2)
            critic_loss_clipped = (clipped_values - returns[step_indices]).pow(2)
            critic_loss = config.value_loss_coef * torch.max(critic_loss_unclipped, critic_loss_clipped).mean()

            agent.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), config.gradient_clip)
            agent.actor_optimizer.step()

            agent.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), config.gradient_clip)
            agent.critic_optimizer.step()

            total_actor_loss += float(actor_loss.item())
            total_critic_loss += float(critic_loss.item())
            total_mobility_loss += float(mobility_policy_loss.item())
            total_offloading_loss += float(offloading_policy_loss.item())
            total_cache_loss += float(cache_policy_loss.item())
            total_mobility_entropy += float(mobility_entropy.mean().item())
            total_offloading_entropy += float(offloading_entropy.mean().item())
            total_cache_entropy += float(cache_entropy.mean().item())
            total_joint_ratio += float(joint_ratio.mean().item())
            update_steps += 1

    decay_stats = agent.decay_exploration_stds()
    divisor = float(max(1, update_steps))
    return {
        "actor_loss": total_actor_loss / divisor,
        "critic_loss": total_critic_loss / divisor,
        "mobility_policy_loss": total_mobility_loss / divisor,
        "offloading_policy_loss": total_offloading_loss / divisor,
        "cache_policy_loss": total_cache_loss / divisor,
        "mobility_entropy": total_mobility_entropy / divisor,
        "offloading_entropy": total_offloading_entropy / divisor,
        "cache_entropy": total_cache_entropy / divisor,
        "joint_ratio_mean": total_joint_ratio / divisor,
        **decay_stats,
    }


def _collect_hybrid_episode(
    *,
    agent: MinimalMultiAgentActorCritic,
    config: MinimalMARLConfig,
    episode_index: int,
) -> tuple[HybridRolloutBuffer, dict[str, Any], float]:
    env = Chapter4Env(
        {
            "seed": config.seed + episode_index,
            "num_uavs": config.num_uavs,
            "assignment_rule": config.assignment_rule,
        }
    )
    env.reset(seed=config.seed + episode_index)
    buffer = HybridRolloutBuffer()
    last_step: dict[str, Any] | None = None
    reward_breakdowns: list[dict[str, float]] = []
    while True:
        policy_inputs = build_joint_policy_inputs(env, config=config)
        with torch.no_grad():
            policy_output = agent.forward_joint(
                flat_observations=policy_inputs["flat_observations"],
                task_slot_features=policy_inputs["task_slot_features"],
                offload_candidate_features=policy_inputs["offload_candidate_features"],
                cache_candidate_features=policy_inputs["cache_candidate_features"],
                offload_candidate_ids=policy_inputs["offload_candidate_ids"],
                action_masks=policy_inputs["action_masks"],
                agent_summaries=policy_inputs["critic_agent_summaries"],
                global_summary=(
                    None
                    if policy_inputs["critic_global_summary"].shape[-1] <= 0
                    else policy_inputs["critic_global_summary"]
                ),
                deterministic=False,
            )
        step_actions = joint_action_payloads(policy_output)
        critic_value = _scalar(policy_output["critic"]["value"])
        last_step = env.step(step_actions, scheduler_mode="joint_action")
        if last_step["info"].get("scheduler_mode") != "joint_action":
            raise RuntimeError("Hybrid trainer expected scheduler_mode='joint_action', but environment resolved differently.")
        team_reward, reward_breakdown = _shape_joint_reward(
            config=config,
            env=env,
            step_result=last_step,
            policy_output=policy_output,
        )
        reward_breakdowns.append(reward_breakdown)
        done = bool(last_step["terminated"] or last_step["truncated"])
        buffer.add(
            critic_agent_summaries=policy_inputs["critic_agent_summaries"].tolist(),
            critic_global_summary=policy_inputs["critic_global_summary"].tolist(),
            observations=policy_inputs["flat_observations"].tolist(),
            task_slot_features=policy_inputs["task_slot_features"].tolist(),
            offload_candidate_features=policy_inputs["offload_candidate_features"].tolist(),
            cache_candidate_features=policy_inputs["cache_candidate_features"].tolist(),
            offload_candidate_ids=policy_inputs["offload_candidate_ids"].tolist(),
            action_masks={key: value.tolist() for key, value in policy_inputs["action_masks"].items()},
            mobility_actions=_numpy(policy_output["joint_action"]["mobility"], dtype=np.float32).tolist(),
            offloading_option_indices=_numpy(policy_output["offloading"]["selected_option_index"], dtype=np.int64).tolist(),
            offloading_plan_ids=_numpy(policy_output["joint_action"]["offloading_plan_ids"], dtype=np.int64).tolist(),
            cache_actions=_numpy(policy_output["joint_action"]["cache_priority_scores"], dtype=np.float32).tolist(),
            mobility_log_probs=_numpy(policy_output["mobility"]["log_prob"], dtype=np.float32).tolist(),
            offloading_log_probs=np.asarray(
                _numpy(policy_output["offloading"]["log_prob"], dtype=np.float32).sum(axis=-1),
                dtype=np.float32,
            ).tolist(),
            cache_log_probs=_numpy(policy_output["cache"]["log_prob"], dtype=np.float32).tolist(),
            reward=team_reward,
            done=done,
            value=critic_value,
        )
        if done:
            break
    episode_log = env.export_episode_log(episode_index=episode_index, seed=config.seed + episode_index)
    assert last_step is not None
    return (
        buffer,
        {
            "summary": env.export_episode_summary(),
            "episode_log": episode_log,
            "last_info": last_step["info"],
            "reward_breakdowns": reward_breakdowns,
        },
        0.0,
    )


def _shape_legacy_reward(
    *,
    config: MinimalMARLConfig,
    env: Chapter4Env,
    step_result: dict[str, Any],
    actions: list[list[float]],
) -> tuple[float, dict[str, float]]:
    step_signals = step_result["step_signals"]
    delta_energy = float(step_signals["step_total_energy"])
    completion_ratio = float(step_signals["step_completion_ratio"])
    cache_hit_ratio = float(step_signals["step_cache_hit_ratio"])
    backlog_ratio = float(step_signals["step_backlog_ratio"])
    deadline_violation_ratio = float(step_signals["step_deadline_violation_ratio"])
    reliability_violation_ratio = float(step_signals["step_reliability_violation_ratio"])
    expired_ratio = float(step_signals["step_expired_ratio"])
    latency = float(step_signals["step_average_latency"])
    action_magnitude = _mean_action_norm(actions)
    latency_norm = latency / max(env.config.task_slack_range[1], 1e-6)
    energy_norm = delta_energy / _step_energy_reference(env)
    reward = (
        config.reward_completion_weight * completion_ratio
        + config.reward_cache_hit_weight * cache_hit_ratio
        - config.reward_expired_weight * expired_ratio
        - config.reward_backlog_weight * backlog_ratio
        - config.reward_latency_weight * latency_norm
        - config.reward_energy_weight * energy_norm
        - config.reward_deadline_weight * deadline_violation_ratio
        - config.reward_reliability_weight * reliability_violation_ratio
        - config.reward_action_magnitude_weight * action_magnitude
    )
    return float(reward), {
        "step_completion_ratio": completion_ratio,
        "step_cache_hit_ratio": cache_hit_ratio,
        "step_backlog_ratio": backlog_ratio,
        "step_deadline_violation_ratio": deadline_violation_ratio,
        "step_reliability_violation_ratio": reliability_violation_ratio,
        "step_expired_ratio": expired_ratio,
        "step_latency": latency,
        "step_latency_norm": latency_norm,
        "step_energy": delta_energy,
        "step_energy_norm": energy_norm,
        "step_action_magnitude": action_magnitude,
        "step_invalid_action_reject_ratio": 0.0,
        "step_infeasible_action_reject_ratio": 0.0,
    }


def _collect_legacy_episode(
    *,
    agent: MinimalMultiAgentActorCritic,
    config: MinimalMARLConfig,
    episode_index: int,
) -> tuple[RolloutBuffer, dict[str, Any], float]:
    env = Chapter4Env(
        {
            "seed": config.seed + episode_index,
            "num_uavs": config.num_uavs,
            "assignment_rule": config.assignment_rule,
        }
    )
    reset_result = env.reset(seed=config.seed + episode_index)
    observations = [[float(value) for value in observation] for observation in reset_result["observations"]]
    buffer = RolloutBuffer()
    last_step: dict[str, Any] | None = None
    reward_breakdowns: list[dict[str, float]] = []
    while True:
        state = [value for observation in observations for value in observation]
        value = agent.value(state)
        actions, log_probs = agent.act(observations, deterministic=False)
        last_step = env.step(actions, scheduler_mode="legacy_heuristic")
        next_observations = [[float(value) for value in observation] for observation in last_step["observations"]]
        team_reward, reward_breakdown = _shape_legacy_reward(
            config=config,
            env=env,
            step_result=last_step,
            actions=actions,
        )
        reward_breakdowns.append(reward_breakdown)
        done = bool(last_step["terminated"] or last_step["truncated"])
        buffer.add(
            state=state,
            observations=observations,
            actions=actions,
            log_probs=log_probs,
            reward=team_reward,
            done=done,
            value=value,
        )
        observations = next_observations
        if done:
            break
    episode_log = env.export_episode_log(episode_index=episode_index, seed=config.seed + episode_index)
    assert last_step is not None
    return (
        buffer,
        {
            "summary": env.export_episode_summary(),
            "episode_log": episode_log,
            "last_info": last_step["info"],
            "reward_breakdowns": reward_breakdowns,
        },
        0.0,
    )


def _training_log_entry(
    *,
    episode_index: int,
    seed: int,
    batch_rewards: np.ndarray,
    summary_metrics: dict[str, float | None],
    reward_breakdowns: list[dict[str, float]],
    update_stats: dict[str, float],
) -> dict[str, Any]:
    return {
        "episode": episode_index,
        "seed": seed,
        "team_return": float(np.sum(batch_rewards)),
        "completion_rate": summary_metrics["completion_rate"],
        "average_latency": summary_metrics["average_latency"],
        "total_energy": summary_metrics["total_energy"],
        "mean_step_completion_rate": float(np.mean([item["step_completion_ratio"] for item in reward_breakdowns]) if reward_breakdowns else 0.0),
        "mean_step_cache_hit_rate": float(np.mean([item["step_cache_hit_ratio"] for item in reward_breakdowns]) if reward_breakdowns else 0.0),
        "mean_step_backlog_rate": float(np.mean([item["step_backlog_ratio"] for item in reward_breakdowns]) if reward_breakdowns else 0.0),
        "mean_step_average_latency": float(np.mean([item["step_latency"] for item in reward_breakdowns]) if reward_breakdowns else 0.0),
        "mean_step_energy": float(np.mean([item["step_energy"] for item in reward_breakdowns]) if reward_breakdowns else 0.0),
        "mean_step_invalid_action_reject_rate": float(np.mean([item["step_invalid_action_reject_ratio"] for item in reward_breakdowns]) if reward_breakdowns else 0.0),
        "mean_step_infeasible_action_reject_rate": float(np.mean([item["step_infeasible_action_reject_ratio"] for item in reward_breakdowns]) if reward_breakdowns else 0.0),
        "mean_step_deadline_violation_rate": float(np.mean([item["step_deadline_violation_ratio"] for item in reward_breakdowns]) if reward_breakdowns else 0.0),
        "mean_step_reliability_violation_rate": float(np.mean([item["step_reliability_violation_ratio"] for item in reward_breakdowns]) if reward_breakdowns else 0.0),
        **update_stats,
    }


def _build_agent(config: MinimalMARLConfig, *, probe_env: Chapter4Env, legacy_mode: bool) -> MinimalMultiAgentActorCritic:
    reset_result = probe_env.reset(seed=config.seed)
    obs_dim = len(reset_result["observations"][0])
    action_dim = len(probe_env.get_action_schema()["fields_per_agent"])
    agent_summary_dim = config.joint_agent_summary_dim if config.joint_agent_summary_dim > 0 else obs_dim
    global_summary_dim = 0 if legacy_mode else config.joint_global_summary_dim
    return MinimalMultiAgentActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_agents=config.num_uavs,
        seed=config.seed,
        action_std_init=config.action_std_init,
        action_std_min=config.action_std_min,
        action_std_decay=config.action_std_decay,
        cache_action_std_init=config.cache_action_std_init,
        cache_action_std_min=config.cache_action_std_min,
        cache_action_std_decay=config.cache_action_std_decay,
        use_movement_budget=config.use_movement_budget,
        hidden_dim=config.hidden_dim,
        max_user_blocks=probe_env.config.observation_max_users,
        user_feature_dim=5,
        task_slot_dim=config.task_slot_feature_dim,
        offloading_candidate_dim=config.offloading_candidate_feature_dim,
        cache_candidate_dim=config.cache_candidate_feature_dim,
        joint_agent_summary_dim=agent_summary_dim,
        joint_global_summary_dim=global_summary_dim,
        device=config.device,
    )


def _training_signal_payload(config: MinimalMARLConfig, *, legacy_mode: bool) -> dict[str, Any]:
    if legacy_mode:
        return {
            "type": "legacy_mobility_proxy_v1",
            "objective_scope": "mobility-only trainer retained as legacy baseline",
            "reward_completion_weight": config.reward_completion_weight,
            "reward_cache_hit_weight": config.reward_cache_hit_weight,
            "reward_latency_weight": config.reward_latency_weight,
            "reward_energy_weight": config.reward_energy_weight,
            "reward_backlog_weight": config.reward_backlog_weight,
            "reward_expired_weight": config.reward_expired_weight,
            "reward_deadline_weight": config.reward_deadline_weight,
            "reward_reliability_weight": config.reward_reliability_weight,
            "reward_action_magnitude_weight": config.reward_action_magnitude_weight,
            "reward_invalid_action_weight": 0.0,
            "reward_infeasible_action_weight": 0.0,
            "terms": [
                "step_completion_ratio",
                "step_cache_hit_ratio",
                "step_backlog_ratio",
                "step_latency_norm",
                "step_energy_norm",
                "step_deadline_violation_ratio",
                "step_reliability_violation_ratio",
            ],
        }
    return {
        "type": "hybrid_joint_system_reward_v1",
        "objective_scope": "joint reward for mobility + offloading + caching with structured action masks",
        "reward_completion_weight": config.reward_completion_weight,
        "reward_cache_hit_weight": config.reward_cache_hit_weight,
        "reward_latency_weight": config.reward_latency_weight,
        "reward_energy_weight": config.reward_energy_weight,
        "reward_backlog_weight": config.reward_backlog_weight,
        "reward_deadline_weight": config.reward_deadline_weight,
        "reward_reliability_weight": config.reward_reliability_weight,
        "aux_reward_expired_weight": config.reward_expired_weight,
        "aux_reward_action_magnitude_weight": config.reward_action_magnitude_weight,
        "reward_invalid_action_weight": config.reward_invalid_action_weight,
        "reward_infeasible_action_weight": config.reward_infeasible_action_weight,
        "terms": [
            "step_completion_ratio",
            "step_cache_hit_ratio",
            "step_latency_norm",
            "step_energy_norm",
            "step_backlog_ratio",
            "step_deadline_violation_ratio",
            "step_reliability_violation_ratio",
            "step_invalid_action_reject_ratio",
            "step_infeasible_action_reject_ratio",
        ],
        "policy_loss": {
            "type": "hybrid_clipped_ppo",
            "mobility_loss_coef": config.mobility_loss_coef,
            "offloading_loss_coef": config.offloading_loss_coef,
            "cache_loss_coef": config.cache_loss_coef,
            "mobility_entropy_coef": config.mobility_entropy_coef,
            "offloading_entropy_coef": config.offloading_entropy_coef,
            "cache_entropy_coef": config.cache_entropy_coef,
        },
    }


def run_training(config: MinimalMARLConfig) -> dict[str, Any]:
    """执行第四章训练流程；默认走 hybrid joint 主路径。"""
    legacy_mode = config.trainer_mode == "legacy_mobility_only"
    probe_env = Chapter4Env({"num_uavs": config.num_uavs, "assignment_rule": config.assignment_rule, "seed": config.seed})
    agent = _build_agent(config, probe_env=probe_env, legacy_mode=legacy_mode)
    agent.configure_optimizers(actor_lr=config.actor_lr, critic_lr=config.critic_lr)
    runtime_device = agent.runtime_device_info()

    episode_logs: list[dict[str, Any]] = []
    training_log: list[dict[str, Any]] = []
    for episode_index in range(config.train_episodes):
        if legacy_mode:
            rollout, outputs, last_value = _collect_legacy_episode(agent=agent, config=config, episode_index=episode_index)
            batch = rollout.finalize(gamma=config.gamma, gae_lambda=config.gae_lambda, last_value=last_value)
            update_stats = agent.update(batch=batch, config=config)
        else:
            rollout, outputs, last_value = _collect_hybrid_episode(agent=agent, config=config, episode_index=episode_index)
            batch = rollout.finalize(gamma=config.gamma, gae_lambda=config.gae_lambda, last_value=last_value)
            update_stats = hybrid_update(agent=agent, batch=batch, config=config)
        summary_metrics = outputs["summary"]["metrics"]
        training_log.append(
            _training_log_entry(
                episode_index=episode_index,
                seed=config.seed + episode_index,
                batch_rewards=batch.rewards,
                summary_metrics=summary_metrics,
                reward_breakdowns=outputs["reward_breakdowns"],
                update_stats=update_stats,
            )
        )
        episode_logs.append(outputs["episode_log"])

    results_dir = trainer_results_dir(config.trainer_mode)
    result_suffix = config.result_suffix()
    model_prefix = "marl_legacy_shared_ppo" if legacy_mode else "marl_hybrid_joint_ppo"
    model_path = results_dir / f"{model_prefix}_{result_suffix}.pt"
    train_log_path = results_dir / f"{model_prefix}_train_{result_suffix}.json"
    agent.save(model_path)
    payload = {
        "algorithm": "shared_ppo_centralized_critic" if legacy_mode else "hybrid_joint_mappo_ppo",
        "trainer_mode": config.trainer_mode,
        "source_inspiration": (
            "legacy mobility-only shared actor with centralized critic"
            if legacy_mode
            else "hybrid joint actor with continuous mobility, categorical offloading, and continuous cache priorities"
        ),
        "framework": {
            "numpy_version": np.__version__,
            **runtime_device,
        },
        "config": config.to_dict(),
        "training_signal": _training_signal_payload(config, legacy_mode=legacy_mode),
        "metric_schemas": probe_env.get_metric_schemas(),
        "paper_controls": {
            "use_movement_budget": config.use_movement_budget,
        },
        "credit_assignment": {
            "type": "team_advantage_broadcast",
            "description": (
                "team advantage is broadcast to each branch and each agent; hybrid PPO uses shared team reward with centralized critic"
                if not legacy_mode
                else "legacy team reward is broadcast to each mobility sample"
            ),
        },
        "interface_contract": agent.tensor_contract(),
        "action_schema": probe_env.get_action_schema(),
        "observation_schema": probe_env.get_observation_schema(),
        "uav_state_schema": probe_env.get_uav_state_schema(),
        "episode_log_schema": probe_env.get_episode_log_schema(),
        "checkpoint_path": str(model_path),
        "model_path": str(model_path),
        "training_log": training_log,
        "episode_logs": episode_logs,
    }
    write_json(train_log_path, payload)
    payload["train_log_path"] = str(train_log_path)
    return payload
