from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from common.uav_mec.logging_utils import write_json

from ..env import Chapter4Env
from .buffer import RolloutBuffer
from .config import MinimalMARLConfig
from .model import MinimalMultiAgentActorCritic


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _mean_action_norm(actions: list[list[float]]) -> float:
    if not actions:
        return 0.0
    return float(sum((action[0] ** 2 + action[1] ** 2) ** 0.5 for action in actions) / len(actions))


def _max_step_move_energy(env: Chapter4Env) -> float:
    base = (
        env.config.num_uavs
        * env.config.uav_speed
        * env.config.time_slot_duration
        * env.config.uav_move_energy_per_meter
    )
    return float(max(base, 1e-6))


def _shape_team_reward(
    *,
    config: MinimalMARLConfig,
    env: Chapter4Env,
    step_result: dict[str, Any],
    previous_total_energy: float,
    actions: list[list[float]],
) -> tuple[float, dict[str, float]]:
    info = step_result["info"]
    metrics = step_result["metrics"]
    generated = max(1, int(info["num_generated_tasks"]))
    completion_rate = float(info["num_completed_tasks"]) / generated
    cache_hit_rate = float(info["num_cache_hits"]) / generated
    deadline_violation_rate = float(info["num_deadline_violations"]) / generated
    reliability_violation_rate = float(info["num_reliability_violations"]) / generated
    delta_energy = max(0.0, float(metrics["total_energy"]) - previous_total_energy)
    normalized_energy = delta_energy / _max_step_move_energy(env)
    latency = float(metrics["average_latency"])
    action_magnitude = _mean_action_norm(actions)
    reward = (
        config.reward_completion_weight * completion_rate
        + config.reward_cache_hit_weight * cache_hit_rate
        - config.reward_latency_weight * latency
        - config.reward_energy_weight * normalized_energy
        - config.reward_deadline_weight * deadline_violation_rate
        - config.reward_reliability_weight * reliability_violation_rate
        - config.reward_action_magnitude_weight * action_magnitude
    )
    return float(reward), {
        "step_completion_rate": completion_rate,
        "step_cache_hit_rate": cache_hit_rate,
        "step_deadline_violation_rate": deadline_violation_rate,
        "step_reliability_violation_rate": reliability_violation_rate,
        "step_energy": delta_energy,
        "step_energy_norm": normalized_energy,
        "step_action_magnitude": action_magnitude,
        "cumulative_average_latency": latency,
    }


def _collect_episode(
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
    previous_total_energy = 0.0
    while True:
        state = [value for observation in observations for value in observation]
        value = agent.value(state)
        actions, log_probs = agent.act(observations, deterministic=False)
        last_step = env.step(actions)
        next_observations = [[float(value) for value in observation] for observation in last_step["observations"]]
        team_reward, reward_breakdown = _shape_team_reward(
            config=config,
            env=env,
            step_result=last_step,
            previous_total_energy=previous_total_energy,
            actions=actions,
        )
        reward_breakdowns.append(reward_breakdown)
        previous_total_energy = float(last_step["metrics"]["total_energy"])
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


def run_training(config: MinimalMARLConfig) -> dict[str, Any]:
    probe_env = Chapter4Env({"num_uavs": config.num_uavs, "assignment_rule": config.assignment_rule, "seed": config.seed})
    reset_result = probe_env.reset(seed=config.seed)
    obs_dim = len(reset_result["observations"][0])
    action_dim = len(probe_env.get_action_schema()["fields_per_agent"])
    agent = MinimalMultiAgentActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_agents=config.num_uavs,
        seed=config.seed,
        action_std_init=config.action_std_init,
        action_std_min=config.action_std_min,
        action_std_decay=config.action_std_decay,
        use_movement_budget=config.use_movement_budget,
        hidden_dim=config.hidden_dim,
        device=config.device,
    )
    agent.configure_optimizers(actor_lr=config.actor_lr, critic_lr=config.critic_lr)

    episode_logs: list[dict[str, Any]] = []
    training_log: list[dict[str, Any]] = []
    for episode_index in range(config.train_episodes):
        rollout, outputs, last_value = _collect_episode(agent=agent, config=config, episode_index=episode_index)
        batch = rollout.finalize(gamma=config.gamma, gae_lambda=config.gae_lambda, last_value=last_value)
        update_stats = agent.update(batch=batch, config=config)
        summary_metrics = outputs["summary"]["metrics"]
        training_log.append(
            {
                "episode": episode_index,
                "seed": config.seed + episode_index,
                "team_return": float(np.sum(batch.rewards)),
                "completion_rate": summary_metrics["completion_rate"],
                "average_latency": summary_metrics["average_latency"],
                "total_energy": summary_metrics["total_energy"],
                "mean_step_energy": float(
                    np.mean([item["step_energy"] for item in outputs["reward_breakdowns"]])
                    if outputs["reward_breakdowns"]
                    else 0.0
                ),
                "mean_step_action_magnitude": float(
                    np.mean([item["step_action_magnitude"] for item in outputs["reward_breakdowns"]])
                    if outputs["reward_breakdowns"]
                    else 0.0
                ),
                **update_stats,
            }
        )
        episode_logs.append(outputs["episode_log"])

    result_suffix = config.result_suffix()
    model_path = RESULTS_DIR / f"marl_shared_ppo_{result_suffix}.pt"
    train_log_path = RESULTS_DIR / f"marl_train_shared_ppo_{result_suffix}.json"
    agent.save(model_path)
    payload = {
        "algorithm": "shared_ppo_centralized_critic",
        "source_inspiration": "minimal PPO/MAPPO-style shared actor with centralized critic, implemented in torch for the unified multi-UAV environment",
        "framework": {
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "device": config.device,
        },
        "config": config.to_dict(),
        "training_signal": {
            "type": "shaped_team_reward_v2",
            "terms": [
                "completion_rate",
                "cache_hit_rate",
                "average_latency",
                "delta_total_energy",
                "deadline_violation_rate",
                "reliability_violation_rate",
                "action_magnitude",
            ],
        },
        "paper_controls": {
            "use_movement_budget": config.use_movement_budget,
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
