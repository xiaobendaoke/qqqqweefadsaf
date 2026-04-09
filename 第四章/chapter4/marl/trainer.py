from __future__ import annotations

from pathlib import Path
from typing import Any

from common.uav_mec.logging_utils import write_json

from ..env import Chapter4Env
from .buffer import RolloutBuffer
from .config import MinimalMARLConfig
from .model import MinimalMultiAgentActorCritic


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


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
    while True:
        state = [value for observation in observations for value in observation]
        value = agent.value(state)
        actions, log_probs = agent.act(observations, deterministic=False)
        last_step = env.step(actions)
        next_observations = [[float(value) for value in observation] for observation in last_step["observations"]]
        team_reward = float(sum(last_step["rewards"]) / max(1, len(last_step["rewards"])))
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
    final_value = 0.0
    episode_log = env.export_episode_log(episode_index=episode_index, seed=config.seed + episode_index)
    assert last_step is not None
    return buffer, {"summary": env.export_episode_summary(), "episode_log": episode_log, "last_info": last_step["info"]}, final_value


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
    )

    episode_logs: list[dict[str, Any]] = []
    training_log: list[dict[str, Any]] = []
    for episode_index in range(config.train_episodes):
        rollout, outputs, last_value = _collect_episode(agent=agent, config=config, episode_index=episode_index)
        batch = rollout.finalize(gamma=config.gamma, gae_lambda=config.gae_lambda, last_value=last_value)
        update_stats = agent.update(
            observations=batch.flat_observations,
            actions=batch.flat_actions,
            advantages=batch.flat_advantages,
            states=batch.states,
            returns=batch.returns,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            entropy_coef=config.entropy_coef,
            gradient_clip=config.gradient_clip,
        )
        summary_metrics = outputs["summary"]["metrics"]
        training_log.append(
            {
                "episode": episode_index,
                "seed": config.seed + episode_index,
                "team_return": float(sum(batch.rewards)),
                "completion_rate": summary_metrics["completion_rate"],
                "average_latency": summary_metrics["average_latency"],
                "total_energy": summary_metrics["total_energy"],
                **update_stats,
            }
        )
        episode_logs.append(outputs["episode_log"])

    model_path = RESULTS_DIR / f"marl_shared_ac_u{config.num_uavs}_{config.assignment_rule}.json"
    train_log_path = RESULTS_DIR / f"marl_train_u{config.num_uavs}_{config.assignment_rule}.json"
    agent.save(model_path)
    payload = {
        "algorithm": "shared_centralized_actor_critic",
        "source_inspiration": "original multi-UAV MAPPO shared-actor + centralized-critic pattern, reimplemented minimally in pure Python",
        "config": config.to_dict(),
        "interface_contract": agent.tensor_contract(),
        "action_schema": probe_env.get_action_schema(),
        "observation_schema": probe_env.get_observation_schema(),
        "uav_state_schema": probe_env.get_uav_state_schema(),
        "episode_log_schema": probe_env.get_episode_log_schema(),
        "model_path": str(model_path),
        "training_log": training_log,
        "episode_logs": episode_logs,
    }
    write_json(train_log_path, payload)
    payload["train_log_path"] = str(train_log_path)
    return payload
