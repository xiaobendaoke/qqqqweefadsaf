"""MARL 评估执行模块。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from common.uav_mec.logging_utils import write_json
from common.uav_mec.simulation.experiment_runner import compare_metric_dicts
from common.uav_mec.simulation.experiment_runner import run_short_experiment

from ..env import Chapter4Env
from ..policies.joint_heuristic_multi import POLICY_LABEL as JOINT_HEURISTIC_LABEL
from ..policies.joint_heuristic_multi import select_actions as joint_heuristic_actions
from ..policies.mobility_heuristic_multi import POLICY_LABEL as LEGACY_HEURISTIC_LABEL
from ..policies.mobility_heuristic_multi import select_actions as legacy_heuristic_actions
from ..results_paths import trainer_results_dir
from .config import MinimalMARLConfig
from .model import MinimalMultiAgentActorCritic
from .trainer import build_joint_policy_inputs
from .trainer import joint_action_payloads


def _resolve_baseline_policy(config: MinimalMARLConfig) -> tuple[str, str, Any]:
    if config.baseline_policy_id == "joint_heuristic":
        return ("joint_heuristic", JOINT_HEURISTIC_LABEL, joint_heuristic_actions)
    if config.baseline_policy_id == "legacy_mobility_only":
        return ("legacy_mobility_only", LEGACY_HEURISTIC_LABEL, legacy_heuristic_actions)
    if config.baseline_policy_id != "auto":
        raise ValueError(f"Unsupported baseline_policy_id: {config.baseline_policy_id}")
    if config.trainer_mode == "legacy_mobility_only":
        return ("legacy_mobility_only", LEGACY_HEURISTIC_LABEL, legacy_heuristic_actions)
    return ("joint_heuristic", JOINT_HEURISTIC_LABEL, joint_heuristic_actions)


def _build_eval_model(config: MinimalMARLConfig, *, probe_env: Chapter4Env, model_path: str | Path) -> MinimalMultiAgentActorCritic:
    reset_result = probe_env.reset(seed=config.seed)
    obs_dim = len(reset_result["observations"][0])
    action_dim = len(probe_env.get_action_schema()["fields_per_agent"])
    agent_summary_dim = config.joint_agent_summary_dim if config.joint_agent_summary_dim > 0 else obs_dim
    global_summary_dim = 0 if config.trainer_mode == "legacy_mobility_only" else config.joint_global_summary_dim
    model = MinimalMultiAgentActorCritic.load(model_path, seed=config.seed, device=config.device)
    assert model.obs_dim == obs_dim
    assert model.action_dim == action_dim
    assert model.num_agents == config.num_uavs
    assert model.joint_agent_summary_dim == agent_summary_dim
    assert model.joint_global_summary_dim == global_summary_dim
    return model


def _run_legacy_policy_episode(
    *,
    model: MinimalMultiAgentActorCritic,
    config: MinimalMARLConfig,
    episode_index: int,
) -> dict[str, Any]:
    env = Chapter4Env(
        {
            "seed": config.seed + episode_index,
            "num_uavs": config.num_uavs,
            "assignment_rule": config.assignment_rule,
        }
    )
    reset_result = env.reset(seed=config.seed + episode_index)
    observations = [[float(value) for value in observation] for observation in reset_result["observations"]]
    while True:
        actions, _ = model.act(observations, deterministic=True)
        step_result = env.step(actions, scheduler_mode="legacy_heuristic")
        observations = [[float(value) for value in observation] for observation in step_result["observations"]]
        if step_result["terminated"] or step_result["truncated"]:
            break
    return {
        "summary": env.export_episode_summary(),
        "episode_log": env.export_episode_log(episode_index=episode_index, seed=config.seed + episode_index),
    }


def _run_joint_policy_episode(
    *,
    model: MinimalMultiAgentActorCritic,
    config: MinimalMARLConfig,
    episode_index: int,
) -> dict[str, Any]:
    env = Chapter4Env(
        {
            "seed": config.seed + episode_index,
            "num_uavs": config.num_uavs,
            "assignment_rule": config.assignment_rule,
        }
    )
    env.reset(seed=config.seed + episode_index)
    while True:
        policy_inputs = build_joint_policy_inputs(env, config=config)
        policy_output = model.forward_joint(
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
            deterministic=True,
        )
        actions = joint_action_payloads(policy_output)
        step_result = env.step(actions, scheduler_mode="joint_action")
        if step_result["terminated"] or step_result["truncated"]:
            break
    return {
        "summary": env.export_episode_summary(),
        "episode_log": env.export_episode_log(episode_index=episode_index, seed=config.seed + episode_index),
    }


def run_evaluation(config: MinimalMARLConfig, *, model_path: str | Path) -> dict[str, Any]:
    probe_env = Chapter4Env({"num_uavs": config.num_uavs, "assignment_rule": config.assignment_rule, "seed": config.seed})
    model = _build_eval_model(config, probe_env=probe_env, model_path=model_path)
    runtime_device = model.runtime_device_info()
    baseline_policy_id, baseline_policy_label, baseline_policy_fn = _resolve_baseline_policy(config)

    marl_episode_logs: list[dict[str, Any]] = []
    marl_metrics: list[dict[str, float | None]] = []
    for episode_index in range(config.eval_episodes):
        if config.trainer_mode == "legacy_mobility_only":
            episode = _run_legacy_policy_episode(model=model, config=config, episode_index=episode_index)
        else:
            episode = _run_joint_policy_episode(model=model, config=config, episode_index=episode_index)
        marl_episode_logs.append(episode["episode_log"])
        marl_metrics.append(episode["summary"]["metrics"])

    averaged_marl_metrics: dict[str, float | None] = {}
    for key in marl_metrics[0].keys():
        values = [metric[key] for metric in marl_metrics if metric[key] is not None]
        averaged_marl_metrics[key] = float(np.mean(values)) if values else None

    baseline_result = run_short_experiment(
        env_factory=Chapter4Env,
        policy_fn=baseline_policy_fn,
        overrides={"num_uavs": config.num_uavs, "assignment_rule": config.assignment_rule},
        episodes=config.eval_episodes,
        seed=config.seed,
    )

    comparison = compare_metric_dicts(averaged_marl_metrics, baseline_result["averaged_metrics"])
    payload = {
        "algorithm": "shared_ppo_centralized_critic" if config.trainer_mode == "legacy_mobility_only" else "hybrid_joint_mappo_ppo",
        "trainer_mode": config.trainer_mode,
        "baseline_policy_id": baseline_policy_id,
        "baseline_policy_label": baseline_policy_label,
        "framework": {
            "numpy_version": np.__version__,
            **runtime_device,
        },
        "config": config.to_dict(),
        "model_path": str(model_path),
        "paper_controls": {
            "use_movement_budget": config.use_movement_budget,
        },
        "interface_contract": model.tensor_contract(),
        "action_schema": probe_env.get_action_schema(),
        "observation_schema": probe_env.get_observation_schema(),
        "uav_state_schema": probe_env.get_uav_state_schema(),
        "episode_log_schema": probe_env.get_episode_log_schema(),
        "metric_schemas": probe_env.get_metric_schemas(),
        "marl_metrics": averaged_marl_metrics,
        "baseline_metrics": baseline_result["averaged_metrics"],
        "heuristic_metrics": baseline_result["averaged_metrics"],
        "comparison": comparison,
        "marl_episode_logs": marl_episode_logs,
        "baseline_episode_logs": baseline_result["episode_logs"],
        "baseline_episode_summaries": baseline_result["episode_summaries"],
        "heuristic_episode_logs": baseline_result["episode_logs"],
        "heuristic_episode_summaries": baseline_result["episode_summaries"],
    }
    results_dir = trainer_results_dir(config.trainer_mode)
    eval_prefix = (
        f"marl_legacy_shared_ppo_eval_vs_{baseline_policy_id}"
        if config.trainer_mode == "legacy_mobility_only"
        else f"marl_hybrid_joint_eval_vs_{baseline_policy_id}"
    )
    eval_path = results_dir / f"{eval_prefix}_{config.result_suffix()}.json"
    write_json(eval_path, payload)
    payload["eval_path"] = str(eval_path)
    return payload
