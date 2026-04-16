"""MARL 评估执行模块。

该模块负责加载训练好的策略模型，在统一多 UAV 环境中运行评估 episode，
并汇总评估指标、日志和与启发式基线的对比结果。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from common.uav_mec.logging_utils import write_json
from common.uav_mec.simulation.experiment_runner import compare_metric_dicts
from common.uav_mec.simulation.experiment_runner import run_short_experiment

from ..env import Chapter4Env
from ..policies.mobility_heuristic_multi import select_actions as heuristic_actions
from .config import MinimalMARLConfig
from .model import MinimalMultiAgentActorCritic


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _run_policy_episode(
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
        step_result = env.step(actions)
        observations = [[float(value) for value in observation] for observation in step_result["observations"]]
        if step_result["terminated"] or step_result["truncated"]:
            break
    return {
        "summary": env.export_episode_summary(),
        "episode_log": env.export_episode_log(episode_index=episode_index, seed=config.seed + episode_index),
    }


def run_evaluation(config: MinimalMARLConfig, *, model_path: str | Path) -> dict[str, Any]:
    probe_env = Chapter4Env({"num_uavs": config.num_uavs, "assignment_rule": config.assignment_rule, "seed": config.seed})
    reset_result = probe_env.reset(seed=config.seed)
    obs_dim = len(reset_result["observations"][0])
    action_dim = len(probe_env.get_action_schema()["fields_per_agent"])
    model = MinimalMultiAgentActorCritic.load(model_path, seed=config.seed, device=config.device)
    assert model.obs_dim == obs_dim
    assert model.action_dim == action_dim
    assert model.num_agents == config.num_uavs

    marl_episode_logs: list[dict[str, Any]] = []
    marl_metrics: list[dict[str, float | None]] = []
    for episode_index in range(config.eval_episodes):
        episode = _run_policy_episode(model=model, config=config, episode_index=episode_index)
        marl_episode_logs.append(episode["episode_log"])
        marl_metrics.append(episode["summary"]["metrics"])

    averaged_marl_metrics: dict[str, float | None] = {}
    for key in marl_metrics[0].keys():
        values = [metric[key] for metric in marl_metrics if metric[key] is not None]
        averaged_marl_metrics[key] = float(np.mean(values)) if values else None

    heuristic_result = run_short_experiment(
        env_factory=Chapter4Env,
        policy_fn=heuristic_actions,
        overrides={"num_uavs": config.num_uavs, "assignment_rule": config.assignment_rule},
        episodes=config.eval_episodes,
        seed=config.seed,
    )

    comparison = compare_metric_dicts(averaged_marl_metrics, heuristic_result["averaged_metrics"])
    payload = {
        "algorithm": "shared_ppo_centralized_critic",
        "framework": {
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "device": config.device,
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
        "marl_metrics": averaged_marl_metrics,
        "heuristic_metrics": heuristic_result["averaged_metrics"],
        "comparison": comparison,
        "marl_episode_logs": marl_episode_logs,
        "heuristic_episode_logs": heuristic_result["episode_logs"],
        "heuristic_episode_summaries": heuristic_result["episode_summaries"],
    }
    eval_path = RESULTS_DIR / f"marl_eval_shared_ppo_{config.result_suffix()}.json"
    write_json(eval_path, payload)
    payload["eval_path"] = str(eval_path)
    return payload
