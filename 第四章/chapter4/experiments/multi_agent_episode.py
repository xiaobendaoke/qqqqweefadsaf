"""第四章多 UAV 单回合演示模块。

该模块用于运行一个多 UAV episode，
并导出便于展示和调试的单回合日志结果，适合作为行为示例或快速检查工具。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common.uav_mec.logging_utils import write_json

from ..env import Chapter4Env
from ..policies.mobility_heuristic_multi import select_actions
from ..results_paths import LEGACY_RESULTS_DIR


def run_multi_agent_episode(*, seed: int, num_uavs: int, assignment_rule: str) -> dict[str, Any]:
    env = Chapter4Env({"seed": seed, "num_uavs": num_uavs, "assignment_rule": assignment_rule})
    reset_result = env.reset(seed=seed)
    observations = reset_result["observations"]
    last_step = None
    while True:
        actions = select_actions(observations, env)
        last_step = env.step(actions)
        observations = last_step["observations"]
        if last_step["terminated"] or last_step["truncated"]:
            break

    episode_log = env.export_episode_log(episode_index=0, seed=seed)
    output_path = LEGACY_RESULTS_DIR / f"multi_agent_episode_u{num_uavs}_{assignment_rule}_seed{seed}.json"
    write_json(output_path, episode_log)
    return {
        "status": "ok",
        "num_uavs": num_uavs,
        "assignment_rule": assignment_rule,
        "agent_ids": env.get_agent_ids(),
        "num_agents": env.get_num_agents(),
        "action_schema": env.get_action_schema(),
        "observation_schema": env.get_observation_schema(),
        "uav_state_schema": env.get_uav_state_schema(),
        "episode_log_schema": env.get_episode_log_schema(),
        "episode_log_path": str(output_path),
        "global_metrics": episode_log["global_metrics"],
        "per_uav_metrics": episode_log["per_uav_metrics"],
        "last_info": last_step["info"] if last_step else {},
    }
