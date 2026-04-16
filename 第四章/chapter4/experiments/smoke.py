"""第四章 smoke test 模块。

该模块提供第四章多 UAV 环境的最小化验证流程，
用于快速检查环境重置、观测构造、单步推进和完整 episode 运行是否正常。
"""

from __future__ import annotations

from typing import Any

from common.uav_mec.simulation.result_exporter import export_smoke_result

from ..env import Chapter4Env
from ..policies.mobility_heuristic_multi import select_actions


def run_smoke(mode: str, *, seed: int = 42, num_uavs: int = 1, assignment_rule: str = "nearest_uav") -> dict[str, Any]:
    results_dir = "第四章/results"

    if mode == "import_only":
        payload = {"mode": mode, "status": "ok", "chapter": "chapter4"}
        export_smoke_result(results_dir, mode, payload)
        return payload

    env = Chapter4Env({"seed": seed, "num_uavs": num_uavs, "assignment_rule": assignment_rule})
    reset_result = env.reset(seed=seed)

    if mode == "env_step":
        actions = select_actions(reset_result["observations"])
        step_result = env.step(actions)
        payload = {
            "mode": mode,
            "status": "ok",
            "assignment_rule": assignment_rule,
            "num_uavs": num_uavs,
            "metrics": step_result["metrics"],
            "info": step_result["info"],
        }
        export_smoke_result(results_dir, f"{mode}_u{num_uavs}_{assignment_rule}_seed{seed}", payload)
        return payload

    if mode == "observation":
        payload = {
            "mode": mode,
            "status": "ok",
            "assignment_rule": assignment_rule,
            "num_uavs": num_uavs,
            "observation_schema": env.get_observation_schema(),
            "observation_length": len(reset_result["observations"][0]) if reset_result["observations"] else 0,
            "uav_states": env.get_uav_states(),
            "observation_samples": reset_result["observations"],
        }
        export_smoke_result(results_dir, f"{mode}_u{num_uavs}_{assignment_rule}_seed{seed}", payload)
        return payload

    if mode == "episode":
        observations = reset_result["observations"]
        last_step = None
        while True:
            actions = select_actions(observations)
            last_step = env.step(actions)
            observations = last_step["observations"]
            if last_step["terminated"] or last_step["truncated"]:
                break
        summary = env.export_episode_summary()
        payload = {
            "mode": mode,
            "status": "ok",
            "assignment_rule": assignment_rule,
            "num_uavs": num_uavs,
            "summary": summary,
            "last_info": last_step["info"] if last_step else {},
        }
        export_smoke_result(results_dir, f"{mode}_u{num_uavs}_{assignment_rule}_seed{seed}", payload)
        return payload

    raise ValueError(f"Unsupported smoke mode: {mode}")
