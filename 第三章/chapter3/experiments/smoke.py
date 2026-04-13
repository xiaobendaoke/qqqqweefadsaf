from __future__ import annotations

from typing import Any

from common.uav_mec.comms.pathloss import received_power_dbm
from common.uav_mec.comms.reliability import success_probability
from common.uav_mec.scheduler.offloading import decide_offloading
from common.uav_mec.scheduler.tdma import TDMAQueue
from common.uav_mec.simulation.result_exporter import export_smoke_result
from common.uav_mec.simulation.task_generator import generate_tasks

from ..env import Chapter3Env
from ..policies.mobility_heuristic import select_actions


def run_smoke(mode: str, *, seed: int = 42) -> dict[str, Any]:
    env = Chapter3Env({"seed": seed})
    results_dir = "第三章/results"

    if mode == "import_only":
        payload = {"mode": mode, "status": "ok", "chapter": "chapter3"}
        export_smoke_result(results_dir, mode, payload)
        return payload

    reset_result = env.reset(seed=seed)
    if mode == "task_contract":
        payload = {
            "mode": mode,
            "status": "ok",
            "observation_dim": len(reset_result["observations"][0]),
            "required_task_fields": ["service_type", "slack", "deadline", "required_reliability", "success_probability"],
        }
        export_smoke_result(results_dir, mode, payload)
        return payload

    if mode == "comms_contract":
        rx = received_power_dbm(tx_power_dbm=20.0, carrier_frequency_hz=2.4e9, distance_m=120.0)
        prob = success_probability(received_power_dbm=rx, noise_power_dbm=-90.0, snr_threshold_db=8.0)
        payload = {"mode": mode, "status": "ok", "received_power_dbm": rx, "success_probability": prob}
        export_smoke_result(results_dir, mode, payload)
        return payload

    if mode == "scheduler_contract":
        tasks = generate_tasks(
            users=env.users,
            current_time=0.0,
            step_index=0,
            config=env.config,
            service_catalog=env.service_catalog,
            rng=env.rng,
        )
        if not tasks:
            env.config.task_arrival_rate = 1.0
            tasks = generate_tasks(
                users=env.users,
                current_time=0.0,
                step_index=0,
                config=env.config,
                service_catalog=env.service_catalog,
                rng=env.rng,
            )
        decision = decide_offloading(
            task=tasks[0],
            ue=env.users[tasks[0].user_id],
            uav=env.uavs[0],
            all_uavs=env.uavs,
            bs=env.bs,
            service_catalog=env.service_catalog,
            config=env.config,
            current_time=0.0,
            tdma_queue=TDMAQueue(),
        )
        payload = {
            "mode": mode,
            "status": "ok",
            "decision_target": decision.target,
            "decision_total_latency": decision.total_latency,
            "decision_success_probability": decision.success_probability,
        }
        export_smoke_result(results_dir, mode, payload)
        return payload

    if mode == "env_step":
        actions = select_actions(reset_result["observations"])
        step_result = env.step(actions)
        payload = {
            "mode": mode,
            "status": "ok",
            "step_result_keys": sorted(step_result.keys()),
            "metrics": step_result["metrics"],
            "info": step_result["info"],
        }
        export_smoke_result(results_dir, mode, payload)
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
        payload = {
            "mode": mode,
            "status": "ok",
            "summary": env.export_episode_summary(),
            "last_info": last_step["info"] if last_step else {},
        }
        export_smoke_result(results_dir, mode, payload)
        return payload

    raise ValueError(f"Unsupported smoke mode: {mode}")
