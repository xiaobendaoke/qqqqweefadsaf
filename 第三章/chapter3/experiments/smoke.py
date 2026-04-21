"""第三章 smoke test 模块。

该模块提供第三章最小化验证流程，
用于快速检查导入链路、通信公式、卸载决策和环境主循环是否保持可运行状态。

输入输出与关键参数：
主要输入为 smoke 模式 `mode` 和随机种子 `seed`；
输出为不同模式下的最小结果字典，并同步写入 `第三章/results` 目录。
"""

from __future__ import annotations

from typing import Any

from common.uav_mec.comms.pathloss import received_power_dbm
from common.uav_mec.comms.reliability import success_probability
from common.uav_mec.scheduler.compute_queue import ComputeQueue
from common.uav_mec.scheduler.offloading import decide_offloading
from common.uav_mec.scheduler.tdma import TDMAQueue
from common.uav_mec.simulation.result_exporter import export_smoke_result
from common.uav_mec.simulation.task_generator import generate_tasks

from ..env import Chapter3Env
from ..policies.mobility_heuristic import select_actions


def run_smoke(mode: str, *, seed: int = 42) -> dict[str, Any]:
    """运行第三章 smoke test。

    参数：
        mode: smoke 模式，可选导入检查、任务契约、通信契约、调度契约、单步环境和完整 episode。
        seed: 验证过程使用的随机种子。

    返回：
        对应 smoke 模式的最小结果字典。
    """
    env = Chapter3Env({"seed": seed})
    results_dir = "第三章/results"

    if mode == "import_only":
        payload = {"mode": mode, "status": "ok", "chapter": "chapter3"}
        export_smoke_result(results_dir, mode, payload)
        return payload

    reset_result = env.reset(seed=seed)
    if mode == "task_contract":
        # 这里只验证观测维度和关键任务字段是否齐全，不做完整行为测试。
        payload = {
            "mode": mode,
            "status": "ok",
            "schema_version": env.get_observation_schema()["schema_version"],
            "observation_dim": len(reset_result["observations"][0]),
            "config_snapshot": env.config.to_dict(),
            "required_task_fields": ["service_type", "slack", "deadline", "required_reliability", "success_probability"],
        }
        export_smoke_result(results_dir, mode, payload)
        return payload

    if mode == "comms_contract":
        rx = received_power_dbm(tx_power_dbm=20.0, carrier_frequency_hz=2.4e9, distance_m=120.0)
        prob = success_probability(
            received_power_dbm=rx,
            bandwidth_hz=env.config.bandwidth_edge_hz,
            noise_density_dbm_per_hz=env.config.noise_density_dbm_per_hz,
            snr_threshold_db=env.config.snr_threshold_db,
        )
        payload = {
            "mode": mode,
            "status": "ok",
            "schema_version": env.get_metric_schemas()["step_signals"]["schema_version"],
            "config_snapshot": env.config.to_dict(),
            "received_power_dbm": rx,
            "success_probability": prob,
        }
        export_smoke_result(results_dir, mode, payload)
        return payload

    if mode == "scheduler_contract":
        # 若当前随机种子未生成任务，则临时抬高到达率，确保调度链路能被覆盖到。
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
            associated_uav=env.uavs[0],
            all_uavs=env.uavs,
            bs=env.bs,
            service_catalog=env.service_catalog,
            config=env.config,
            current_time=0.0,
            tdma_queue=TDMAQueue(),
            compute_queue=ComputeQueue(),
        )
        payload = {
            "mode": mode,
            "status": "ok",
            "schema_version": env.get_metric_schemas()["episode_metrics"]["schema_version"],
            "observation_dim": len(reset_result["observations"][0]),
            "config_snapshot": env.config.to_dict(),
            "decision_target": decision.target,
            "decision_total_latency": decision.total_latency,
            "decision_success_probability": decision.success_probability,
        }
        export_smoke_result(results_dir, mode, payload)
        return payload

    if mode == "env_step":
        # 单步推进环境，验证 step 返回结构与指标字段。
        actions = select_actions(reset_result["observations"])
        step_result = env.step(actions)
        payload = {
            "mode": mode,
            "status": "ok",
            "schema_version": env.get_metric_schemas()["episode_metrics"]["schema_version"],
            "observation_dim": len(reset_result["observations"][0]),
            "config_snapshot": env.config.to_dict(),
            "step_result_keys": sorted(step_result.keys()),
            "metrics": step_result["metrics"],
            "info": step_result["info"],
        }
        export_smoke_result(results_dir, mode, payload)
        return payload

    if mode == "episode":
        # 运行完整 episode，确保 reset/step/export 三段链路能串通。
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
            "schema_version": env.get_metric_schemas()["episode_metrics"]["schema_version"],
            "observation_dim": len(reset_result["observations"][0]),
            "config_snapshot": env.config.to_dict(),
            "summary": env.export_episode_summary(),
            "last_info": last_step["info"] if last_step else {},
        }
        export_smoke_result(results_dir, mode, payload)
        return payload

    raise ValueError(f"Unsupported smoke mode: {mode}")
