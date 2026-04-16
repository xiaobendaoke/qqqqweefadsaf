"""任务卸载决策模块。

该模块统一枚举本地执行、关联 UAV 执行、协同 UAV 执行和基站执行四类候选方案，
并综合传输时延、计算等待、缓存命中、能量约束和可靠性要求选择最终执行目标。

输入输出与关键参数：
输入包括任务对象、用户状态、UAV 集合、基站、服务目录、队列状态和系统配置；
输出为 `OffloadingDecision`，其中记录最终目标、代价分解和可行性判断结果。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from ..comms.pathloss import distance_3d, received_power_dbm
from ..comms.rates import shannon_rate_bps
from ..comms.reliability import success_probability
from ..config import SystemConfig
from ..entities import BaseStation, ServiceCatalog, UAVNode, UserEquipment
from ..task import Task
from .compute_queue import ComputeQueue
from .service_cache import cache_lookup
from .tdma import TDMAQueue


@dataclass(slots=True)
class OffloadingDecision:
    """描述一次任务卸载评估的完整结果与代价分解。"""

    target: Literal["local", "uav", "collaborator", "bs"]
    associated_uav_id: int | None
    assigned_uav_id: int | None
    cache_hit: bool
    queue_delay: float
    fetch_wait_delay: float
    fetch_delay: float
    transmission_delay: float
    compute_wait_delay: float
    compute_delay: float
    total_latency: float
    success_probability: float
    energy_ok: bool
    reliability_ok: bool
    deadline_ok: bool
    completed: bool
    reason: str
    fetch_source: str | None = None
    ue_local_energy: float = 0.0
    ue_uplink_energy: float = 0.0
    uav_compute_energy: float = 0.0
    bs_compute_energy: float = 0.0
    relay_fetch_energy: float = 0.0
    ue_tx_wait: float = 0.0
    ue_tx_delay: float = 0.0
    relay_wait: float = 0.0
    relay_delay: float = 0.0
    completion_time: float = 0.0
    uav_tx_energy_by_id: dict[int, float] | None = None
    cache_events: list[dict[str, object]] | None = None


def _rate_for_link(
    *,
    sender_position: tuple[float, float] | list[float],
    receiver_position: tuple[float, float] | list[float],
    sender_height: float,
    receiver_height: float,
    bandwidth_hz: float,
    tx_power_dbm: float,
    config: SystemConfig,
) -> tuple[float, float]:
    """估算链路速率与接收功率，为传输时延/可靠性提供基础。"""
    distance = distance_3d(sender_position, receiver_position, sender_height, receiver_height)
    power = received_power_dbm(
        tx_power_dbm=tx_power_dbm,
        carrier_frequency_hz=config.carrier_frequency_hz,
        distance_m=distance,
    )
    rate = shannon_rate_bps(
        bandwidth_hz=bandwidth_hz,
        received_power_dbm=power,
        noise_power_dbm=config.noise_power_dbm,
    )
    return rate, power


def _schedule_serial_link(
    *,
    sender_position: tuple[float, float] | list[float],
    receiver_position: tuple[float, float] | list[float],
    sender_height: float,
    receiver_height: float,
    payload_bits: float,
    bandwidth_hz: float,
    config: SystemConfig,
    tdma_queue: TDMAQueue,
    current_time: float,
    queue_id: str,
    tx_power_w: float = 0.0,
    tx_power_dbm: float | None = None,
) -> tuple[float, float, float, float]:
    """在 TDMA 队列中预排一次串行链路传输，并返回时延、等待和能耗。"""
    resolved_tx_power_dbm = tx_power_dbm
    if resolved_tx_power_dbm is None:
        if tx_power_w > 0.0:
            resolved_tx_power_dbm = 10.0 * math.log10(max(tx_power_w, 1.0e-12) * 1000.0)
        else:
            resolved_tx_power_dbm = config.tx_power_dbm
    rate, power = _rate_for_link(
        sender_position=sender_position,
        receiver_position=receiver_position,
        sender_height=sender_height,
        receiver_height=receiver_height,
        bandwidth_hz=bandwidth_hz,
        tx_power_dbm=resolved_tx_power_dbm,
        config=config,
    )
    tx_delay = float(payload_bits / max(rate, 1e-6))
    _, _, wait_delay = tdma_queue.schedule(current_time, tx_delay, queue_id=queue_id)
    probability = success_probability(
        received_power_dbm=power,
        noise_power_dbm=config.noise_power_dbm,
        snr_threshold_db=config.snr_threshold_db,
    )
    energy = float(tx_power_w * tx_delay) if tx_power_w > 0.0 else float(payload_bits * config.relay_tx_energy_per_bit)
    return tx_delay, wait_delay, probability, energy


def _schedule_compute(
    *,
    current_time: float,
    cpu_cycles: float,
    compute_hz: float,
    queue_id: str,
    compute_queue: ComputeQueue,
) -> tuple[float, float, float]:
    """在计算队列中预排任务执行。"""
    compute_delay = float(cpu_cycles / max(compute_hz, 1e-6))
    _, end_time, wait_delay = compute_queue.schedule(current_time, compute_delay, queue_id=queue_id)
    return compute_delay, wait_delay, end_time


def _estimate_fetch(
    *,
    candidate_uav: UAVNode,
    all_uavs: list[UAVNode],
    bs: BaseStation,
    service_catalog: ServiceCatalog,
    config: SystemConfig,
    tdma_queue: TDMAQueue,
    current_time: float,
    task: Task,
) -> tuple[bool, float, float, str | None, float, float]:
    """比较本地缓存、BS 与其他 UAV 的服务获取代价，选择更优 fetch 来源。"""
    if cache_lookup(candidate_uav, task.service_type):
        return True, 0.0, 0.0, "local_cache", 0.0, 1.0

    fetch_bits = service_catalog.get_fetch_size_bits(task.service_type)
    best = (float("inf"), float("inf"), "bs", 0.0, 0.0)

    bs_delay, bs_wait, bs_prob, _ = _schedule_serial_link(
        sender_position=bs.position,
        receiver_position=candidate_uav.position,
        sender_height=bs.height,
        receiver_height=candidate_uav.altitude,
        payload_bits=fetch_bits,
        bandwidth_hz=config.bandwidth_backhaul_hz,
        config=config,
        tdma_queue=tdma_queue.clone(),
        current_time=current_time,
        queue_id=f"fetch:bs->uav:{candidate_uav.uav_id}",
    )
    # The current metric contract tracks only UAV-originated relay / peer-fetch
    # transmission energy. BS-originated fetch latency still counts, but its
    # transmission energy is not folded into relay_fetch_energy.
    best = (bs_wait + bs_delay, bs_wait, "bs", 0.0, bs_prob)

    for peer_uav in all_uavs:
        if peer_uav.uav_id == candidate_uav.uav_id or not cache_lookup(peer_uav, task.service_type):
            continue
        peer_delay, peer_wait, peer_prob, peer_energy = _schedule_serial_link(
            sender_position=peer_uav.position,
            receiver_position=candidate_uav.position,
            sender_height=peer_uav.altitude,
            receiver_height=candidate_uav.altitude,
            payload_bits=fetch_bits,
            bandwidth_hz=config.bandwidth_backhaul_hz,
            config=config,
            tdma_queue=tdma_queue.clone(),
            current_time=current_time,
            queue_id=f"fetch:uav:{peer_uav.uav_id}->uav:{candidate_uav.uav_id}",
        )
        candidate = (peer_wait + peer_delay, peer_wait, f"uav:{peer_uav.uav_id}", peer_energy, peer_prob)
        if candidate[0] < best[0]:
            best = candidate
    fetch_total, fetch_wait, fetch_source, relay_energy, fetch_probability = best
    fetch_delay = fetch_total - fetch_wait
    return False, fetch_wait, fetch_delay, fetch_source, relay_energy, fetch_probability


def decide_offloading(
    *,
    task: Task,
    ue: UserEquipment,
    associated_uav: UAVNode | None,
    all_uavs: list[UAVNode],
    bs: BaseStation,
    service_catalog: ServiceCatalog,
    config: SystemConfig,
    current_time: float,
    tdma_queue: TDMAQueue,
    compute_queue: ComputeQueue,
) -> OffloadingDecision:
    """枚举 local / associated UAV / collaborator UAV / BS 四类执行选项并择优。"""
    uav_lookup = {uav.uav_id: uav for uav in all_uavs}
    local_compute_delay = float(task.cpu_cycles / max(ue.compute_hz, 1e-6))
    local_compute_energy = float(task.cpu_cycles * config.ue_local_compute_energy_per_cycle)
    local_completion_time = current_time + local_compute_delay
    local_energy_ok = ue.remaining_energy_j + 1.0e-9 >= local_compute_energy
    local_option = OffloadingDecision(
        target="local",
        associated_uav_id=associated_uav.uav_id if associated_uav is not None else None,
        assigned_uav_id=None,
        cache_hit=False,
        queue_delay=0.0,
        fetch_wait_delay=0.0,
        fetch_delay=0.0,
        transmission_delay=0.0,
        compute_wait_delay=0.0,
        compute_delay=local_compute_delay,
        total_latency=local_compute_delay,
        success_probability=1.0,
        energy_ok=local_energy_ok,
        reliability_ok=True,
        deadline_ok=local_compute_delay <= task.slack,
        completed=local_energy_ok and local_completion_time <= task.deadline,
        reason="local_execution",
        ue_local_energy=local_compute_energy,
        completion_time=local_completion_time,
    )

    options: list[OffloadingDecision] = [local_option]

    def build_uav_option(server_uav: UAVNode, *, collaborator_from: UAVNode | None = None) -> OffloadingDecision:
        """构造由某架 UAV 执行的候选方案，必要时包含协同中继。"""
        ue_tx_delay, ue_tx_wait, tx_prob, ue_uplink_energy = _schedule_serial_link(
            sender_position=ue.position,
            receiver_position=associated_uav.position if associated_uav is not None else server_uav.position,
            sender_height=0.0,
            receiver_height=associated_uav.altitude if associated_uav is not None else server_uav.altitude,
            payload_bits=task.input_size_bits,
            bandwidth_hz=config.bandwidth_edge_hz,
            config=config,
            tdma_queue=tdma_queue.clone(),
            current_time=current_time,
            queue_id=f"uav:{(associated_uav or server_uav).uav_id}",
            tx_power_w=config.ue_uplink_power_w,
        )
        relay_delay = 0.0
        relay_wait = 0.0
        total_relay_energy = 0.0
        probability = tx_prob
        tx_end_time = current_time + ue_tx_wait + ue_tx_delay
        relay_energy_by_uav: dict[int, float] = {}

        if collaborator_from is not None:
            # 当关联 UAV 与执行 UAV 不同，先把用户任务中继到目标 UAV。
            relay_delay, relay_wait, relay_prob, relay_energy = _schedule_serial_link(
                sender_position=collaborator_from.position,
                receiver_position=server_uav.position,
                sender_height=collaborator_from.altitude,
                receiver_height=server_uav.altitude,
                payload_bits=task.input_size_bits,
                bandwidth_hz=config.bandwidth_backhaul_hz,
                config=config,
                tdma_queue=tdma_queue.clone(),
                current_time=tx_end_time,
                queue_id=f"relay:uav:{collaborator_from.uav_id}->uav:{server_uav.uav_id}",
            )
            total_relay_energy += relay_energy
            relay_energy_by_uav[collaborator_from.uav_id] = relay_energy_by_uav.get(collaborator_from.uav_id, 0.0) + relay_energy
            probability *= relay_prob
            tx_end_time = tx_end_time + relay_wait + relay_delay

        cache_hit, fetch_wait_delay, fetch_delay, fetch_source, fetch_energy, fetch_probability = _estimate_fetch(
            candidate_uav=server_uav,
            all_uavs=all_uavs,
            bs=bs,
            service_catalog=service_catalog,
            config=config,
            tdma_queue=tdma_queue,
            current_time=tx_end_time,
            task=task,
        )
        probability *= fetch_probability
        if fetch_source is not None and fetch_source.startswith("uav:"):
            peer_uav_id = int(fetch_source.split(":", 1)[1])
            relay_energy_by_uav[peer_uav_id] = relay_energy_by_uav.get(peer_uav_id, 0.0) + fetch_energy
        compute_delay, compute_wait_delay, completion_time = _schedule_compute(
            current_time=tx_end_time + fetch_wait_delay + fetch_delay,
            cpu_cycles=task.cpu_cycles,
            compute_hz=server_uav.compute_hz,
            queue_id=f"uav:{server_uav.uav_id}",
            compute_queue=compute_queue.clone(),
        )
        total_latency = completion_time - current_time
        uav_energy_requirements: dict[int, float] = {server_uav.uav_id: float(task.cpu_cycles * config.uav_compute_energy_per_cycle)}
        for source_uav_id, relay_energy in relay_energy_by_uav.items():
            uav_energy_requirements[source_uav_id] = uav_energy_requirements.get(source_uav_id, 0.0) + float(relay_energy)
        energy_ok = ue.remaining_energy_j + 1.0e-9 >= ue_uplink_energy and all(
            uav_lookup[uav_id].remaining_energy_j + 1.0e-9 >= required_energy
            for uav_id, required_energy in uav_energy_requirements.items()
        )
        probability = min(max(probability, 0.0), 1.0)
        reliability_ok = probability >= task.required_reliability
        deadline_ok = completion_time <= task.deadline
        return OffloadingDecision(
            target="collaborator" if collaborator_from is not None else "uav",
            associated_uav_id=associated_uav.uav_id if associated_uav is not None else server_uav.uav_id,
            assigned_uav_id=server_uav.uav_id,
            cache_hit=cache_hit,
            queue_delay=ue_tx_wait + relay_wait,
            fetch_wait_delay=fetch_wait_delay,
            fetch_delay=fetch_delay,
            transmission_delay=ue_tx_delay + relay_delay,
            compute_wait_delay=compute_wait_delay,
            compute_delay=compute_delay,
            total_latency=total_latency,
            success_probability=probability,
            energy_ok=energy_ok,
            reliability_ok=reliability_ok,
            deadline_ok=deadline_ok,
            completed=energy_ok and reliability_ok and deadline_ok,
            reason="collaborative_execution" if collaborator_from is not None else "uav_execution",
            fetch_source=fetch_source,
            ue_uplink_energy=ue_uplink_energy,
            uav_compute_energy=uav_energy_requirements[server_uav.uav_id],
            relay_fetch_energy=total_relay_energy + fetch_energy,
            ue_tx_wait=ue_tx_wait,
            ue_tx_delay=ue_tx_delay,
            relay_wait=relay_wait,
            relay_delay=relay_delay,
            completion_time=completion_time,
            uav_tx_energy_by_id=relay_energy_by_uav,
            cache_events=[],
        )

    if associated_uav is not None:
        options.append(build_uav_option(associated_uav))
        for collaborator_uav in all_uavs:
            if collaborator_uav.uav_id == associated_uav.uav_id:
                continue
            options.append(build_uav_option(collaborator_uav, collaborator_from=associated_uav))

    bs_tx_delay, bs_queue_delay, bs_prob, bs_uplink_energy = _schedule_serial_link(
        sender_position=ue.position,
        receiver_position=bs.position,
        sender_height=0.0,
        receiver_height=bs.height,
        payload_bits=task.input_size_bits,
        bandwidth_hz=config.bandwidth_edge_hz,
        config=config,
        tdma_queue=tdma_queue.clone(),
        current_time=current_time,
        queue_id="bs",
        tx_power_w=config.ue_uplink_power_w,
    )
    bs_compute_delay, bs_compute_wait, bs_completion_time = _schedule_compute(
        current_time=current_time + bs_queue_delay + bs_tx_delay,
        cpu_cycles=task.cpu_cycles,
        compute_hz=bs.compute_hz,
        queue_id="bs",
        compute_queue=compute_queue.clone(),
    )
    bs_total_latency = bs_completion_time - current_time
    bs_energy_ok = ue.remaining_energy_j + 1.0e-9 >= bs_uplink_energy
    bs_option = OffloadingDecision(
        target="bs",
        associated_uav_id=associated_uav.uav_id if associated_uav is not None else None,
        assigned_uav_id=None,
        cache_hit=False,
        queue_delay=bs_queue_delay,
        fetch_wait_delay=0.0,
        fetch_delay=0.0,
        transmission_delay=bs_tx_delay,
        compute_wait_delay=bs_compute_wait,
        compute_delay=bs_compute_delay,
        total_latency=bs_total_latency,
        success_probability=bs_prob,
        energy_ok=bs_energy_ok,
        reliability_ok=bs_prob >= task.required_reliability,
        deadline_ok=bs_completion_time <= task.deadline,
        completed=(bs_energy_ok and bs_prob >= task.required_reliability and bs_completion_time <= task.deadline),
        reason="bs_execution",
        ue_uplink_energy=bs_uplink_energy,
        bs_compute_energy=float(task.cpu_cycles * config.bs_compute_energy_per_cycle),
        completion_time=bs_completion_time,
    )
    options.append(bs_option)

    # 先选完全可完成的方案；若都无法满足可靠性或 deadline，则退化为能量可行且总时延最小。
    feasible = [option for option in options if option.completed]
    energy_feasible = [option for option in options if option.energy_ok]
    if feasible:
        decision = min(feasible, key=lambda item: item.total_latency)
    elif energy_feasible:
        decision = min(energy_feasible, key=lambda item: item.total_latency)
    else:
        return OffloadingDecision(
            target="local",
            associated_uav_id=associated_uav.uav_id if associated_uav is not None else None,
            assigned_uav_id=None,
            cache_hit=False,
            queue_delay=0.0,
            fetch_wait_delay=0.0,
            fetch_delay=0.0,
            transmission_delay=0.0,
            compute_wait_delay=0.0,
            compute_delay=0.0,
            total_latency=0.0,
            success_probability=0.0,
            energy_ok=False,
            reliability_ok=False,
            deadline_ok=False,
            completed=False,
            reason="insufficient_energy",
            completion_time=float("inf"),
        )

    if decision.energy_ok and decision.target in {"uav", "collaborator"} and decision.assigned_uav_id is not None:
        primary_uav_id = decision.associated_uav_id if decision.associated_uav_id is not None else decision.assigned_uav_id
        _, stage_end_time, _ = tdma_queue.schedule(
            current_time,
            decision.ue_tx_delay,
            queue_id=f"uav:{primary_uav_id}",
        )
        if decision.target == "collaborator" and decision.associated_uav_id is not None and decision.associated_uav_id != decision.assigned_uav_id:
            _, stage_end_time, _ = tdma_queue.schedule(
                stage_end_time,
                decision.relay_delay,
                queue_id=f"relay:uav:{decision.associated_uav_id}->uav:{decision.assigned_uav_id}",
            )
        if decision.fetch_delay > 0.0 and decision.fetch_source is not None:
            # 只有最终被采纳的 fetch 链路才会真正占用公共传输队列。
            if decision.fetch_source == "bs":
                _, stage_end_time, _ = tdma_queue.schedule(
                    stage_end_time,
                    decision.fetch_delay,
                    queue_id=f"fetch:bs->uav:{decision.assigned_uav_id}",
                )
            elif decision.fetch_source.startswith("uav:"):
                peer_id = decision.fetch_source.split(":", 1)[1]
                _, stage_end_time, _ = tdma_queue.schedule(
                    stage_end_time,
                    decision.fetch_delay,
                    queue_id=f"fetch:uav:{peer_id}->uav:{decision.assigned_uav_id}",
                )
        compute_queue.schedule(stage_end_time, decision.compute_delay, queue_id=f"uav:{decision.assigned_uav_id}")
    elif decision.energy_ok and decision.target == "bs":
        _, bs_tx_end_time, _ = tdma_queue.schedule(current_time, decision.transmission_delay, queue_id="bs")
        compute_queue.schedule(bs_tx_end_time, decision.compute_delay, queue_id="bs")

    return decision
