"""任务卸载决策模块。

该模块统一枚举本地执行、关联 UAV 执行、协同 UAV 执行和基站执行四类候选方案，
并综合传输时延、计算等待、缓存命中、能量约束和可靠性要求选择最终执行目标。

输入输出与关键参数：
输入包括任务对象、用户状态、UAV 集合、基站、服务目录、队列状态和系统配置；
输出为 `OffloadingDecision`，其中记录最终目标、代价分解和可行性判断结果。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from .tdma import BACKHAUL_QUEUE_ID
from .tdma import EDGE_ACCESS_QUEUE_ID
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
    bs_fetch_tx_energy: float = 0.0
    ue_tx_wait: float = 0.0
    ue_tx_delay: float = 0.0
    relay_wait: float = 0.0
    relay_delay: float = 0.0
    completion_time: float = 0.0
    energy_shortfall_j: float = 0.0
    deadline_overrun_s: float = 0.0
    reliability_gap: float = 0.0
    uav_tx_energy_by_id: dict[int, float] | None = None
    cache_events: list[dict[str, object]] | None = None


@dataclass(slots=True)
class OffloadingCandidate:
    """面向结构化 observation 的候选 plan 表示。"""

    candidate_id: int
    target: str
    associated_uav_id: int | None
    assigned_uav_id: int | None
    fetch_source: str | None
    tx_wait: float
    tx_delay: float
    fetch_wait: float
    fetch_delay: float
    compute_wait: float
    compute_delay: float
    total_latency_est: float
    success_prob_est: float
    energy_est: float
    cache_hit_flag: float
    deadline_margin: float
    feasible_flag: float
    energy_ok: bool
    reliability_ok: bool
    deadline_ok: bool
    completed: bool
    reason: str
    decision: OffloadingDecision

    def to_observation_dict(self) -> dict[str, object]:
        return {
            "candidate_id": int(self.candidate_id),
            "target": self.target,
            "associated_uav_id": self.associated_uav_id,
            "assigned_uav_id": self.assigned_uav_id,
            "fetch_source": self.fetch_source,
            "tx_wait": float(self.tx_wait),
            "tx_delay": float(self.tx_delay),
            "fetch_wait": float(self.fetch_wait),
            "fetch_delay": float(self.fetch_delay),
            "compute_wait": float(self.compute_wait),
            "compute_delay": float(self.compute_delay),
            "total_latency_est": float(self.total_latency_est),
            "success_prob_est": float(self.success_prob_est),
            "energy_est": float(self.energy_est),
            "cache_hit_flag": float(self.cache_hit_flag),
            "deadline_margin": float(self.deadline_margin),
            "feasible_flag": float(self.feasible_flag),
            "energy_ok": bool(self.energy_ok),
            "reliability_ok": bool(self.reliability_ok),
            "deadline_ok": bool(self.deadline_ok),
            "completed": bool(self.completed),
            "reason": self.reason,
        }


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
        noise_density_dbm_per_hz=config.noise_density_dbm_per_hz,
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
    """在共享资源类队列中预排一次串行链路传输，并返回时延、等待和能耗。"""
    resolved_tx_power_dbm = tx_power_dbm
    resolved_tx_power_w = float(tx_power_w)
    if resolved_tx_power_dbm is None:
        if resolved_tx_power_w > 0.0:
            resolved_tx_power_dbm = 10.0 * math.log10(max(resolved_tx_power_w, 1.0e-12) * 1000.0)
        else:
            resolved_tx_power_dbm = config.tx_power_dbm
    if resolved_tx_power_w <= 0.0:
        resolved_tx_power_w = 10.0 ** (resolved_tx_power_dbm / 10.0) / 1000.0
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
        bandwidth_hz=bandwidth_hz,
        noise_density_dbm_per_hz=config.noise_density_dbm_per_hz,
        snr_threshold_db=config.snr_threshold_db,
    )
    energy = float(resolved_tx_power_w * tx_delay)
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
) -> tuple[bool, float, float, str | None, float, float, float]:
    """比较本地缓存、BS 与其他 UAV 的服务获取代价，选择更优 fetch 来源。"""
    if cache_lookup(candidate_uav, task.service_type):
        return True, 0.0, 0.0, "local_cache", 0.0, 0.0, 1.0

    fetch_bits = service_catalog.get_fetch_size_bits(task.service_type)
    best = (float("inf"), float("inf"), "bs", 0.0, 0.0, 0.0)

    bs_delay, bs_wait, bs_prob, bs_fetch_energy = _schedule_serial_link(
        sender_position=bs.position,
        receiver_position=candidate_uav.position,
        sender_height=bs.height,
        receiver_height=candidate_uav.altitude,
        payload_bits=fetch_bits,
        bandwidth_hz=config.bandwidth_backhaul_hz,
        config=config,
        tdma_queue=tdma_queue.clone(),
        current_time=current_time,
        queue_id=BACKHAUL_QUEUE_ID,
        tx_power_w=config.bs_backhaul_tx_power_w,
    )
    best = (bs_wait + bs_delay, bs_wait, "bs", 0.0, bs_fetch_energy, bs_prob)

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
            queue_id=BACKHAUL_QUEUE_ID,
            tx_power_w=config.uav_tx_power_w,
        )
        candidate = (peer_wait + peer_delay, peer_wait, f"uav:{peer_uav.uav_id}", peer_energy, 0.0, peer_prob)
        if candidate[0] < best[0]:
            best = candidate
    fetch_total, fetch_wait, fetch_source, relay_energy, bs_fetch_tx_energy, fetch_probability = best
    fetch_delay = fetch_total - fetch_wait
    return False, fetch_wait, fetch_delay, fetch_source, relay_energy, bs_fetch_tx_energy, fetch_probability


def _violation_rank(option: OffloadingDecision) -> tuple[float, ...]:
    """在不存在完全可行解时，按约束违约严重度排序。"""
    return (
        1.0 if option.energy_shortfall_j > 1.0e-9 else 0.0,
        1.0 if option.reliability_gap > 1.0e-9 else 0.0,
        1.0 if option.deadline_overrun_s > 1.0e-9 else 0.0,
        float(option.energy_shortfall_j),
        float(option.reliability_gap),
        float(option.deadline_overrun_s),
        float(option.total_latency),
    )


def _decision_energy_estimate(decision: OffloadingDecision) -> float:
    return float(
        decision.ue_local_energy
        + decision.ue_uplink_energy
        + decision.uav_compute_energy
        + decision.bs_compute_energy
        + decision.relay_fetch_energy
        + decision.bs_fetch_tx_energy
    )


def select_candidate_by_plan_id(
    candidates: list[OffloadingCandidate],
    candidate_plan_id: int,
) -> OffloadingCandidate | None:
    """按 candidate_plan_id 查找策略选中的 plan。"""
    if candidate_plan_id < 0:
        return None
    for candidate in candidates:
        if candidate.candidate_id == candidate_plan_id:
            return candidate
    return None


def _decision_to_candidate(decision: OffloadingDecision, *, candidate_id: int, current_time: float, task: Task) -> OffloadingCandidate:
    feasible_flag = 1.0 if (decision.energy_ok and decision.reliability_ok and decision.deadline_ok) else 0.0
    deadline_margin = float(task.deadline - (current_time + decision.total_latency))
    return OffloadingCandidate(
        candidate_id=candidate_id,
        target=decision.target,
        associated_uav_id=decision.associated_uav_id,
        assigned_uav_id=decision.assigned_uav_id,
        fetch_source=decision.fetch_source,
        tx_wait=float(decision.queue_delay),
        tx_delay=float(decision.transmission_delay),
        fetch_wait=float(decision.fetch_wait_delay),
        fetch_delay=float(decision.fetch_delay),
        compute_wait=float(decision.compute_wait_delay),
        compute_delay=float(decision.compute_delay),
        total_latency_est=float(decision.total_latency),
        success_prob_est=float(decision.success_probability),
        energy_est=_decision_energy_estimate(decision),
        cache_hit_flag=1.0 if decision.cache_hit else 0.0,
        deadline_margin=deadline_margin,
        feasible_flag=feasible_flag,
        energy_ok=decision.energy_ok,
        reliability_ok=decision.reliability_ok,
        deadline_ok=decision.deadline_ok,
        completed=decision.completed,
        reason=decision.reason,
        decision=decision,
    )


def commit_offloading_decision(
    *,
    decision: OffloadingDecision,
    task: Task,
    current_time: float,
    tdma_queue: TDMAQueue,
    compute_queue: ComputeQueue,
) -> OffloadingDecision:
    """把已选定的卸载决策真正写入共享队列资源，并返回实际排队后的结果。"""
    if not decision.energy_ok:
        return decision
    if decision.target == "local":
        _, end_time, compute_wait = compute_queue.schedule(current_time, decision.compute_delay, queue_id=f"ue:{task.user_id}")
        total_latency = float(end_time - current_time)
        deadline_ok = bool(end_time <= task.deadline)
        return replace(
            decision,
            queue_delay=0.0,
            fetch_wait_delay=0.0,
            compute_wait_delay=float(compute_wait),
            total_latency=total_latency,
            completion_time=float(end_time),
            deadline_ok=deadline_ok,
            completed=bool(decision.energy_ok and decision.reliability_ok and deadline_ok),
            deadline_overrun_s=max(0.0, float(end_time - task.deadline)),
        )
    if decision.target in {"uav", "collaborator"} and decision.assigned_uav_id is not None:
        _, stage_end_time, ue_wait = tdma_queue.schedule(
            current_time,
            decision.ue_tx_delay,
            queue_id=EDGE_ACCESS_QUEUE_ID,
        )
        relay_wait = 0.0
        if (
            decision.target == "collaborator"
            and decision.associated_uav_id is not None
            and decision.associated_uav_id != decision.assigned_uav_id
        ):
            _, stage_end_time, relay_wait = tdma_queue.schedule(
                stage_end_time,
                decision.relay_delay,
                queue_id=BACKHAUL_QUEUE_ID,
            )
        fetch_wait = 0.0
        if decision.fetch_delay > 0.0 and decision.fetch_source is not None:
            _, stage_end_time, fetch_wait = tdma_queue.schedule(
                stage_end_time,
                decision.fetch_delay,
                queue_id=BACKHAUL_QUEUE_ID,
            )
        _, end_time, compute_wait = compute_queue.schedule(stage_end_time, decision.compute_delay, queue_id=f"uav:{decision.assigned_uav_id}")
        total_latency = float(end_time - current_time)
        deadline_ok = bool(end_time <= task.deadline)
        return replace(
            decision,
            queue_delay=float(ue_wait + relay_wait),
            fetch_wait_delay=float(fetch_wait),
            compute_wait_delay=float(compute_wait),
            total_latency=total_latency,
            completion_time=float(end_time),
            deadline_ok=deadline_ok,
            completed=bool(decision.energy_ok and decision.reliability_ok and deadline_ok),
            deadline_overrun_s=max(0.0, float(end_time - task.deadline)),
        )
    if decision.target == "bs":
        _, bs_tx_end_time, tx_wait = tdma_queue.schedule(
            current_time,
            decision.transmission_delay,
            queue_id=EDGE_ACCESS_QUEUE_ID,
        )
        _, end_time, compute_wait = compute_queue.schedule(bs_tx_end_time, decision.compute_delay, queue_id="bs")
        total_latency = float(end_time - current_time)
        deadline_ok = bool(end_time <= task.deadline)
        return replace(
            decision,
            queue_delay=float(tx_wait),
            compute_wait_delay=float(compute_wait),
            total_latency=total_latency,
            completion_time=float(end_time),
            deadline_ok=deadline_ok,
            completed=bool(decision.energy_ok and decision.reliability_ok and deadline_ok),
            deadline_overrun_s=max(0.0, float(end_time - task.deadline)),
        )
    return decision

def _enumerate_offloading_options(
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
) -> list[OffloadingDecision]:
    """枚举 local / associated UAV / collaborator UAV / BS 四类执行选项。"""
    uav_lookup = {uav.uav_id: uav for uav in all_uavs}
    local_compute_delay, local_compute_wait, local_completion_time = _schedule_compute(
        current_time=current_time,
        cpu_cycles=task.cpu_cycles,
        compute_hz=ue.compute_hz,
        queue_id=f"ue:{ue.user_id}",
        compute_queue=compute_queue.clone(),
    )
    local_compute_energy = float(task.cpu_cycles * config.ue_local_compute_energy_per_cycle)
    local_energy_ok = ue.remaining_energy_j + 1.0e-9 >= local_compute_energy
    local_deadline_overrun = max(0.0, local_completion_time - task.deadline)
    local_energy_shortfall = max(0.0, local_compute_energy - ue.remaining_energy_j)
    local_option = OffloadingDecision(
        target="local",
        associated_uav_id=associated_uav.uav_id if associated_uav is not None else None,
        assigned_uav_id=None,
        cache_hit=False,
        queue_delay=0.0,
        fetch_wait_delay=0.0,
        fetch_delay=0.0,
        transmission_delay=0.0,
        compute_wait_delay=local_compute_wait,
        compute_delay=local_compute_delay,
        total_latency=local_completion_time - current_time,
        success_probability=1.0,
        energy_ok=local_energy_ok,
        reliability_ok=True,
        deadline_ok=local_completion_time <= task.deadline,
        completed=local_energy_ok and local_completion_time <= task.deadline,
        reason="local_execution",
        ue_local_energy=local_compute_energy,
        completion_time=local_completion_time,
        energy_shortfall_j=local_energy_shortfall,
        deadline_overrun_s=local_deadline_overrun,
        reliability_gap=0.0,
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
            queue_id=EDGE_ACCESS_QUEUE_ID,
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
                queue_id=BACKHAUL_QUEUE_ID,
                tx_power_w=config.uav_tx_power_w,
            )
            total_relay_energy += relay_energy
            relay_energy_by_uav[collaborator_from.uav_id] = relay_energy_by_uav.get(collaborator_from.uav_id, 0.0) + relay_energy
            probability *= relay_prob
            tx_end_time = tx_end_time + relay_wait + relay_delay

        cache_hit, fetch_wait_delay, fetch_delay, fetch_source, fetch_energy, bs_fetch_tx_energy, fetch_probability = _estimate_fetch(
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
        ue_shortfall = max(0.0, ue_uplink_energy - ue.remaining_energy_j)
        uav_shortfall = sum(
            max(0.0, required_energy - uav_lookup[uav_id].remaining_energy_j)
            for uav_id, required_energy in uav_energy_requirements.items()
        )
        energy_ok = ue_shortfall <= 1.0e-9 and uav_shortfall <= 1.0e-9 and all(
            uav_lookup[uav_id].remaining_energy_j + 1.0e-9 >= required_energy
            for uav_id, required_energy in uav_energy_requirements.items()
        )
        probability = min(max(probability, 0.0), 1.0)
        reliability_ok = probability >= task.required_reliability
        deadline_ok = completion_time <= task.deadline
        reliability_gap = max(0.0, task.required_reliability - probability)
        deadline_overrun = max(0.0, completion_time - task.deadline)
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
            bs_fetch_tx_energy=bs_fetch_tx_energy,
            ue_tx_wait=ue_tx_wait,
            ue_tx_delay=ue_tx_delay,
            relay_wait=relay_wait,
            relay_delay=relay_delay,
            completion_time=completion_time,
            energy_shortfall_j=ue_shortfall + uav_shortfall,
            deadline_overrun_s=deadline_overrun,
            reliability_gap=reliability_gap,
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
        queue_id=EDGE_ACCESS_QUEUE_ID,
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
    bs_energy_shortfall = max(0.0, bs_uplink_energy - ue.remaining_energy_j)
    bs_reliability_gap = max(0.0, task.required_reliability - bs_prob)
    bs_deadline_overrun = max(0.0, bs_completion_time - task.deadline)
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
        energy_shortfall_j=bs_energy_shortfall,
        deadline_overrun_s=bs_deadline_overrun,
        reliability_gap=bs_reliability_gap,
    )
    options.append(bs_option)
    return options


def enumerate_offloading_candidates(
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
) -> list[OffloadingCandidate]:
    """输出结构化 observation 所需的候选卸载 plan 特征。"""
    options = _enumerate_offloading_options(
        task=task,
        ue=ue,
        associated_uav=associated_uav,
        all_uavs=all_uavs,
        bs=bs,
        service_catalog=service_catalog,
        config=config,
        current_time=current_time,
        tdma_queue=tdma_queue,
        compute_queue=compute_queue,
    )
    return [
        _decision_to_candidate(decision, candidate_id=index, current_time=current_time, task=task)
        for index, decision in enumerate(options)
    ]


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
    """legacy baseline 路径：在候选集中选择一个最终执行方案。"""
    options = _enumerate_offloading_options(
        task=task,
        ue=ue,
        associated_uav=associated_uav,
        all_uavs=all_uavs,
        bs=bs,
        service_catalog=service_catalog,
        config=config,
        current_time=current_time,
        tdma_queue=tdma_queue,
        compute_queue=compute_queue,
    )

    # 先选完全可完成的方案；若都无法满足可靠性或 deadline，则退化为能量可行且总时延最小。
    feasible = [option for option in options if option.completed]
    if feasible:
        decision = min(feasible, key=lambda item: item.total_latency)
    else:
        decision = min(options, key=_violation_rank)

    commit_offloading_decision(
        decision=decision,
        task=task,
        current_time=current_time,
        tdma_queue=tdma_queue,
        compute_queue=compute_queue,
    )
    return decision
