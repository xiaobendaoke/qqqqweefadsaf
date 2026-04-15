from __future__ import annotations

from dataclasses import dataclass
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
    cache_events: list[dict[str, object]] | None = None


def _rate_for_link(
    *,
    sender_position: tuple[float, float] | list[float],
    receiver_position: tuple[float, float] | list[float],
    sender_height: float,
    receiver_height: float,
    bandwidth_hz: float,
    config: SystemConfig,
) -> tuple[float, float]:
    distance = distance_3d(sender_position, receiver_position, sender_height, receiver_height)
    power = received_power_dbm(
        tx_power_dbm=config.tx_power_dbm,
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
) -> tuple[float, float, float, float]:
    rate, power = _rate_for_link(
        sender_position=sender_position,
        receiver_position=receiver_position,
        sender_height=sender_height,
        receiver_height=receiver_height,
        bandwidth_hz=bandwidth_hz,
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
) -> tuple[bool, float, float, str | None, float]:
    if cache_lookup(candidate_uav, task.service_type):
        return True, 0.0, 0.0, "local_cache", 0.0

    fetch_bits = service_catalog.get_fetch_size_bits(task.service_type)
    best = (float("inf"), float("inf"), "bs", 0.0)

    bs_delay, bs_wait, _, bs_energy = _schedule_serial_link(
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
    best = (bs_wait + bs_delay, bs_wait, "bs", bs_energy)

    for peer_uav in all_uavs:
        if peer_uav.uav_id == candidate_uav.uav_id or not cache_lookup(peer_uav, task.service_type):
            continue
        peer_delay, peer_wait, _, peer_energy = _schedule_serial_link(
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
        candidate = (peer_wait + peer_delay, peer_wait, f"uav:{peer_uav.uav_id}", peer_energy)
        if candidate[0] < best[0]:
            best = candidate
    fetch_total, fetch_wait, fetch_source, relay_energy = best
    fetch_delay = fetch_total - fetch_wait
    return False, fetch_wait, fetch_delay, fetch_source, relay_energy


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
    local_compute_delay = float(task.cpu_cycles / max(ue.compute_hz, 1e-6))
    local_compute_energy = float(task.cpu_cycles * config.ue_local_compute_energy_per_cycle)
    local_completion_time = current_time + local_compute_delay
    local_option = OffloadingDecision(
        target="local",
        associated_uav_id=associated_uav.uav_id if associated_uav is not None else None,
        assigned_uav_id=None,
        cache_hit=True,
        queue_delay=0.0,
        fetch_wait_delay=0.0,
        fetch_delay=0.0,
        transmission_delay=0.0,
        compute_wait_delay=0.0,
        compute_delay=local_compute_delay,
        total_latency=local_compute_delay,
        success_probability=1.0,
        reliability_ok=True,
        deadline_ok=local_compute_delay <= task.slack,
        completed=local_completion_time <= task.deadline,
        reason="local_execution",
        ue_local_energy=local_compute_energy,
        completion_time=local_completion_time,
    )

    options: list[OffloadingDecision] = [local_option]

    def build_uav_option(server_uav: UAVNode, *, collaborator_from: UAVNode | None = None) -> OffloadingDecision:
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

        if collaborator_from is not None:
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
            probability = min(probability, relay_prob)
            tx_end_time = tx_end_time + relay_wait + relay_delay

        cache_hit, fetch_wait_delay, fetch_delay, fetch_source, fetch_energy = _estimate_fetch(
            candidate_uav=server_uav,
            all_uavs=all_uavs,
            bs=bs,
            service_catalog=service_catalog,
            config=config,
            tdma_queue=tdma_queue,
            current_time=tx_end_time,
            task=task,
        )
        compute_delay, compute_wait_delay, completion_time = _schedule_compute(
            current_time=tx_end_time + fetch_wait_delay + fetch_delay,
            cpu_cycles=task.cpu_cycles,
            compute_hz=server_uav.compute_hz,
            queue_id=f"uav:{server_uav.uav_id}",
            compute_queue=compute_queue.clone(),
        )
        total_latency = completion_time - current_time
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
            reliability_ok=reliability_ok,
            deadline_ok=deadline_ok,
            completed=reliability_ok and deadline_ok,
            reason="collaborative_execution" if collaborator_from is not None else "uav_execution",
            fetch_source=fetch_source,
            ue_uplink_energy=ue_uplink_energy,
            uav_compute_energy=float(task.cpu_cycles * config.uav_compute_energy_per_cycle),
            relay_fetch_energy=total_relay_energy + fetch_energy,
            ue_tx_wait=ue_tx_wait,
            ue_tx_delay=ue_tx_delay,
            relay_wait=relay_wait,
            relay_delay=relay_delay,
            completion_time=completion_time,
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
    bs_option = OffloadingDecision(
        target="bs",
        associated_uav_id=associated_uav.uav_id if associated_uav is not None else None,
        assigned_uav_id=None,
        cache_hit=True,
        queue_delay=bs_queue_delay,
        fetch_wait_delay=0.0,
        fetch_delay=0.0,
        transmission_delay=bs_tx_delay,
        compute_wait_delay=bs_compute_wait,
        compute_delay=bs_compute_delay,
        total_latency=bs_total_latency,
        success_probability=bs_prob,
        reliability_ok=bs_prob >= task.required_reliability,
        deadline_ok=bs_completion_time <= task.deadline,
        completed=(bs_prob >= task.required_reliability and bs_completion_time <= task.deadline),
        reason="bs_execution",
        ue_uplink_energy=bs_uplink_energy,
        bs_compute_energy=float(task.cpu_cycles * config.bs_compute_energy_per_cycle),
        completion_time=bs_completion_time,
    )
    options.append(bs_option)

    feasible = [option for option in options if option.completed]
    decision = min(feasible, key=lambda item: item.total_latency) if feasible else min(options, key=lambda item: item.total_latency)

    if decision.target in {"uav", "collaborator"} and decision.assigned_uav_id is not None:
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
    elif decision.target == "bs":
        _, bs_tx_end_time, _ = tdma_queue.schedule(current_time, decision.transmission_delay, queue_id="bs")
        compute_queue.schedule(bs_tx_end_time, decision.compute_delay, queue_id="bs")

    return decision
