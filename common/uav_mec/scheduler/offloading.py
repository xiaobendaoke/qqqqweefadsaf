from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..comms.pathloss import distance_3d, received_power_dbm
from ..comms.rates import shannon_rate_bps
from ..comms.reliability import success_probability
from ..config import SystemConfig
from ..entities import BaseStation, ServiceCatalog, UAVNode, UserEquipment
from ..task import Task
from .service_cache import apply_service_cache_policy, cache_lookup
from .tdma import TDMAQueue


@dataclass(slots=True)
class OffloadingDecision:
    target: Literal["local", "uav", "collaborator", "bs"]
    assigned_uav_id: int | None
    cache_hit: bool
    queue_delay: float
    fetch_delay: float
    transmission_delay: float
    compute_delay: float
    total_latency: float
    success_probability: float
    reliability_ok: bool
    deadline_ok: bool
    completed: bool
    reason: str
    fetch_source: str | None = None


def _link_terms(
    *,
    sender_position: tuple[float, float] | object,
    receiver_position: tuple[float, float] | object,
    sender_height: float,
    receiver_height: float,
    payload_bits: float,
    bandwidth_hz: float,
    config: SystemConfig,
    tdma_queue: TDMAQueue,
    current_time: float,
    queue_id: str,
) -> tuple[float, float, float]:
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
    tx_delay = payload_bits / rate
    _, _, queue_delay = tdma_queue.schedule(current_time, tx_delay, queue_id=queue_id)
    prob = success_probability(
        received_power_dbm=power,
        noise_power_dbm=config.noise_power_dbm,
        snr_threshold_db=config.snr_threshold_db,
    )
    return tx_delay, queue_delay, prob


def decide_offloading(
    *,
    task: Task,
    ue: UserEquipment,
    uav: UAVNode,
    all_uavs: list[UAVNode],
    bs: BaseStation,
    service_catalog: ServiceCatalog,
    config: SystemConfig,
    current_time: float,
    tdma_queue: TDMAQueue,
) -> OffloadingDecision:
    uav_queue_id = f"uav:{uav.uav_id}"
    bs_queue_id = "bs"
    local_compute_delay = task.cpu_cycles / ue.compute_hz
    local_option = OffloadingDecision(
        target="local",
        assigned_uav_id=None,
        cache_hit=True,
        queue_delay=0.0,
        fetch_delay=0.0,
        transmission_delay=0.0,
        compute_delay=local_compute_delay,
        total_latency=local_compute_delay,
        success_probability=1.0,
        reliability_ok=True,
        deadline_ok=local_compute_delay <= task.slack,
        completed=local_compute_delay <= task.slack,
        reason="local_execution",
        fetch_source=None,
    )

    def estimate_fetch(candidate_uav: UAVNode) -> tuple[bool, float, str | None]:
        if cache_lookup(candidate_uav, task.service_type):
            return True, 0.0, "local_cache"

        fetch_bits = service_catalog.get_fetch_size_bits(task.service_type)
        best_delay = None
        best_source = None

        fetch_distance_bs = distance_3d(candidate_uav.position, bs.position, candidate_uav.altitude, bs.height)
        fetch_power_bs = received_power_dbm(
            tx_power_dbm=config.tx_power_dbm,
            carrier_frequency_hz=config.carrier_frequency_hz,
            distance_m=fetch_distance_bs,
        )
        fetch_rate_bs = shannon_rate_bps(
            bandwidth_hz=config.bandwidth_backhaul_hz,
            received_power_dbm=fetch_power_bs,
            noise_power_dbm=config.noise_power_dbm,
        )
        best_delay = fetch_bits / fetch_rate_bs
        best_source = "bs"

        for peer_uav in all_uavs:
            if peer_uav.uav_id == candidate_uav.uav_id or not cache_lookup(peer_uav, task.service_type):
                continue
            fetch_distance_peer = distance_3d(
                candidate_uav.position,
                peer_uav.position,
                candidate_uav.altitude,
                peer_uav.altitude,
            )
            fetch_power_peer = received_power_dbm(
                tx_power_dbm=config.tx_power_dbm,
                carrier_frequency_hz=config.carrier_frequency_hz,
                distance_m=fetch_distance_peer,
            )
            fetch_rate_peer = shannon_rate_bps(
                bandwidth_hz=config.bandwidth_backhaul_hz,
                received_power_dbm=fetch_power_peer,
                noise_power_dbm=config.noise_power_dbm,
            )
            fetch_delay_peer = fetch_bits / fetch_rate_peer
            if best_delay is None or fetch_delay_peer < best_delay:
                best_delay = fetch_delay_peer
                best_source = f"uav:{peer_uav.uav_id}"

        return False, float(best_delay or 0.0), best_source

    def build_uav_option(candidate_uav: UAVNode, *, target: Literal["uav", "collaborator"]) -> OffloadingDecision:
        temp_queue = tdma_queue.clone()
        tx_delay, queue_delay, success_prob = _link_terms(
            sender_position=ue.position,
            receiver_position=candidate_uav.position,
            sender_height=0.0,
            receiver_height=candidate_uav.altitude,
            payload_bits=task.input_size_bits,
            bandwidth_hz=config.bandwidth_edge_hz,
            config=config,
            tdma_queue=temp_queue,
            current_time=current_time,
            queue_id=f"uav:{candidate_uav.uav_id}",
        )
        cache_hit, fetch_delay, fetch_source = estimate_fetch(candidate_uav)
        compute_delay = task.cpu_cycles / candidate_uav.compute_hz
        total_latency = queue_delay + fetch_delay + tx_delay + compute_delay
        return OffloadingDecision(
            target=target,
            assigned_uav_id=candidate_uav.uav_id,
            cache_hit=cache_hit,
            queue_delay=queue_delay,
            fetch_delay=fetch_delay,
            transmission_delay=tx_delay,
            compute_delay=compute_delay,
            total_latency=total_latency,
            success_probability=success_prob,
            reliability_ok=success_prob >= task.required_reliability,
            deadline_ok=total_latency <= task.slack,
            completed=(success_prob >= task.required_reliability and total_latency <= task.slack),
            reason=f"{target}_execution",
            fetch_source=fetch_source,
        )

    uav_option = build_uav_option(uav, target="uav")
    collaborator_options = [
        build_uav_option(candidate_uav, target="collaborator")
        for candidate_uav in all_uavs
        if candidate_uav.uav_id != uav.uav_id
    ]

    temp_bs_queue = tdma_queue.clone()
    bs_tx_delay, bs_queue_delay, bs_success_prob = _link_terms(
        sender_position=ue.position,
        receiver_position=bs.position,
        sender_height=0.0,
        receiver_height=bs.height,
        payload_bits=task.input_size_bits,
        bandwidth_hz=config.bandwidth_edge_hz,
        config=config,
        tdma_queue=temp_bs_queue,
        current_time=current_time,
        queue_id=bs_queue_id,
    )
    bs_compute_delay = task.cpu_cycles / bs.compute_hz
    bs_total = bs_queue_delay + bs_tx_delay + bs_compute_delay
    bs_option = OffloadingDecision(
        target="bs",
        assigned_uav_id=None,
        cache_hit=True,
        queue_delay=bs_queue_delay,
        fetch_delay=0.0,
        transmission_delay=bs_tx_delay,
        compute_delay=bs_compute_delay,
        total_latency=bs_total,
        success_probability=bs_success_prob,
        reliability_ok=bs_success_prob >= task.required_reliability,
        deadline_ok=bs_total <= task.slack,
        completed=(bs_success_prob >= task.required_reliability and bs_total <= task.slack),
        reason="bs_execution",
        fetch_source=None,
    )

    candidate_options = [local_option, uav_option, *collaborator_options, bs_option]
    feasible = [option for option in candidate_options if option.completed]
    decision = min(feasible, key=lambda item: item.total_latency) if feasible else min(
        candidate_options, key=lambda item: item.total_latency
    )

    if decision.target in {"uav", "collaborator"} and decision.assigned_uav_id is not None:
        tdma_queue.schedule(current_time, decision.transmission_delay, queue_id=f"uav:{decision.assigned_uav_id}")
    elif decision.target == "bs":
        tdma_queue.schedule(current_time, decision.transmission_delay, queue_id=bs_queue_id)
    if decision.target in {"uav", "collaborator"} and decision.assigned_uav_id is not None and not decision.cache_hit:
        executed_uav = next(item for item in all_uavs if item.uav_id == decision.assigned_uav_id)
        apply_service_cache_policy(executed_uav, task.service_type)
    return decision
