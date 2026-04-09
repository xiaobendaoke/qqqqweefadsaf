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
    target: Literal["local", "uav", "bs"]
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
    )

    temp_uav_queue = tdma_queue.clone()
    uav_tx_delay, uav_queue_delay, uav_success_prob = _link_terms(
        sender_position=ue.position,
        receiver_position=uav.position,
        sender_height=0.0,
        receiver_height=uav.altitude,
        payload_bits=task.input_size_bits,
        bandwidth_hz=config.bandwidth_edge_hz,
        config=config,
        tdma_queue=temp_uav_queue,
        current_time=current_time,
        queue_id=uav_queue_id,
    )
    uav_cache_hit = cache_lookup(uav, task.service_type)
    fetch_delay = 0.0
    if not uav_cache_hit:
        fetch_bits = service_catalog.get_fetch_size_bits(task.service_type)
        fetch_distance = distance_3d(uav.position, bs.position, uav.altitude, bs.height)
        fetch_power = received_power_dbm(
            tx_power_dbm=config.tx_power_dbm,
            carrier_frequency_hz=config.carrier_frequency_hz,
            distance_m=fetch_distance,
        )
        fetch_rate = shannon_rate_bps(
            bandwidth_hz=config.bandwidth_backhaul_hz,
            received_power_dbm=fetch_power,
            noise_power_dbm=config.noise_power_dbm,
        )
        fetch_delay = fetch_bits / fetch_rate
    uav_compute_delay = task.cpu_cycles / uav.compute_hz
    uav_total = uav_queue_delay + fetch_delay + uav_tx_delay + uav_compute_delay
    uav_option = OffloadingDecision(
        target="uav",
        assigned_uav_id=uav.uav_id,
        cache_hit=uav_cache_hit,
        queue_delay=uav_queue_delay,
        fetch_delay=fetch_delay,
        transmission_delay=uav_tx_delay,
        compute_delay=uav_compute_delay,
        total_latency=uav_total,
        success_probability=uav_success_prob,
        reliability_ok=uav_success_prob >= task.required_reliability,
        deadline_ok=uav_total <= task.slack,
        completed=(uav_success_prob >= task.required_reliability and uav_total <= task.slack),
        reason="uav_execution",
    )

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
    )

    feasible = [option for option in (local_option, uav_option, bs_option) if option.completed]
    decision = min(feasible, key=lambda item: item.total_latency) if feasible else min(
        (local_option, uav_option, bs_option), key=lambda item: item.total_latency
    )

    if decision.target == "uav":
        tdma_queue.schedule(current_time, decision.transmission_delay, queue_id=uav_queue_id)
    elif decision.target == "bs":
        tdma_queue.schedule(current_time, decision.transmission_delay, queue_id=bs_queue_id)
    if decision.target == "uav" and not decision.cache_hit:
        apply_service_cache_policy(uav, task.service_type)
    return decision
