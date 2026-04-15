from __future__ import annotations

from dataclasses import dataclass

from ..config import SystemConfig
from ..entities import ServiceCatalog, UAVNode


@dataclass(slots=True)
class CacheEvent:
    uav_id: int
    action: str
    service_type: int
    value_score: float
    reason: str


def cache_lookup(uav: UAVNode, service_type: int) -> bool:
    return service_type in uav.service_cache


def record_service_request(uav: UAVNode, service_type: int, *, config: SystemConfig, service_catalog: ServiceCatalog) -> float:
    request_count = uav.cache_request_counts.get(service_type, 0) + 1
    uav.cache_request_counts[service_type] = request_count
    previous_ema = uav.cache_ema_scores.get(service_type, 0.0)
    ema = config.cache_ema_alpha * previous_ema + (1.0 - config.cache_ema_alpha) * float(request_count)
    uav.cache_ema_scores[service_type] = ema
    value = _service_value(ema=ema, request_count=request_count, service_size_bits=service_catalog.get_fetch_size_bits(service_type), config=config)
    uav.cache_value_scores[service_type] = value
    return value


def cache_score_snapshot(uav: UAVNode, *, num_service_types: int) -> list[float]:
    return [float(uav.cache_value_scores.get(service_type, 0.0)) for service_type in range(num_service_types)]


def apply_service_cache_policy(
    uav: UAVNode,
    service_type: int,
    *,
    config: SystemConfig,
    service_catalog: ServiceCatalog,
    opportunistic: bool = True,
) -> list[CacheEvent]:
    events: list[CacheEvent] = []
    value = record_service_request(uav, service_type, config=config, service_catalog=service_catalog)
    if service_type in uav.service_cache:
        events.append(
            CacheEvent(
                uav_id=uav.uav_id,
                action="retain",
                service_type=service_type,
                value_score=value,
                reason="cache_hit",
            )
        )
        return events
    if not opportunistic:
        return events

    if len(uav.service_cache) >= uav.service_cache_capacity and uav.service_cache:
        victim = min(
            uav.service_cache,
            key=lambda candidate: (
                uav.cache_value_scores.get(candidate, config.cache_value_floor),
                uav.cache_ema_scores.get(candidate, 0.0),
                candidate,
            ),
        )
        victim_value = float(uav.cache_value_scores.get(victim, config.cache_value_floor))
        if victim_value > value and len(uav.service_cache) >= uav.service_cache_capacity:
            return events
        uav.service_cache.remove(victim)
        events.append(
            CacheEvent(
                uav_id=uav.uav_id,
                action="evict",
                service_type=int(victim),
                value_score=victim_value,
                reason="low_value",
            )
        )

    uav.service_cache.add(service_type)
    events.append(
        CacheEvent(
            uav_id=uav.uav_id,
            action="admit",
            service_type=service_type,
            value_score=value,
            reason="opportunistic_admission",
        )
    )
    return events


def refresh_service_cache(
    uav: UAVNode,
    *,
    config: SystemConfig,
    service_catalog: ServiceCatalog,
) -> list[CacheEvent]:
    if uav.service_cache_capacity <= 0:
        uav.service_cache.clear()
        return []
    candidate_services = [
        service_type
        for service_type in range(service_catalog.num_service_types)
        if uav.cache_request_counts.get(service_type, 0) > 0 or service_type in uav.service_cache
    ]
    ranked_services = sorted(
        candidate_services,
        key=lambda service_type: (
            uav.cache_value_scores.get(service_type, 0.0),
            uav.cache_ema_scores.get(service_type, 0.0),
            uav.cache_request_counts.get(service_type, 0),
        ),
        reverse=True,
    )
    keep = set(ranked_services[: uav.service_cache_capacity])
    events: list[CacheEvent] = []
    for service_type in sorted(uav.service_cache - keep):
        events.append(
            CacheEvent(
                uav_id=uav.uav_id,
                action="evict",
                service_type=int(service_type),
                value_score=float(uav.cache_value_scores.get(service_type, 0.0)),
                reason="refresh",
            )
        )
    for service_type in sorted(keep - uav.service_cache):
        events.append(
            CacheEvent(
                uav_id=uav.uav_id,
                action="admit",
                service_type=int(service_type),
                value_score=float(uav.cache_value_scores.get(service_type, 0.0)),
                reason="refresh",
            )
        )
    uav.service_cache = keep
    return events


def _service_value(*, ema: float, request_count: int, service_size_bits: int, config: SystemConfig) -> float:
    size_term = max(float(service_size_bits), 1.0)
    return max(config.cache_value_floor, (ema + 0.25 * float(request_count)) / size_term)
