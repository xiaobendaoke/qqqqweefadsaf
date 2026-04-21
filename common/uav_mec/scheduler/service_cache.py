"""服务缓存管理模块。

该模块负责维护 UAV 侧服务请求热度、价值分数、机会式准入和周期性刷新策略，
用于近似建模有限缓存容量下的服务驻留与替换过程。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import SystemConfig
from ..entities import ServiceCatalog, UAVNode


@dataclass(slots=True)
class CacheEvent:
    """记录缓存保留、驱逐与准入事件，便于离线分析策略行为。"""

    uav_id: int
    action: str
    service_type: int
    value_score: float
    reason: str


def cache_lookup(uav: UAVNode, service_type: int) -> bool:
    """判断目标服务当前是否已经驻留在 UAV 本地缓存。"""
    return service_type in uav.service_cache


def record_service_request(uav: UAVNode, service_type: int, *, config: SystemConfig, service_catalog: ServiceCatalog) -> float:
    """更新服务请求统计与价值分数，为后续缓存替换提供依据。"""
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


def apply_cache_action(
    uav: UAVNode,
    service_priorities: list[float],
    *,
    config: SystemConfig,
    service_catalog: ServiceCatalog,
) -> list[CacheEvent]:
    """学习主路径：按策略给出的 priority 分数投影缓存集合。"""
    if len(service_priorities) != service_catalog.num_service_types:
        raise ValueError(
            f"Cache action length must equal num_service_types={service_catalog.num_service_types}, "
            f"got {len(service_priorities)}."
        )
    if uav.service_cache_capacity <= 0:
        events: list[CacheEvent] = []
        for service_type in sorted(uav.service_cache):
            events.append(
                CacheEvent(
                    uav_id=uav.uav_id,
                    action="evict",
                    service_type=int(service_type),
                    value_score=float(service_priorities[service_type]),
                    reason="policy_capacity_zero",
                )
            )
        uav.service_cache.clear()
        return events

    ranked_services = sorted(
        range(service_catalog.num_service_types),
        key=lambda service_type: (float(service_priorities[service_type]), -int(service_type)),
        reverse=True,
    )
    keep = {
        service_type
        for service_type in ranked_services
        if float(service_priorities[service_type]) > 0.0
    }
    if len(keep) > uav.service_cache_capacity:
        keep = set(
            sorted(
                keep,
                key=lambda service_type: (float(service_priorities[service_type]), -int(service_type)),
                reverse=True,
            )[: uav.service_cache_capacity]
        )

    events: list[CacheEvent] = []
    for service_type in sorted(uav.service_cache - keep):
        events.append(
            CacheEvent(
                uav_id=uav.uav_id,
                action="evict",
                service_type=int(service_type),
                value_score=float(service_priorities[service_type]),
                reason="policy_priority",
            )
        )
    for service_type in sorted(keep - uav.service_cache):
        events.append(
            CacheEvent(
                uav_id=uav.uav_id,
                action="admit",
                service_type=int(service_type),
                value_score=float(service_priorities[service_type]),
                reason="policy_priority",
            )
        )
    for service_type in sorted(keep & uav.service_cache):
        events.append(
            CacheEvent(
                uav_id=uav.uav_id,
                action="retain",
                service_type=int(service_type),
                value_score=float(service_priorities[service_type]),
                reason="policy_priority",
            )
        )
    uav.service_cache = keep
    return events


def apply_service_cache_policy(
    uav: UAVNode,
    service_type: int,
    *,
    config: SystemConfig,
    service_catalog: ServiceCatalog,
    opportunistic: bool = True,
) -> list[CacheEvent]:
    """在执行路径上按服务价值做机会式缓存准入与替换。"""
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
        # 驱逐优先淘汰价值分数最低、近期热度也最低的服务。
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
    """按刷新周期重排缓存内容，使热点服务在 episode 内逐步稳定下来。"""
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
