from __future__ import annotations

from ..entities.uav import UAVNode


def cache_lookup(uav: UAVNode, service_type: int) -> bool:
    return service_type in uav.service_cache


def apply_service_cache_policy(uav: UAVNode, service_type: int) -> None:
    if service_type in uav.service_cache:
        return
    if len(uav.service_cache) >= uav.service_cache_capacity and uav.service_cache:
        oldest = sorted(uav.service_cache)[0]
        uav.service_cache.remove(oldest)
    uav.service_cache.add(service_type)
