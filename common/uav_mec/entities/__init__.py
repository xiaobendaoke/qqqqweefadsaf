"""场景实体聚合入口。

该模块统一导出 UAV-MEC 仿真中使用的核心实体类型，
包括基站、UAV、用户设备和服务目录，便于环境层与调度层共享引用。
"""

from .base_station import BaseStation
from .service_catalog import ServiceCatalog
from .ue import UserEquipment
from .uav import UAVNode

__all__ = ["BaseStation", "ServiceCatalog", "UserEquipment", "UAVNode"]
