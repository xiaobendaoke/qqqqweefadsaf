"""共享通信建模工具聚合入口。

该模块统一导出 UAV-MEC 场景下使用的链路几何、路径损耗、速率估计与可靠性计算函数，
为卸载决策、链路调度和 smoke 验证提供共同的通信层接口。

边界说明：
该模块仅负责聚合导出，不直接实现具体通信公式，
实际计算逻辑分散在 `pathloss.py`、`rates.py` 与 `reliability.py` 中。
"""

from .pathloss import distance_2d, distance_3d, received_power_dbm
from .rates import noise_power_dbm_from_density
from .rates import shannon_rate_bps
from .reliability import success_probability

__all__ = [
    "distance_2d",
    "distance_3d",
    "received_power_dbm",
    "noise_power_dbm_from_density",
    "shannon_rate_bps",
    "success_probability",
]
