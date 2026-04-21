"""边缘基站实体模块。

该模块定义基站在仿真中的最小状态表示，
包括位置、高度、算力和累计计算能耗，用于承担 BS 执行分支的任务处理。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import SystemConfig


@dataclass(slots=True)
class BaseStation:
    """表示边缘云/基站节点，仅跟踪计算资源与累计计算能耗。"""

    position: tuple[float, float]
    height: float
    compute_hz: float
    cumulative_compute_energy_j: float = 0.0
    cumulative_fetch_tx_energy_j: float = 0.0

    @classmethod
    def from_config(cls, config: SystemConfig) -> "BaseStation":
        """从系统配置构造基站实例。"""
        return cls(
            position=config.bs_position,
            height=config.bs_height,
            compute_hz=config.bs_compute_hz,
        )
