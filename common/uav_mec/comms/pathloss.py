"""通信几何与路径损耗计算模块。

该模块用于计算 UAV、用户和基站之间的二维/三维距离，
并基于自由空间路径损耗近似模型估计接收功率，是后续速率与可靠性计算的基础。

输入输出与关键参数：
输入通常包括发射端与接收端位置、节点高度、发射功率和载波频率；
输出为几何距离或接收功率（dBm）。
"""

from __future__ import annotations

import math
from typing import Sequence


def distance_2d(a: Sequence[float], b: Sequence[float]) -> float:
    """计算平面中两个节点之间的欧氏距离。"""
    return math.dist((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))


def distance_3d(a: Sequence[float], b: Sequence[float], height_a: float, height_b: float) -> float:
    """在考虑节点高度差的情况下计算三维欧氏距离。"""
    ground = distance_2d(a, b)
    return math.sqrt((height_a - height_b) ** 2 + ground**2)


def received_power_dbm(*, tx_power_dbm: float, carrier_frequency_hz: float, distance_m: float) -> float:
    """根据发射功率、载波频率和链路距离估计接收功率。"""
    distance_km = max(distance_m / 1000.0, 1e-6)
    frequency_mhz = carrier_frequency_hz / 1e6
    path_loss_db = 32.44 + 20.0 * math.log10(frequency_mhz) + 20.0 * math.log10(distance_km)
    return tx_power_dbm - path_loss_db
