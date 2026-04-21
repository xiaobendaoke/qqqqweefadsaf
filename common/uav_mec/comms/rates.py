"""无线链路速率估计模块。

该模块基于 Shannon 容量公式，将带宽、接收功率和噪声功率映射为理论传输速率，
用于近似用户上行、UAV 协同转发以及服务拉取链路的吞吐能力。

输入输出与关键参数：
输入包括链路带宽 `bandwidth_hz`、接收功率 `received_power_dbm` 和噪声功率 `noise_power_dbm`；
输出为单位为 bit/s 的速率估计值。
"""

from __future__ import annotations

import math


def noise_power_dbm_from_density(*, bandwidth_hz: float, noise_density_dbm_per_hz: float) -> float:
    """将噪声功率谱密度换算到给定带宽下的总噪声功率。"""
    return float(noise_density_dbm_per_hz + 10.0 * math.log10(max(bandwidth_hz, 1.0)))


def shannon_rate_bps(
    *,
    bandwidth_hz: float,
    received_power_dbm: float,
    noise_density_dbm_per_hz: float,
) -> float:
    """根据 Shannon 公式计算链路理论速率。"""
    noise_power_dbm = noise_power_dbm_from_density(
        bandwidth_hz=bandwidth_hz,
        noise_density_dbm_per_hz=noise_density_dbm_per_hz,
    )
    snr_linear = 10.0 ** ((received_power_dbm - noise_power_dbm) / 10.0)
    return max(1.0, bandwidth_hz * math.log2(1.0 + max(snr_linear, 0.0)))
