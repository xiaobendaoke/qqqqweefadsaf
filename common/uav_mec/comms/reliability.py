"""链路可靠性近似估计模块。

该模块根据接收功率、噪声功率与设定的 SNR 门限，
把链路质量映射为一个近似的传输成功概率，用于任务卸载时的可靠性约束判断。

输入输出与关键参数：
输入为接收功率、噪声功率和 SNR 阈值；
输出为区间 `[0, 1]` 内的成功概率估计。
"""

from __future__ import annotations

import math

from .rates import noise_power_dbm_from_density


def success_probability(
    *,
    received_power_dbm: float,
    bandwidth_hz: float,
    noise_density_dbm_per_hz: float,
    snr_threshold_db: float,
) -> float:
    noise_power_dbm = noise_power_dbm_from_density(
        bandwidth_hz=bandwidth_hz,
        noise_density_dbm_per_hz=noise_density_dbm_per_hz,
    )
    snr_avg_linear = 10.0 ** ((received_power_dbm - noise_power_dbm) / 10.0)
    snr_threshold_linear = 10.0 ** (snr_threshold_db / 10.0)
    if snr_avg_linear <= 0:
        return 0.0
    probability = math.exp(-snr_threshold_linear / snr_avg_linear)
    return min(max(probability, 0.0), 1.0)
