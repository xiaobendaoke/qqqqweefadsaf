from __future__ import annotations

import math


def success_probability(*, received_power_dbm: float, noise_power_dbm: float, snr_threshold_db: float) -> float:
    snr_avg_linear = 10.0 ** ((received_power_dbm - noise_power_dbm) / 10.0)
    snr_threshold_linear = 10.0 ** (snr_threshold_db / 10.0)
    if snr_avg_linear <= 0:
        return 0.0
    probability = math.exp(-snr_threshold_linear / snr_avg_linear)
    return min(max(probability, 0.0), 1.0)
