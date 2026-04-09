from __future__ import annotations

import math


def shannon_rate_bps(*, bandwidth_hz: float, received_power_dbm: float, noise_power_dbm: float) -> float:
    snr_linear = 10.0 ** ((received_power_dbm - noise_power_dbm) / 10.0)
    return max(1.0, bandwidth_hz * math.log2(1.0 + max(snr_linear, 0.0)))
