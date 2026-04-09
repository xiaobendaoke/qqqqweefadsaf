from __future__ import annotations

import math
from typing import Sequence


def distance_2d(a: Sequence[float], b: Sequence[float]) -> float:
    return math.dist((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))


def distance_3d(a: Sequence[float], b: Sequence[float], height_a: float, height_b: float) -> float:
    ground = distance_2d(a, b)
    return math.sqrt((height_a - height_b) ** 2 + ground**2)


def received_power_dbm(*, tx_power_dbm: float, carrier_frequency_hz: float, distance_m: float) -> float:
    distance_km = max(distance_m / 1000.0, 1e-6)
    frequency_mhz = carrier_frequency_hz / 1e6
    path_loss_db = 32.44 + 20.0 * math.log10(frequency_mhz) + 20.0 * math.log10(distance_km)
    return tx_power_dbm - path_loss_db
