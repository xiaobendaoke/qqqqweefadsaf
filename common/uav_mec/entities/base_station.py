from __future__ import annotations

from dataclasses import dataclass

from ..config import SystemConfig


@dataclass(slots=True)
class BaseStation:
    position: tuple[float, float]
    height: float
    compute_hz: float
    cumulative_compute_energy_j: float = 0.0

    @classmethod
    def from_config(cls, config: SystemConfig) -> "BaseStation":
        return cls(
            position=config.bs_position,
            height=config.bs_height,
            compute_hz=config.bs_compute_hz,
        )
