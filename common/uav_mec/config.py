from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class SystemConfig:
    seed: int = 42
    num_uavs: int = 1
    num_users: int = 6
    steps_per_episode: int = 8
    time_slot_duration: float = 1.0
    area_width: float = 400.0
    area_height: float = 400.0
    uav_altitude: float = 60.0
    uav_speed: float = 18.0
    uav_compute_hz: float = 5e9
    uav_energy_capacity_j: float = 5_000.0
    uav_move_energy_per_meter: float = 2.0
    uav_compute_energy_per_cycle: float = 1e-10
    uav_coverage_radius: float = 180.0
    uav_service_cache_capacity: int = 3
    ue_compute_hz: float = 1e9
    ue_energy_capacity_j: float = 2_500.0
    ue_local_compute_energy_per_cycle: float = 1.5e-10
    ue_uplink_power_w: float = 0.8
    ue_move_distance: float = 3.0
    task_arrival_rate: float = 0.45
    task_arrival_max_per_step: int = 3
    task_input_size_range_bits: tuple[float, float] = (1.2e6, 3.5e6)
    task_cpu_cycles_range: tuple[float, float] = (8e8, 2.5e9)
    task_slack_range: tuple[float, float] = (1.0, 3.0)
    required_reliability: float = 0.90
    num_service_types: int = 5
    service_size_bits: tuple[int, ...] = (700_000, 900_000, 1_100_000, 1_300_000, 1_500_000)
    bs_position: tuple[float, float] = (350.0, 350.0)
    bs_height: float = 30.0
    bs_compute_hz: float = 8e9
    bs_compute_energy_per_cycle: float = 6e-11
    carrier_frequency_hz: float = 2.4e9
    tx_power_dbm: float = 20.0
    noise_power_dbm: float = -90.0
    snr_threshold_db: float = 8.0
    bandwidth_edge_hz: float = 10e6
    bandwidth_backhaul_hz: float = 20e6
    relay_tx_energy_per_bit: float = 1.0e-8
    observation_max_users: int = 3
    observation_max_neighbors: int = 2
    assignment_rule: str = "nearest_uav"
    cache_ema_alpha: float = 0.65
    cache_value_floor: float = 1.0e-6
    cache_refresh_interval_steps: int = 2
    fixed_uav_positions: tuple[tuple[float, float], ...] | None = None
    fixed_user_positions: tuple[tuple[float, float], ...] | None = None
    chapter_name: str = "common"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_config(overrides: dict[str, Any] | None = None) -> SystemConfig:
    config = SystemConfig()
    if not overrides:
        return config
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise KeyError(f"Unknown config key: {key}")
        setattr(config, key, value)
    return config
