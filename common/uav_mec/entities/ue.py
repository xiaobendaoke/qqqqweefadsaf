from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from ..config import SystemConfig


@dataclass(slots=True)
class UserEquipment:
    user_id: int
    position: list[float]
    compute_hz: float
    completed_tasks: int = 0
    generated_tasks: int = 0

    @classmethod
    def random_init(cls, user_id: int, config: SystemConfig, rng: random.Random) -> "UserEquipment":
        if config.fixed_user_positions and user_id < len(config.fixed_user_positions):
            fixed = config.fixed_user_positions[user_id]
            position = [float(fixed[0]), float(fixed[1])]
        else:
            position = [rng.uniform(0.0, config.area_width), rng.uniform(0.0, config.area_height)]
        return cls(
            user_id=user_id,
            position=position,
            compute_hz=config.ue_compute_hz,
        )

    def move(self, config: SystemConfig, rng: random.Random) -> None:
        delta_x = rng.uniform(-config.ue_move_distance, config.ue_move_distance)
        delta_y = rng.uniform(-config.ue_move_distance, config.ue_move_distance)
        self.position[0] = min(max(self.position[0] + delta_x, 0.0), config.area_width)
        self.position[1] = min(max(self.position[1] + delta_y, 0.0), config.area_height)
