from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class MinimalMARLConfig:
    seed: int = 42
    num_uavs: int = 2
    assignment_rule: str = "nearest_uav"
    train_episodes: int = 24
    eval_episodes: int = 4
    gamma: float = 0.98
    gae_lambda: float = 0.95
    actor_lr: float = 0.01
    critic_lr: float = 0.02
    hidden_dim: int = 32
    entropy_coef: float = 0.002
    gradient_clip: float = 5.0
    action_std_init: float = 0.35
    action_std_min: float = 0.10
    action_std_decay: float = 0.995

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_marl_config(overrides: dict[str, Any] | None = None) -> MinimalMARLConfig:
    config = MinimalMARLConfig()
    if not overrides:
        return config
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise KeyError(f"Unknown MARL config key: {key}")
        setattr(config, key, value)
    return config
