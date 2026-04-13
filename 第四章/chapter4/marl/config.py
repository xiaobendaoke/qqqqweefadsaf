from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class MinimalMARLConfig:
    seed: int = 42
    num_uavs: int = 2
    assignment_rule: str = "nearest_uav"
    output_tag: str = "default"
    train_episodes: int = 24
    eval_episodes: int = 4
    gamma: float = 0.98
    gae_lambda: float = 0.95
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    hidden_dim: int = 128
    entropy_coef: float = 0.01
    gradient_clip: float = 0.5
    action_std_init: float = 0.25
    action_std_min: float = 0.08
    action_std_decay: float = 0.995
    ppo_epochs: int = 8
    ppo_clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    value_loss_coef: float = 0.5
    minibatch_size: int = 64
    device: str = "cpu"
    reward_completion_weight: float = 1.0
    reward_cache_hit_weight: float = 0.10
    reward_latency_weight: float = 0.05
    reward_energy_weight: float = 1.20
    reward_deadline_weight: float = 0.50
    reward_reliability_weight: float = 0.50
    reward_action_magnitude_weight: float = 0.15
    use_movement_budget: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def result_suffix(self) -> str:
        base = f"u{self.num_uavs}_{self.assignment_rule}"
        if not self.output_tag or self.output_tag == "default":
            return base
        return f"{base}_{self.output_tag}"


def build_marl_config(overrides: dict[str, Any] | None = None) -> MinimalMARLConfig:
    config = MinimalMARLConfig()
    if not overrides:
        return config
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise KeyError(f"Unknown MARL config key: {key}")
        setattr(config, key, value)
    return config
