from __future__ import annotations

import math


_PATROL_SEQUENCE = [
    [0.75, 0.0],
    [0.53, 0.53],
    [0.0, 0.75],
    [-0.53, 0.53],
    [-0.75, 0.0],
    [-0.53, -0.53],
    [0.0, -0.75],
    [0.53, -0.53],
]


def select_actions(observations: list[list[float]], env=None) -> list[list[float]]:
    if env is None:
        return [[0.0, 0.0] for _ in observations]
    phase = int(getattr(env, "current_step", 0)) % len(_PATROL_SEQUENCE)
    action = _PATROL_SEQUENCE[phase]
    normalized = _normalize(action)
    return [normalized[:] for _ in observations]


def _normalize(action: list[float]) -> list[float]:
    norm = math.hypot(action[0], action[1])
    if norm <= 1e-6:
        return [0.0, 0.0]
    return [float(action[0]) / max(1.0, norm), float(action[1]) / max(1.0, norm)]
