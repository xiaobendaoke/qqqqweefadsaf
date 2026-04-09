from __future__ import annotations

import math


def select_actions(observations: list[list[float]], env=None) -> list[list[float]]:
    actions: list[list[float]] = []
    associated_start = None
    if env is not None:
        schema = env.get_observation_schema()
        associated_start = int(schema["section_slices"]["associated_user_state"][0])
    for obs in observations:
        start = associated_start if associated_start is not None else max(0, len(obs) - 15)
        nearest_dx = float(obs[start]) if len(obs) > start else 0.0
        nearest_dy = float(obs[start + 1]) if len(obs) > start + 1 else 0.0
        norm = math.hypot(nearest_dx, nearest_dy)
        if norm > 1e-6:
            nearest_dx /= max(1.0, norm)
            nearest_dy /= max(1.0, norm)
        actions.append([nearest_dx, nearest_dy])
    return actions
