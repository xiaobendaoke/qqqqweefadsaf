from __future__ import annotations


def select_actions(observations: list[list[float]], env=None) -> list[list[float]]:
    return [[0.0, 0.0] for _ in observations]
