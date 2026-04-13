from __future__ import annotations

from ..solvers import MPCShellOptimizer


_DEFAULT_OPTIMIZER = MPCShellOptimizer()


def select_actions(observations: list[list[float]], env=None) -> list[list[float]]:
    if env is None:
        return [[0.0, 0.0] for _ in observations]
    optimizer = getattr(env, "_chapter3_mpc_optimizer", None)
    if optimizer is None:
        optimizer = MPCShellOptimizer()
        env._chapter3_mpc_optimizer = optimizer
    return [optimizer.plan_action(observation, env) for observation in observations]
