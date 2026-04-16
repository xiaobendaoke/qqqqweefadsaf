"""第三章 MPC shell 移动策略入口。

该模块负责根据环境观测调用轻量 MPC 优化器，为单 UAV 场景生成下一步移动动作。
它位于第三章策略层与求解器层之间，用于把统一观测接口衔接到候选动作优化过程。

核心方法说明：
采用 receding-horizon 的 MPC shell 思路，在有限候选动作集合中选择预测收益最高的动作，
兼顾用户任务积压、时延紧迫性、覆盖收益、缓存收益与移动能耗。

输入输出与关键参数：
输入为环境返回的局部观测 `observations` 和环境对象 `env`；
输出为与 UAV 数量一致的二维归一化动作列表。
预测时域长度、各项打分权重等关键参数由 `MPCShellOptimizer` 内部维护。

边界说明：
该模块本身不直接实现候选动作打分逻辑，具体优化过程位于
`chapter3/solvers/ch3_mpc_optimizer.py`。
"""

from __future__ import annotations

from ..solvers import MPCShellOptimizer


_DEFAULT_OPTIMIZER = MPCShellOptimizer()


def select_actions(observations: list[list[float]], env=None) -> list[list[float]]:
    """根据当前观测生成第三章 MPC shell 策略动作。

    参数：
        observations: 环境返回的局部观测列表，每个元素对应一个 UAV 的观测向量。
        env: 当前环境实例，用于读取状态并复用缓存的 `MPCShellOptimizer`。

    返回：
        与智能体数量一致的二维动作列表，每个动作表示归一化平面位移方向。

    说明：
        当 `env` 为空时无法访问运行时状态，因此退化为零动作；
        当 `env` 存在时，会优先复用环境上缓存的优化器对象以保持跨 step 的动作惯性信息。
    """
    if env is None:
        return [[0.0, 0.0] for _ in observations]
    optimizer = getattr(env, "_chapter3_mpc_optimizer", None)
    if optimizer is None:
        optimizer = MPCShellOptimizer()
        env._chapter3_mpc_optimizer = optimizer
    return [optimizer.plan_action(observation, env) for observation in observations]
