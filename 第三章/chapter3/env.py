"""第三章单 UAV 环境模块。

该模块基于共享环境骨架构建第三章使用的单 UAV 仿真实例，
用于承接启发式策略、固定基线和 MPC shell 策略的实验运行。

输入输出与关键参数：
外部可通过 `overrides` 传入配置覆盖项；
模块会固定 `num_uavs=1` 并标记章节名为 `chapter3`，以保证结果口径与第四章区分。
"""

from __future__ import annotations

from common.uav_mec.base_env import BaseEnv
from common.uav_mec.config import build_config


class Chapter3Env(BaseEnv):
    """第三章单 UAV 环境包装，固定 `num_uavs=1` 并标记章节名。"""

    def __init__(self, overrides: dict | None = None) -> None:
        config = build_config(overrides)
        config.num_uavs = 1
        config.chapter_name = "chapter3"
        super().__init__(config)
