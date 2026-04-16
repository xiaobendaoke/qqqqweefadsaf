"""第四章多 UAV 环境模块。

该模块基于共享环境骨架构建第四章使用的多 UAV 仿真实例，
用于承接多 UAV 启发式实验、MARL 训练与评估流程。

输入输出与关键参数：
外部可通过 `overrides` 传入 UAV 数量、关联规则等配置覆盖项；
模块会将章节名固定为 `chapter4`，以保持日志与结果口径一致。
"""

from __future__ import annotations

from common.uav_mec.base_env import BaseEnv
from common.uav_mec.config import build_config


class Chapter4Env(BaseEnv):
    """第四章多 UAV 环境包装，仅负责章节标识与配置透传。"""

    def __init__(self, overrides: dict | None = None) -> None:
        config = build_config(overrides)
        config.chapter_name = "chapter4"
        super().__init__(config)
