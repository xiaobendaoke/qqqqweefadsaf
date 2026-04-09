from __future__ import annotations

from common.uav_mec.base_env import BaseEnv
from common.uav_mec.config import build_config


class Chapter3Env(BaseEnv):
    def __init__(self, overrides: dict | None = None) -> None:
        config = build_config(overrides)
        config.num_uavs = 1
        config.chapter_name = "chapter3"
        super().__init__(config)
