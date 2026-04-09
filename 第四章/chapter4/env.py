from __future__ import annotations

from common.uav_mec.base_env import BaseEnv
from common.uav_mec.config import build_config


class Chapter4Env(BaseEnv):
    def __init__(self, overrides: dict | None = None) -> None:
        config = build_config(overrides)
        config.chapter_name = "chapter4"
        super().__init__(config)
