"""Smoke 与实验结果导出模块。

该模块为最小验证脚本提供统一的结果文件命名与写出逻辑，
避免不同 smoke 模式在结果目录结构和文件格式上出现不一致。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..logging_utils import write_json


def export_smoke_result(results_dir: str | Path, mode: str, payload: dict[str, Any]) -> Path:
    target = Path(results_dir) / f"smoke_{mode}.json"
    write_json(target, payload)
    return target
