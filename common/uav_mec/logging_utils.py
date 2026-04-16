"""日志与结果文件写出模块。

该模块提供实验结果落盘时使用的最小 JSON 写出工具，
用于第三章、第四章以及 smoke test 脚本统一输出结构化结果。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """将字典结果写为 UTF-8 编码、带缩进的 JSON 文件。

    参数：
        path: 输出文件路径，可以是字符串或 `Path` 对象。
        payload: 需要序列化并写入磁盘的结果字典。

    说明：
        若目标目录不存在，会自动创建父目录。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
