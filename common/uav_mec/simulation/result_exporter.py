from __future__ import annotations

from pathlib import Path
from typing import Any

from ..logging_utils import write_json


def export_smoke_result(results_dir: str | Path, mode: str, payload: dict[str, Any]) -> Path:
    target = Path(results_dir) / f"smoke_{mode}.json"
    write_json(target, payload)
    return target
