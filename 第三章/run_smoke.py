"""第三章 smoke test 命令行入口模块。

该模块用于从命令行触发第三章最小验证流程，
便于快速检查环境与核心依赖是否正常工作。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapter3.experiments.smoke import run_smoke


def main() -> None:
    """解析命令行参数并执行第三章 smoke test。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="episode",
        choices=["import_only", "task_contract", "comms_contract", "scheduler_contract", "env_step", "episode"],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_smoke(args.mode, seed=args.seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
