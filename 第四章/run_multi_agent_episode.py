"""第四章多 UAV 单回合演示命令行入口模块。

该模块用于从命令行快速运行一个多 UAV episode，
便于调试环境行为和查看单回合日志结果。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapter4.experiments import run_multi_agent_episode


def main() -> None:
    """解析命令行参数并执行单回合演示。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-uavs", type=int, default=2)
    parser.add_argument("--assignment-rule", type=str, default="nearest_uav", choices=["nearest_uav", "least_loaded_uav"])
    args = parser.parse_args()
    result = run_multi_agent_episode(seed=args.seed, num_uavs=args.num_uavs, assignment_rule=args.assignment_rule)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
