"""第四章启发式实验命令行入口模块。

该模块负责解析 default、hard 和 sensitive 三类实验参数，
并调度第四章启发式实验入口。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapter4.experiments import run_experiment, run_sensitive_experiment


def main() -> None:
    """解析命令行参数并执行第四章启发式实验。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--profile", type=str, default="default", choices=["default", "hard", "sensitive"])
    parser.add_argument("--num-uavs", type=int, default=1)
    parser.add_argument("--assignment-rule", type=str, default="nearest_uav", choices=["nearest_uav", "least_loaded_uav"])
    args = parser.parse_args()
    if args.profile == "sensitive":
        result = run_sensitive_experiment(
            seed=args.seed,
            episodes=args.episodes,
            num_uavs=args.num_uavs,
            assignment_rule=args.assignment_rule,
        )
    else:
        result = run_experiment(
            seed=args.seed,
            episodes=args.episodes,
            hard=args.profile == "hard",
            num_uavs=args.num_uavs,
            assignment_rule=args.assignment_rule,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
