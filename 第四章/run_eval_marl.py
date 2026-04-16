"""第四章 MARL 评估命令行入口模块。

该模块负责解析命令行参数并触发已训练模型的评估流程，
适用于单次 checkpoint 验证和与启发式基线的快速比较。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapter4.marl import run_marl_evaluation


def main() -> None:
    """解析命令行参数并执行 MARL 评估。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--num-uavs", type=int, default=2)
    parser.add_argument("--assignment-rule", type=str, default="nearest_uav", choices=["nearest_uav", "least_loaded_uav"])
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tag", type=str, default="default")
    args = parser.parse_args()
    result = run_marl_evaluation(
        seed=args.seed,
        eval_episodes=args.eval_episodes,
        num_uavs=args.num_uavs,
        assignment_rule=args.assignment_rule,
        model_path=args.model_path,
        overrides={"output_tag": args.tag},
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
