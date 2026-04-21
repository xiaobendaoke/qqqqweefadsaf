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


def _resolve_trainer_mode(policy_family: str) -> str:
    if policy_family in {"mobility_only_rl", "legacy_mobility_only"}:
        return "legacy_mobility_only"
    if policy_family in {"joint_rl", "hybrid_joint"}:
        return "hybrid_joint"
    raise ValueError(f"Unsupported policy family: {policy_family}")


def main() -> None:
    """解析命令行参数并执行 MARL 评估。"""
    parser = argparse.ArgumentParser(
        description="Evaluate Chapter 4 RL policies against either the legacy mobility baseline or the new joint heuristic baseline."
    )
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--num-uavs", type=int, default=2)
    parser.add_argument("--assignment-rule", type=str, default="nearest_uav", choices=["nearest_uav", "least_loaded_uav"])
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument(
        "--policy-family",
        type=str,
        default="joint_rl",
        choices=["mobility_only_rl", "joint_rl", "legacy_mobility_only", "hybrid_joint"],
        help="RL checkpoint family: legacy mobility-only RL or new joint RL.",
    )
    parser.add_argument(
        "--baseline-policy",
        type=str,
        default="auto",
        choices=["auto", "legacy_mobility_only", "joint_heuristic"],
        help="Baseline used for comparison. `auto` picks a fair default from the RL family.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Torch device request: auto, cpu, cuda, or cuda:N.")
    args = parser.parse_args()
    trainer_mode = _resolve_trainer_mode(args.policy_family)
    baseline_suffix = args.baseline_policy if args.baseline_policy != "auto" else "auto_baseline"
    output_tag = args.tag if args.tag != "default" else f"{'joint_rl' if trainer_mode == 'hybrid_joint' else 'mobility_only_rl'}_{baseline_suffix}"
    result = run_marl_evaluation(
        seed=args.seed,
        eval_episodes=args.eval_episodes,
        num_uavs=args.num_uavs,
        assignment_rule=args.assignment_rule,
        model_path=args.model_path,
        overrides={
            "output_tag": output_tag,
            "device": args.device,
            "trainer_mode": trainer_mode,
            "baseline_policy_id": args.baseline_policy,
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
