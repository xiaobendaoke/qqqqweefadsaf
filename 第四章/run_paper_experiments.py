"""第四章 stage-5 论文实验命令行入口模块。

该模块负责从命令行触发论文阶段的实验矩阵、调参和图表生成流程。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapter4.marl import run_paper_experiments
from chapter4.experiments import recommended_experiment_matrix


def main() -> None:
    """解析命令行参数并执行 stage-5 论文实验。"""
    parser = argparse.ArgumentParser(
        description="Run the Chapter 4 paper pipeline, or print the recommended fair-comparison matrix for mobility-only RL, joint heuristic, and joint RL."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=142)
    parser.add_argument("--tuning-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--tuning-eval-offset", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--train-episode-scale", type=float, default=1.0)
    parser.add_argument("--selected-candidate", type=str, default=None, help="Optional explicit tuning candidate label override.")
    parser.add_argument(
        "--print-matrix",
        action="store_true",
        help="Print the recommended fair-comparison experiment matrix and exit.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Torch device request: auto, cpu, cuda, or cuda:N.")
    args = parser.parse_args()
    if args.print_matrix:
        print(json.dumps(recommended_experiment_matrix(), indent=2, ensure_ascii=False))
        return
    result = run_paper_experiments(
        seed=args.seed,
        eval_seed=args.eval_seed,
        tuning_seeds=args.tuning_seeds,
        tuning_eval_offset=args.tuning_eval_offset,
        eval_episodes=args.eval_episodes,
        train_episode_scale=args.train_episode_scale,
        selected_candidate_name=args.selected_candidate,
        device=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
