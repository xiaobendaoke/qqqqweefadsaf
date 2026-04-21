"""第四章终稿复现实验命令行入口模块。

该模块负责从命令行触发 stage-6 多随机种子复现实验与结果打包流程。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapter4.marl import run_final_paper_package


def main() -> None:
    """解析命令行参数并执行终稿复现实验。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[72, 82, 92])
    parser.add_argument("--train-episodes", type=int, default=240)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--selected-candidate", type=str, default=None, help="Optional explicit main candidate label override.")
    parser.add_argument("--output-dir-name", type=str, default="paper_stage6_v2")
    parser.add_argument("--device", type=str, default="auto", help="Torch device request: auto, cpu, cuda, or cuda:N.")
    args = parser.parse_args()
    result = run_final_paper_package(
        seeds=args.seeds,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        selected_candidate_name=args.selected_candidate,
        output_dir_name=args.output_dir_name,
        device=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
