"""第三章图表打包命令行入口模块。

该模块负责从命令行触发第三章四策略实验与图表生成流程。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapter3.experiments import run_chapter3_figure_package


def main() -> None:
    """解析参数并运行第三章图表打包流程。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--profile", type=str, default="default", choices=["default", "hard"])
    parser.add_argument("--steps-per-episode", type=int, default=20)
    parser.add_argument("--compare-episodes", type=int, default=None)
    args = parser.parse_args()
    result = run_chapter3_figure_package(
        seed=args.seed,
        episodes=args.episodes,
        hard=args.profile == "hard",
        steps_per_episode=args.steps_per_episode,
        compare_episodes=args.compare_episodes,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
