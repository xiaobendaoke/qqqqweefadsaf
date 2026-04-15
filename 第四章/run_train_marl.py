from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapter4.marl import run_marl_training


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the shared-PPO Chapter 4 agent. The CLI defaults are for quick iteration and are not the final paper preset."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-episodes", type=int, default=24, help="Training episodes for this run; final paper runs use the stage-6 pipeline.")
    parser.add_argument("--num-uavs", type=int, default=2)
    parser.add_argument("--assignment-rule", type=str, default="nearest_uav", choices=["nearest_uav", "least_loaded_uav"])
    parser.add_argument("--tag", type=str, default="default")
    args = parser.parse_args()
    result = run_marl_training(
        seed=args.seed,
        train_episodes=args.train_episodes,
        num_uavs=args.num_uavs,
        assignment_rule=args.assignment_rule,
        overrides={"output_tag": args.tag},
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
