from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapter3.experiments import compare_with_chapter4, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--profile", type=str, default="default", choices=["default", "hard"])
    parser.add_argument("--policy", type=str, default="heuristic", choices=["heuristic", "mpc"])
    parser.add_argument("--compare-ch4", action="store_true")
    args = parser.parse_args()

    if args.compare_ch4:
        result = compare_with_chapter4(seed=args.seed, episodes=args.episodes)
    else:
        result = run_experiment(
            seed=args.seed,
            episodes=args.episodes,
            hard=args.profile == "hard",
            policy=args.policy,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
