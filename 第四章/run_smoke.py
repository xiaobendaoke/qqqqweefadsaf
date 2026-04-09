from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chapter4.experiments.smoke import run_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="episode", choices=["import_only", "env_step", "observation", "episode"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-uavs", type=int, default=1)
    parser.add_argument("--assignment-rule", type=str, default="nearest_uav", choices=["nearest_uav", "least_loaded_uav"])
    args = parser.parse_args()
    result = run_smoke(args.mode, seed=args.seed, num_uavs=args.num_uavs, assignment_rule=args.assignment_rule)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
