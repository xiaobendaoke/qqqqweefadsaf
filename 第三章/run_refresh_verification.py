"""第三章验证刷新入口。

该脚本用于重跑 smoke、主要单 UAV 基线与 compare-ch4，
并将关键结果汇总为单个 JSON，供 VERIFICATION.md 与论文材料引用。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.uav_mec.logging_utils import write_json
from chapter3.experiments import compare_with_chapter4, run_experiment
from chapter3.experiments.smoke import run_smoke


def main() -> None:
    seed = 42
    payload = {
        "seed": seed,
        "smoke": {
            mode: run_smoke(mode, seed=seed)
            for mode in ("import_only", "task_contract", "comms_contract", "scheduler_contract", "env_step", "episode")
        },
        "experiments": {
            "heuristic": run_experiment(seed=seed, episodes=1, hard=False, policy="heuristic", export_trajectory=False),
            "mpc": run_experiment(seed=seed, episodes=1, hard=False, policy="mpc", export_trajectory=False),
            "fixed_point": run_experiment(
                seed=seed,
                episodes=1,
                hard=False,
                policy="fixed_point",
                export_trajectory=False,
                steps_per_episode=20,
            ),
            "fixed_patrol": run_experiment(
                seed=seed,
                episodes=1,
                hard=False,
                policy="fixed_patrol",
                export_trajectory=False,
                steps_per_episode=20,
            ),
        },
        "compare_ch4": compare_with_chapter4(seed=seed, episodes=1),
    }
    output_path = Path("第三章/results/verification_refresh.json")
    write_json(output_path, payload)
    print(json.dumps({"verification_refresh_path": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
