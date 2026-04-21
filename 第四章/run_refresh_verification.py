"""第四章验证刷新入口。

该脚本用于重跑 smoke、启发式主实验、sensitive assignment 与 compare-ch4，
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
from chapter4.experiments import run_experiment, run_sensitive_experiment
from chapter4.experiments.smoke import run_smoke
from chapter4.results_paths import stage6_dir


def _compare_ch4() -> dict[str, object]:
    third_chapter_root = REPO_ROOT / "第三章"
    if str(third_chapter_root) not in sys.path:
        sys.path.insert(0, str(third_chapter_root))
    from chapter3.experiments import compare_with_chapter4

    return compare_with_chapter4(seed=42, episodes=1)


def main() -> None:
    seed = 42
    payload = {
        "seed": seed,
        "smoke": {
            "u1": {
                mode: run_smoke(mode, seed=seed, num_uavs=1, assignment_rule="nearest_uav")
                for mode in ("import_only", "env_step", "observation", "episode")
            },
            "u2": {
                mode: run_smoke(mode, seed=seed, num_uavs=2, assignment_rule="nearest_uav")
                for mode in ("env_step", "observation", "episode")
            },
        },
        "experiments": {
            "default_u2": run_experiment(seed=seed, episodes=1, hard=False, num_uavs=2, assignment_rule="nearest_uav"),
            "hard_u2": run_experiment(seed=seed, episodes=1, hard=True, num_uavs=2, assignment_rule="nearest_uav"),
            "sensitive_u2_nearest": run_sensitive_experiment(seed=seed, episodes=1, num_uavs=2, assignment_rule="nearest_uav"),
            "sensitive_u2_least_loaded": run_sensitive_experiment(seed=seed, episodes=1, num_uavs=2, assignment_rule="least_loaded_uav"),
            "sensitive_u3_nearest": run_sensitive_experiment(seed=seed, episodes=1, num_uavs=3, assignment_rule="nearest_uav"),
            "sensitive_u3_least_loaded": run_sensitive_experiment(seed=seed, episodes=1, num_uavs=3, assignment_rule="least_loaded_uav"),
        },
        "compare_ch4": _compare_ch4(),
    }
    output_path = stage6_dir() / "verification_refresh.json"
    write_json(output_path, payload)
    print(json.dumps({"verification_refresh_path": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
