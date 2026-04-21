"""第四章结果目录辅助模块。

当前目录约定：

- `第四章/results/legacy/`: legacy mobility-only baseline 与相关 smoke / episode
- `第四章/results/joint/`: joint heuristic / joint RL 与相关 smoke / eval
- `第四章/results/stage5/`: stage-5 论文调参与主配置选择
- `第四章/results/stage6/`: stage-6 终稿复跑与验证刷新
"""

from __future__ import annotations

from pathlib import Path


RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
LEGACY_RESULTS_DIR = RESULTS_ROOT / "legacy"
JOINT_RESULTS_DIR = RESULTS_ROOT / "joint"
STAGE5_RESULTS_ROOT = RESULTS_ROOT / "stage5"
STAGE6_RESULTS_ROOT = RESULTS_ROOT / "stage6"


def trainer_results_dir(trainer_mode: str) -> Path:
    return LEGACY_RESULTS_DIR if trainer_mode == "legacy_mobility_only" else JOINT_RESULTS_DIR


def baseline_results_dir(policy_id: str) -> Path:
    return LEGACY_RESULTS_DIR if policy_id == "legacy_mobility_only" else JOINT_RESULTS_DIR


def stage5_dir(name: str = "paper_stage5_v2") -> Path:
    return STAGE5_RESULTS_ROOT / name


def stage6_dir(name: str = "paper_stage6_v2") -> Path:
    return STAGE6_RESULTS_ROOT / name
