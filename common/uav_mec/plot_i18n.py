"""绘图中文化辅助模块。"""

from __future__ import annotations

from typing import Any


CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans CN",
    "Arial Unicode MS",
]

POLICY_LABEL_CN = {
    "heuristic": "启发式",
    "mpc": "MPC 壳策略",
    "fixed_point": "固定点",
    "fixed_patrol": "固定巡航",
}

ASSIGNMENT_RULE_LABEL_CN = {
    "nearest_uav": "最近无人机",
    "least_loaded_uav": "最小负载无人机",
}

METHOD_LABEL_CN = {
    "ppo": "本文方法（PPO）",
    "heuristic": "启发式",
}

VARIANT_LABEL_CN = {
    "main": "主方法（PPO）",
    "with_reward_shaping": "加入奖励塑形",
    "no_movement_budget": "无移动预算",
}

ENERGY_COMPONENT_LABEL_CN = {
    "uav_move_energy": "无人机移动能耗",
    "uav_compute_energy": "无人机计算能耗",
    "ue_local_energy": "用户本地计算能耗",
    "ue_uplink_energy": "用户上行传输能耗",
    "bs_compute_energy": "基站计算能耗",
    "relay_fetch_energy": "中继/回源能耗",
    "bs_fetch_tx_energy": "基站回源传输能耗",
}

METRIC_LABEL_CN = {
    "completion_rate": "任务完成率",
    "average_latency": "平均时延",
    "average_latency_completed": "已完成任务平均时延",
    "latency_per_generated_task": "单位生成任务时延",
    "total_energy": "总能耗",
    "cache_hit_rate": "缓存命中率",
    "fairness_user_completion": "用户公平性",
    "fairness_uav_load": "无人机负载公平性",
    "deadline_violation_rate": "截止违约率",
    "reliability_violation_rate": "可靠性违约率",
    "team_return": "团队回报",
    "mean_step_action_magnitude": "平均动作幅度",
    "mean_step_energy": "平均步系统能耗",
    "delta_total_energy": "总能耗差值",
    "delta_average_latency": "平均时延差值",
    "delta_completion_rate": "完成率差值",
}


def configure_matplotlib_for_chinese(plt: Any) -> Any:
    """配置 Matplotlib 全局字体，尽可能稳定显示中文。"""
    plt.rcParams["font.sans-serif"] = CHINESE_FONT_CANDIDATES
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def assignment_rule_label(rule: str) -> str:
    return ASSIGNMENT_RULE_LABEL_CN.get(rule, rule)


def method_label(method: str) -> str:
    return METHOD_LABEL_CN.get(method, method)


def policy_label(policy: str) -> str:
    return POLICY_LABEL_CN.get(policy, policy)


def variant_label(variant: str) -> str:
    return VARIANT_LABEL_CN.get(variant, variant)


def metric_label(metric: str) -> str:
    return METRIC_LABEL_CN.get(metric, metric)
