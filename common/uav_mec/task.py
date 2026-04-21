"""任务对象定义模块。

该模块定义任务从生成到完成或过期全过程中的状态字段，
包括服务类型、时延约束、执行目标、能耗分解和可靠性结果，是环境日志的核心载体。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Task:
    """描述单个任务从生成到完成/过期全过程中的状态与成本。"""

    task_id: str
    user_id: int
    service_type: int
    input_size_bits: float
    cpu_cycles: float
    arrival_time: float
    slack: float
    deadline: float
    required_reliability: float
    status: str = "pending"
    associated_uav_id: int | None = None
    assigned_uav_id: int | None = None
    execution_target: str | None = None
    cache_hit: bool | None = None
    fetch_source: str | None = None
    fetch_wait_delay: float = 0.0
    fetch_delay: float = 0.0
    queue_delay: float = 0.0
    transmission_delay: float = 0.0
    compute_wait_delay: float = 0.0
    compute_delay: float = 0.0
    total_latency: float = 0.0
    success_probability: float = 0.0
    completed: bool = False
    scheduled_time: float | None = None
    completion_time: float | None = None
    ue_local_energy: float = 0.0
    ue_uplink_energy: float = 0.0
    uav_compute_energy: float = 0.0
    bs_compute_energy: float = 0.0
    relay_fetch_energy: float = 0.0
    bs_fetch_tx_energy: float = 0.0

    def mark_result(
        self,
        *,
        execution_target: str,
        cache_hit: bool,
        fetch_source: str | None,
        fetch_wait_delay: float,
        fetch_delay: float,
        queue_delay: float,
        transmission_delay: float,
        compute_wait_delay: float,
        compute_delay: float,
        total_latency: float,
        success_probability: float,
        completed: bool,
        status: str,
        scheduled_time: float,
        completion_time: float,
        associated_uav_id: int | None = None,
        assigned_uav_id: int | None = None,
        ue_local_energy: float = 0.0,
        ue_uplink_energy: float = 0.0,
        uav_compute_energy: float = 0.0,
        bs_compute_energy: float = 0.0,
        relay_fetch_energy: float = 0.0,
        bs_fetch_tx_energy: float = 0.0,
    ) -> None:
        """将一次卸载决策回写到任务对象，形成统一日志记录。"""
        self.execution_target = execution_target
        self.cache_hit = cache_hit
        self.fetch_source = fetch_source
        self.fetch_wait_delay = fetch_wait_delay
        self.fetch_delay = fetch_delay
        self.queue_delay = queue_delay
        self.transmission_delay = transmission_delay
        self.compute_wait_delay = compute_wait_delay
        self.compute_delay = compute_delay
        self.total_latency = total_latency
        self.success_probability = success_probability
        self.completed = completed
        self.status = status
        self.scheduled_time = scheduled_time
        self.completion_time = completion_time
        self.associated_uav_id = associated_uav_id
        self.assigned_uav_id = assigned_uav_id
        self.ue_local_energy = ue_local_energy
        self.ue_uplink_energy = ue_uplink_energy
        self.uav_compute_energy = uav_compute_energy
        self.bs_compute_energy = bs_compute_energy
        self.relay_fetch_energy = relay_fetch_energy
        self.bs_fetch_tx_energy = bs_fetch_tx_energy
