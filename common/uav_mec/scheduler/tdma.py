"""传输队列近似模块。

该模块以轻量 TDMA 排队模型近似共享无线链路的串行占用过程，
用于估计用户上行、UAV 协同转发和服务拉取链路的等待时延与队列长度。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TDMAQueue:
    """按链路 id 串行化传输时隙，近似 TDMA 发送队列。"""

    next_available_time_by_queue: dict[str, float] = field(default_factory=dict)
    scheduled_end_times_by_queue: dict[str, list[float]] = field(default_factory=dict)

    def _prune(self, current_time: float, queue_id: str) -> None:
        """移除当前时刻前已结束的传输占用。"""
        end_times = self.scheduled_end_times_by_queue.get(queue_id, [])
        if not end_times:
            self.next_available_time_by_queue[queue_id] = max(0.0, self.next_available_time_by_queue.get(queue_id, 0.0))
            return
        while end_times and end_times[0] <= current_time:
            end_times.pop(0)
        self.scheduled_end_times_by_queue[queue_id] = end_times
        self.next_available_time_by_queue[queue_id] = end_times[-1] if end_times else current_time

    def estimate_wait(self, current_time: float, queue_id: str = "default") -> float:
        self._prune(current_time, queue_id)
        return max(0.0, self.next_available_time_by_queue.get(queue_id, 0.0) - current_time)

    def schedule(self, current_time: float, duration: float, queue_id: str = "default") -> tuple[float, float, float]:
        """登记一次链路占用，并返回开始/结束/等待时间。"""
        self._prune(current_time, queue_id)
        start_time = max(current_time, self.next_available_time_by_queue.get(queue_id, 0.0))
        wait_time = max(0.0, start_time - current_time)
        end_time = start_time + duration
        self.next_available_time_by_queue[queue_id] = end_time
        self.scheduled_end_times_by_queue.setdefault(queue_id, []).append(end_time)
        return start_time, end_time, wait_time

    def reset(self) -> None:
        self.next_available_time_by_queue.clear()
        self.scheduled_end_times_by_queue.clear()

    def clone(self) -> "TDMAQueue":
        return TDMAQueue(
            next_available_time_by_queue=dict(self.next_available_time_by_queue),
            scheduled_end_times_by_queue={key: list(value) for key, value in self.scheduled_end_times_by_queue.items()},
        )

    def get_queue_length(self, queue_id: str = "default", current_time: float | None = None) -> int:
        if current_time is not None:
            self._prune(current_time, queue_id)
        return len(self.scheduled_end_times_by_queue.get(queue_id, []))
