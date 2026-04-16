"""计算资源排队近似模块。

该模块使用轻量的单服务器排队模型，
近似描述 UAV 与基站在计算资源上的串行执行过程，用于估计等待时延与队列长度。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ComputeQueue:
    """按队列 id 串行化计算资源，近似 UAV/BS 的单服务器排队过程。"""

    next_available_time_by_queue: dict[str, float] = field(default_factory=dict)
    scheduled_end_times_by_queue: dict[str, list[float]] = field(default_factory=dict)

    def _prune(self, current_time: float, queue_id: str) -> None:
        """清理已完成任务，保证等待时间和队长估计基于当前时刻。"""
        end_times = self.scheduled_end_times_by_queue.get(queue_id, [])
        while end_times and end_times[0] <= current_time:
            end_times.pop(0)
        self.scheduled_end_times_by_queue[queue_id] = end_times
        self.next_available_time_by_queue[queue_id] = end_times[-1] if end_times else current_time

    def estimate_wait(self, current_time: float, queue_id: str = "default") -> float:
        self._prune(current_time, queue_id)
        return max(0.0, self.next_available_time_by_queue.get(queue_id, 0.0) - current_time)

    def schedule(self, current_time: float, duration: float, queue_id: str = "default") -> tuple[float, float, float]:
        """在指定队列中登记一段计算占用，并返回开始/结束/等待时间。"""
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

    def clone(self) -> "ComputeQueue":
        return ComputeQueue(
            next_available_time_by_queue=dict(self.next_available_time_by_queue),
            scheduled_end_times_by_queue={key: list(value) for key, value in self.scheduled_end_times_by_queue.items()},
        )

    def get_queue_length(self, queue_id: str = "default", current_time: float | None = None) -> int:
        if current_time is not None:
            self._prune(current_time, queue_id)
        return len(self.scheduled_end_times_by_queue.get(queue_id, []))
