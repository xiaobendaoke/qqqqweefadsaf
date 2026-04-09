from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TDMAQueue:
    next_available_time_by_queue: dict[str, float] = field(default_factory=dict)
    queue_length_by_queue: dict[str, int] = field(default_factory=dict)

    def estimate_wait(self, current_time: float, queue_id: str = "default") -> float:
        return max(0.0, self.next_available_time_by_queue.get(queue_id, 0.0) - current_time)

    def schedule(self, current_time: float, duration: float, queue_id: str = "default") -> tuple[float, float, float]:
        start_time = max(current_time, self.next_available_time_by_queue.get(queue_id, 0.0))
        wait_time = max(0.0, start_time - current_time)
        end_time = start_time + duration
        self.next_available_time_by_queue[queue_id] = end_time
        self.queue_length_by_queue[queue_id] = self.queue_length_by_queue.get(queue_id, 0) + 1
        return start_time, end_time, wait_time

    def reset(self) -> None:
        self.next_available_time_by_queue.clear()
        self.queue_length_by_queue.clear()

    def clone(self) -> "TDMAQueue":
        return TDMAQueue(
            next_available_time_by_queue=dict(self.next_available_time_by_queue),
            queue_length_by_queue=dict(self.queue_length_by_queue),
        )

    def get_queue_length(self, queue_id: str = "default") -> int:
        return self.queue_length_by_queue.get(queue_id, 0)
