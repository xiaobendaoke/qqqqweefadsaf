from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MPCShellOptimizer:
    horizon: int = 4
    temporal_decay: float = 0.84
    completion_weight: float = 1.15
    energy_weight: float = 0.08
    queue_weight: float = 0.16
    coverage_weight: float = 0.24
    cache_bonus: float = 0.20
    cache_miss_penalty: float = 0.14
    urgency_weight: float = 0.70
    backlog_weight: float = 0.18
    inertia_weight: float = 0.24
    turn_penalty_weight: float = 0.12
    _last_action: list[float] | None = None

    def plan_action(self, observation: list[float], env: Any) -> list[float]:
        schema = env.get_observation_schema()
        user_start = int(schema["section_slices"]["associated_user_state"][0])
        user_end = int(schema["section_slices"]["associated_user_state"][1])
        cache_start = int(schema["section_slices"]["local_cache_summary"][0])
        cache_end = int(schema["section_slices"]["local_cache_summary"][1])
        cache_score_start = int(schema["section_slices"]["cache_score_summary"][0])
        cache_score_end = int(schema["section_slices"]["cache_score_summary"][1])
        tx_start = int(schema["section_slices"]["tx_queue_summary"][0])
        compute_start = int(schema["section_slices"]["compute_queue_summary"][0])
        backlog_start = int(schema["section_slices"]["task_backlog_summary"][0])

        user_blocks = self._parse_user_blocks(observation[user_start:user_end])
        cache_bitmap = observation[cache_start:cache_end]
        cache_scores = observation[cache_score_start:cache_score_end]
        tx_queue = observation[tx_start : tx_start + 2]
        compute_queue = observation[compute_start : compute_start + 2]
        backlog_summary = observation[backlog_start : backlog_start + 2]
        previous_action = self._last_action or [0.0, 0.0]
        candidate_actions = self._candidate_actions(user_blocks, previous_action=previous_action)
        best_action = [0.0, 0.0]
        best_score = float("-inf")
        current_position = list(env.uavs[0].position)
        pending_tasks = list(getattr(env, "pending_tasks", []))
        users = {user.user_id: user for user in getattr(env, "users", [])}
        max_step_distance = env.config.uav_speed * env.config.time_slot_duration

        for action in candidate_actions:
            score = 0.0
            action_norm = math.hypot(action[0], action[1])
            predicted_position = list(current_position)
            for step_index in range(self.horizon):
                weight = self.temporal_decay ** step_index
                predicted_position[0] = min(
                    max(predicted_position[0] + action[0] * max_step_distance, 0.0),
                    env.config.area_width,
                )
                predicted_position[1] = min(
                    max(predicted_position[1] + action[1] * max_step_distance, 0.0),
                    env.config.area_height,
                )
                score += weight * self._score_step(
                    user_blocks=user_blocks,
                    pending_tasks=pending_tasks,
                    users=users,
                    cache_bitmap=cache_bitmap,
                    cache_scores=cache_scores,
                    tx_queue=tx_queue,
                    compute_queue=compute_queue,
                    backlog_summary=backlog_summary,
                    action=action,
                    action_norm=action_norm,
                    previous_action=previous_action,
                    predicted_position=predicted_position,
                    current_time=env.current_step * env.config.time_slot_duration + step_index * env.config.time_slot_duration,
                    coverage_radius=env.config.uav_coverage_radius,
                )
            if score > best_score:
                best_score = score
                best_action = action
        self._last_action = [float(best_action[0]), float(best_action[1])]
        return best_action

    def _parse_user_blocks(self, values: list[float]) -> list[dict[str, float]]:
        blocks: list[dict[str, float]] = []
        for offset in range(0, len(values), 5):
            block = values[offset : offset + 5]
            if len(block) < 5:
                continue
            blocks.append(
                {
                    "rel_x": float(block[0]),
                    "rel_y": float(block[1]),
                    "pending": max(0.0, float(block[2])),
                    "min_slack": min(1.0, max(0.0, float(block[3]))),
                    "service_type_norm": min(1.0, max(0.0, float(block[4]))),
                }
            )
        return blocks

    def _candidate_actions(self, user_blocks: list[dict[str, float]], *, previous_action: list[float]) -> list[list[float]]:
        candidate_map: dict[tuple[float, float], list[float]] = {}

        def add_candidate(action: list[float]) -> None:
            rounded = (round(float(action[0]), 4), round(float(action[1]), 4))
            candidate_map.setdefault(rounded, [float(action[0]), float(action[1])])

        add_candidate([0.0, 0.0])
        for scale in (0.35, 0.6, 0.85):
            for angle_deg in range(0, 360, 45):
                angle_rad = math.radians(angle_deg)
                add_candidate([scale * math.cos(angle_rad), scale * math.sin(angle_rad)])
        if math.hypot(previous_action[0], previous_action[1]) > 1e-6:
            add_candidate(previous_action)
            add_candidate([0.75 * previous_action[0], 0.75 * previous_action[1]])
        for block in user_blocks:
            if block["pending"] <= 0.0:
                continue
            direction = self._normalize_direction(block["rel_x"], block["rel_y"])
            add_candidate(direction)
            add_candidate([0.7 * direction[0], 0.7 * direction[1]])
            if math.hypot(previous_action[0], previous_action[1]) > 1e-6:
                blended = self._normalize_direction(
                    0.65 * direction[0] + 0.35 * previous_action[0],
                    0.65 * direction[1] + 0.35 * previous_action[1],
                )
                add_candidate(blended)
        return list(candidate_map.values())

    def _normalize_direction(self, rel_x: float, rel_y: float) -> list[float]:
        norm = math.hypot(rel_x, rel_y)
        if norm <= 1e-6:
            return [0.0, 0.0]
        return [rel_x / max(1.0, norm), rel_y / max(1.0, norm)]

    def _score_step(
        self,
        *,
        user_blocks: list[dict[str, float]],
        pending_tasks: list[Any],
        users: dict[int, Any],
        cache_bitmap: list[float],
        cache_scores: list[float],
        tx_queue: list[float],
        compute_queue: list[float],
        backlog_summary: list[float],
        action: list[float],
        action_norm: float,
        previous_action: list[float],
        predicted_position: list[float],
        current_time: float,
        coverage_radius: float,
    ) -> float:
        score = -self.energy_weight * action_norm
        prev_norm = math.hypot(previous_action[0], previous_action[1])
        if prev_norm > 1e-6 and action_norm > 1e-6:
            alignment = (action[0] * previous_action[0] + action[1] * previous_action[1]) / max(1e-6, action_norm * prev_norm)
            score += self.inertia_weight * alignment
        score -= self.turn_penalty_weight * math.hypot(action[0] - previous_action[0], action[1] - previous_action[1])
        score -= self.queue_weight * (float(tx_queue[1]) + float(compute_queue[1]))
        score -= self.backlog_weight * float(backlog_summary[0])

        if pending_tasks and users:
            for task in pending_tasks:
                user = users.get(task.user_id)
                if user is None:
                    continue
                distance = math.dist((predicted_position[0], predicted_position[1]), (user.position[0], user.position[1]))
                distance_norm = min(1.5, distance / max(coverage_radius, 1e-6))
                time_remaining = max(0.0, task.deadline - current_time)
                urgency = 1.0 / max(0.15, time_remaining)
                service_index = int(task.service_type)
                cache_term = self.cache_bonus if service_index < len(cache_bitmap) and cache_bitmap[service_index] >= 0.5 else -self.cache_miss_penalty
                if service_index < len(cache_scores):
                    cache_term += 0.08 * float(cache_scores[service_index])
                coverage_term = self.coverage_weight if distance <= coverage_radius else -self.coverage_weight * distance_norm
                completion_proxy = max(0.0, 1.15 - distance_norm)
                score += self.completion_weight * completion_proxy
                score += self.urgency_weight * urgency
                score += cache_term
                score += coverage_term
        else:
            for block in user_blocks:
                if block["pending"] <= 0.0:
                    continue
                predicted_distance = math.hypot(block["rel_x"], block["rel_y"])
                urgency = block["pending"] + self.urgency_weight * (1.0 - block["min_slack"])
                completion_proxy = urgency / max(0.08, predicted_distance + 0.08)
                service_index = min(len(cache_bitmap) - 1, max(0, round(block["service_type_norm"] * max(1, len(cache_bitmap) - 1)))) if cache_bitmap else 0
                cache_term = self.cache_bonus if cache_bitmap and cache_bitmap[service_index] >= 0.5 else -self.cache_miss_penalty
                score += self.completion_weight * completion_proxy + cache_term
        return float(score)
