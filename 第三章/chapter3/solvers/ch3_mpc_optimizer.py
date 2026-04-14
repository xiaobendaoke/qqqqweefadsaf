from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MPCShellOptimizer:
    horizon: int = 4
    temporal_decay: float = 0.82
    completion_weight: float = 1.05
    energy_weight: float = 0.10
    cache_bonus: float = 0.18
    cache_miss_penalty: float = 0.10
    urgency_weight: float = 0.55
    inertia_weight: float = 0.22
    turn_penalty_weight: float = 0.16
    travel_scale_bias: float = 0.92
    _last_action: list[float] | None = None

    def plan_action(self, observation: list[float], env: Any) -> list[float]:
        schema = env.get_observation_schema()
        user_start = int(schema["section_slices"]["associated_user_state"][0])
        user_end = int(schema["section_slices"]["associated_user_state"][1])
        cache_start = int(schema["section_slices"]["local_cache_summary"][0])
        cache_end = int(schema["section_slices"]["local_cache_summary"][1])

        user_blocks = self._parse_user_blocks(observation[user_start:user_end])
        cache_bitmap = observation[cache_start:cache_end]
        previous_action = self._last_action or [0.0, 0.0]
        candidate_actions = self._candidate_actions(user_blocks, previous_action=previous_action)
        best_action = [0.0, 0.0]
        best_score = float("-inf")
        move_scale_x = env.config.uav_speed * env.config.time_slot_duration / max(env.config.area_width, 1e-6)
        move_scale_y = env.config.uav_speed * env.config.time_slot_duration / max(env.config.area_height, 1e-6)

        for action in candidate_actions:
            score = 0.0
            action_norm = math.hypot(action[0], action[1])
            for step_index in range(self.horizon):
                weight = self.temporal_decay ** step_index
                score += weight * self._score_step(
                    user_blocks=user_blocks,
                    cache_bitmap=cache_bitmap,
                    action=action,
                    action_norm=action_norm,
                    previous_action=previous_action,
                    move_scale_x=move_scale_x * (step_index + 1),
                    move_scale_y=move_scale_y * (step_index + 1),
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
        cache_bitmap: list[float],
        action: list[float],
        action_norm: float,
        previous_action: list[float],
        move_scale_x: float,
        move_scale_y: float,
    ) -> float:
        score = -self.energy_weight * action_norm
        prev_norm = math.hypot(previous_action[0], previous_action[1])
        if prev_norm > 1e-6 and action_norm > 1e-6:
            alignment = (action[0] * previous_action[0] + action[1] * previous_action[1]) / max(1e-6, action_norm * prev_norm)
            score += self.inertia_weight * alignment
        score -= self.turn_penalty_weight * math.hypot(action[0] - previous_action[0], action[1] - previous_action[1])
        for block in user_blocks:
            if block["pending"] <= 0.0:
                continue
            predicted_rel_x = block["rel_x"] - action[0] * move_scale_x
            predicted_rel_y = block["rel_y"] - action[1] * move_scale_y
            predicted_distance = math.hypot(predicted_rel_x, predicted_rel_y)
            urgency = block["pending"] + self.urgency_weight * (1.0 - block["min_slack"])
            completion_proxy = urgency / max(0.08, self.travel_scale_bias * predicted_distance + 0.05)
            service_index = 0
            if cache_bitmap:
                service_index = min(len(cache_bitmap) - 1, max(0, round(block["service_type_norm"] * (len(cache_bitmap) - 1))))
            cache_term = self.cache_bonus if cache_bitmap and cache_bitmap[service_index] >= 0.5 else -self.cache_miss_penalty
            slack_term = 0.5 * (1.0 - predicted_distance) * block["min_slack"]
            score += self.completion_weight * completion_proxy + cache_term + slack_term
        return float(score)
