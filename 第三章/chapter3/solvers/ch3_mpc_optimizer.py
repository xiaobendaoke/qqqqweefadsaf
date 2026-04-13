from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MPCShellOptimizer:
    horizon: int = 3
    temporal_decay: float = 0.85
    completion_weight: float = 1.0
    energy_weight: float = 0.12
    cache_bonus: float = 0.18
    cache_miss_penalty: float = 0.10
    urgency_weight: float = 0.55

    def plan_action(self, observation: list[float], env: Any) -> list[float]:
        schema = env.get_observation_schema()
        user_start = int(schema["section_slices"]["associated_user_state"][0])
        user_end = int(schema["section_slices"]["associated_user_state"][1])
        cache_start = int(schema["section_slices"]["local_cache_summary"][0])
        cache_end = int(schema["section_slices"]["local_cache_summary"][1])

        user_blocks = self._parse_user_blocks(observation[user_start:user_end])
        cache_bitmap = observation[cache_start:cache_end]
        candidate_actions = self._candidate_actions(user_blocks)
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
                    move_scale_x=move_scale_x * (step_index + 1),
                    move_scale_y=move_scale_y * (step_index + 1),
                )
            if score > best_score:
                best_score = score
                best_action = action
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

    def _candidate_actions(self, user_blocks: list[dict[str, float]]) -> list[list[float]]:
        candidates: list[list[float]] = [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
        for block in user_blocks:
            if block["pending"] <= 0.0:
                continue
            direction = self._normalize_direction(block["rel_x"], block["rel_y"])
            if direction not in candidates:
                candidates.append(direction)
        return candidates

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
        move_scale_x: float,
        move_scale_y: float,
    ) -> float:
        score = -self.energy_weight * action_norm
        for block in user_blocks:
            if block["pending"] <= 0.0:
                continue
            predicted_rel_x = block["rel_x"] - action[0] * move_scale_x
            predicted_rel_y = block["rel_y"] - action[1] * move_scale_y
            predicted_distance = math.hypot(predicted_rel_x, predicted_rel_y)
            urgency = block["pending"] + self.urgency_weight * (1.0 - block["min_slack"])
            completion_proxy = urgency / max(0.08, predicted_distance + 0.05)
            service_index = 0
            if cache_bitmap:
                service_index = min(len(cache_bitmap) - 1, max(0, round(block["service_type_norm"] * (len(cache_bitmap) - 1))))
            cache_term = self.cache_bonus if cache_bitmap and cache_bitmap[service_index] >= 0.5 else -self.cache_miss_penalty
            slack_term = 0.5 * (1.0 - predicted_distance) * block["min_slack"]
            score += self.completion_weight * completion_proxy + cache_term + slack_term
        return float(score)
