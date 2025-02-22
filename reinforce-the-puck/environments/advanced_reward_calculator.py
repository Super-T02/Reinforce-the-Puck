"""
File: advanced_reward_calculator.py
Author: Jonathan Schwab and Tom Freudenmann
Content: This file contains the WeightConfig and TimedReward classes.
"""

import math
from logging import getLogger

import numpy as np
from utils.config import WeightConfig


class Weights:
    def __init__(
        self,
        winner_weight: float = 10.0,
        closeness_puck_weight: float = 0.5,
        touch_puck_weight: float = 0.1,
        puck_direction_weight: float = 1.0,
        no_touch_penalty: float = -10.0,
        timed_penalty_active: bool = False,
        block_puck_weight: float = 1.0,
        stay_in_goal_weight: float = 1.0,
        **kwargs,
    ):
        self.winner_weight = winner_weight
        self.closeness_puck_weight = closeness_puck_weight
        self.touch_puck_weight = touch_puck_weight
        self.puck_direction_weight = puck_direction_weight
        self.no_touch_penalty = no_touch_penalty
        self.timed_penalty_active = timed_penalty_active
        self.block_puck_weight = block_puck_weight
        self.stay_in_goal_weight = stay_in_goal_weight

    def get(self, key: str, default: float = 0.0) -> float:
        return getattr(self, key, default)

    @staticmethod
    def from_dict(d: dict):
        return Weights(**d)

    @staticmethod
    def from_config(config: WeightConfig):
        return Weights(**config.to_dict())


class TimedReward:
    def __init__(
        self,
        max_steps: int,
        weights: Weights = Weights(),
    ):
        # Environment-based constants
        self.W = 10.0
        self.H = 8.0
        self.CENTER_X = self.W / 2.0
        self.CENTER_Y = self.H / 2.0
        self.GOAL_SIZE = 75 / 60
        self.SCALE = 60
        self.my_goal_center = np.array([0.83, 4.0], dtype=np.float32)

        # Initialize the reward calculator
        self._step_counter = 0
        self._reward = 0.0
        self._touched_puck = False
        self._max_steps = max_steps
        self._logger = getLogger(__name__)

        # Reward weights
        self._weights = weights
        self._time_weight = self._weights.winner_weight / self._max_steps

    def reset(self):
        self._step_counter = 0
        self._reward = 0.0
        self._touched_puck = False

    def compute_reward(self, observations, info) -> float:
        """Compute the reward for the given observations and info.

        Args:
            observations (np.ndarray): The observations from the environment.
            info (dict): The info dictionary from the environment.

        Returns:
            float: The computed reward.
        """
        if self._step_counter >= self._max_steps:
            self._logger.warning("Max steps reached. Please reset reward calc.")
            return 0.0

        self._step_counter += 1

        # Compute the reward
        reward = self._generic_reward(info)

        # Penalize for time
        if self._weights.timed_penalty_active:
            reward -= self._time_weight

        # Negative reward if the puck is not touched after 10 % of the steps
        if (
            self._step_counter > 0.1 * self._max_steps
            and not self._touched_puck
            and self._puck_in_own_half(observations)
            and self._puck_still(observations)
        ):
            reward += self._weights.no_touch_penalty

        # Check Prevent goal reward
        reward += self._calculate_blocking_reward(observations)
        reward += self._calculate_stay_in_goal_weight(observations)

        return reward

    def _puck_in_own_half(self, observations):
        puck_pos = np.array(observations[12:14], dtype=np.float32)
        puck_pos = puck_pos + [self.CENTER_X, self.CENTER_Y]

        return puck_pos[0] < self.CENTER_X

    def _puck_still(self, observations):
        puck_vel = np.array(observations[14:16], dtype=np.float32)
        puck_speed = np.linalg.norm(puck_vel)
        return puck_speed < 1e-6

    def _calculate_blocking_reward(self, observations):
        p1_pos = np.array(observations[0:2], dtype=np.float32)
        p2_pos = np.array(observations[6:8], dtype=np.float32)
        puck_pos = np.array(observations[12:14], dtype=np.float32)
        puck_vel = np.array(observations[14:16], dtype=np.float32)

        p1_pos = p1_pos + [self.CENTER_X, self.CENTER_Y]
        p2_pos = p2_pos + [self.CENTER_X, self.CENTER_Y]
        puck_pos = puck_pos + [self.CENTER_X, self.CENTER_Y]

        # Reward for blocking the puck
        reward_block_puck = 0
        if (
            puck_pos[0] < self.CENTER_X and puck_vel[0] < 0
        ):  # Puck in own half, moving toward goal
            if (
                self._dist_positions(p1_pos, p2_pos) < 50.0 / self.SCALE
            ):  # Close enough to block
                reward_block_puck += self._weights.block_puck_weight
        return reward_block_puck

    def _calculate_stay_in_goal_weight(self, observations):
        p1_pos = np.array(observations[0:2], dtype=np.float32)
        p2_pos = np.array(observations[6:8], dtype=np.float32)
        puck_pos = np.array(observations[12:14], dtype=np.float32)
        puck_vel = np.array(observations[14:16], dtype=np.float32)

        p1_pos = p1_pos + [self.CENTER_X, self.CENTER_Y]
        p2_pos = p2_pos + [self.CENTER_X, self.CENTER_Y]
        puck_pos = puck_pos + [self.CENTER_X, self.CENTER_Y]

        # Reward for staying in goal
        reward_stay_in_goal = 0
        dist_to_goal = self._dist_positions(p1_pos, self.my_goal_center)
        reward_stay_in_goal -= dist_to_goal * self._weights.stay_in_goal_weight
        return reward_stay_in_goal

    def _dist_positions(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def _generic_reward(self, info) -> float:
        winner = info.get("winner", 0.0)  # 0: tie, -1: opponent, 1: agent
        closeness_puck = info.get("reward_closeness_to_puck", 0.0)
        touch_puck = info.get("reward_touch_puck", 0.0)
        puck_direction = info.get("reward_puck_direction", 0.0)

        if touch_puck > 0.0:
            self._touched_puck = True

        r = (
            winner * self._weights.winner_weight
            + closeness_puck * self._weights.closeness_puck_weight
            + touch_puck * self._weights.touch_puck_weight
            + puck_direction * self._weights.puck_direction_weight
        )
        return r
