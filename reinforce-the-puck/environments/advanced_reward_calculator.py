import math
from logging import getLogger

import numpy as np


class AdaptiveHockeyRewardCalculator:
    """
    Adapts offense/defense weighting based on puck position and speed
    in the given scaled environment:
      - W = 600 / 60 = 10
      - H = 480 / 60 = 8
      - The center is (5, 4)
      - Left (my) goal center
      - Right (opponent) goal center
    """

    def __init__(
        self,
        closeness_weight=0.1,
        touch_weight=1.0,
        direction_weight=0.5,
        defense_weight=0.2,
        offense_weight=0.3,
        win_bonus=100.0,
        lose_penalty=-94.0,
        puck_speed_threshold=14.0,
    ):
        self.logger = getLogger(__name__)
        # Environment-based constants
        self.W = 10.0
        self.H = 8.0
        self.CENTER_X = self.W / 2.0
        self.CENTER_Y = self.H / 2.0

        # Approximate goal centers in scaled coordinates
        self.my_goal_center = np.array([0.83, 4.0], dtype=np.float32)
        self.opponent_goal_center = np.array([9.17, 4.0], dtype=np.float32)

        # Main reward weights
        self.closeness_weight = closeness_weight
        self.touch_weight = touch_weight
        self.direction_weight = direction_weight
        self.defense_weight = defense_weight
        self.offense_weight = offense_weight
        self.win_bonus = win_bonus
        self.lose_penalty = lose_penalty

        # Threshold to consider puck "slow"
        self.puck_speed_threshold = puck_speed_threshold

    def compute_reward(self, observations, info):
        # Base partial rewards from the environment
        reward = 0.0
        reward += self.closeness_weight * info.get("reward_closeness_to_puck", 0.0)
        reward += self.touch_weight * info.get("reward_touch_puck", 0.0)
        reward += self.direction_weight * info.get("reward_puck_direction", 0.0)

        # Observations layout (scaled):
        # [ p1_x, p1_y, p1_angle, p1_linvel_x, p1_linvel_y, p1_angvel,
        #   p2_x, p2_y, p2_angle, p2_linvel_x, p2_linvel_y, p2_angvel,
        #   puck_x, puck_y, puck_linvel_x, puck_linvel_y, ...
        # ]
        p1_pos = np.array(observations[0:2], dtype=np.float32)
        p2_pos = np.array(observations[6:8], dtype=np.float32)
        puck_pos = np.array(observations[12:14], dtype=np.float32)
        puck_vel = np.array(observations[14:16], dtype=np.float32)

        p1_pos = p1_pos + [
            self.CENTER_X,
            self.CENTER_Y,
        ]  # because  in env: self.player2.position - [CENTER_X, CENTER_Y],

        p2_pos = p2_pos + [self.CENTER_X, self.CENTER_Y]
        puck_pos = puck_pos + [self.CENTER_X, self.CENTER_Y]

        # Decide whether to emphasize offense or defense
        puck_speed = np.linalg.norm(puck_vel)  # euclidean norm
        is_in_own_half = puck_pos[0] < self.CENTER_X
        is_puck_slow = puck_speed < self.puck_speed_threshold

        # self.logger.info(f"Puck speed: {puck_speed}, in own half: {is_in_own_half}")
        # self.logger.info(f"Puck pos: {puck_pos}, puck vel: {puck_vel}")
        # self.logger.info(f"Player 1 pos: {p1_pos}, player 2 pos: {p2_pos}")

        # If puck is in our half and slow -> offense, else defense
        if is_in_own_half and is_puck_slow:  # todo??
            w_offense, w_defense = 1.0, 0.1
        # self.logger.info("Offense")
        else:
            w_offense, w_defense = 0.1, 1.0  # soft weight
        # self.logger.info("Defense")

        # Compute partial strategy rewards
        r_defense = self._defense_reward(p1_pos, p2_pos, puck_pos)
        r_offense = self._offense_reward(puck_pos, puck_vel)

        # Combine with adaptive weights
        reward += w_offense * (self.offense_weight * r_offense)
        reward += w_defense * (self.defense_weight * r_defense)

        # Handle win/lose outcome
        winner = info.get("winner", None)
        if winner == 1:
            reward += self.win_bonus
        elif winner == 2:
            reward += self.lose_penalty

        return reward

    def _defense_reward(self, p1_pos, p2_pos, puck_pos):
        # Reward agent for being better positioned on line puck->my_goal
        dist_agent_line = self._point_line_distance(  # distance from player pos to line (going through puck and my goal)
            p1_pos, puck_pos, self.my_goal_center
        )
        dist_opponent_line = self._point_line_distance(  # also consider opponent --> target: be better than opponent (when he is close to the line, I must be closer)
            p2_pos, puck_pos, self.my_goal_center
        )

        return np.clip(dist_opponent_line - dist_agent_line, -5.0, 5.0)

    def _offense_reward(self, puck_pos, puck_vel):
        # Reward alignment puck->opponent_goal
        vec_goal = (
            self.opponent_goal_center - puck_pos
        )  # vector from puck to opponent goal
        dot_val = np.dot(
            vec_goal, puck_vel
        )  # measure of how much puck is moving towards the goal
        norm_vec = np.linalg.norm(vec_goal) + 1e-6
        norm_vel = np.linalg.norm(puck_vel) + 1e-6
        cos_angle = dot_val / (
            norm_vec * norm_vel
        )  # normalized (make independent of puck speed)
        return np.clip(cos_angle, -1.0, 1.0)

    @staticmethod
    def _point_line_distance(point, line_start, line_end):
        # Distance from point to infinite line through line_start->line_end
        line_vec = line_end - line_start
        p_vec = point - line_start
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq < 1e-9:
            return np.linalg.norm(point - line_start)
        proj = np.dot(p_vec, line_vec) / line_len_sq
        proj_point = line_start + proj * line_vec
        return np.linalg.norm(point - proj_point)
