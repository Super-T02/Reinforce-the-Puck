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
      GOAL_SIZE = 75
    """

    def __init__(
        self,
        closeness_weight=0.3,
        touch_weight=2.0,
        direction_weight=0.5,
        defense_weight=0.2,
        offense_weight=0.3,
        win_bonus=70.0,
        lose_penalty=-60.0,
        puck_speed_threshold=14.0,
    ):
        self.logger = getLogger(__name__)
        # Environment-based constants
        self.W = 10.0
        self.H = 8.0
        self.CENTER_X = self.W / 2.0
        self.CENTER_Y = self.H / 2.0
        self.GOAL_SIZE = 75 / 60

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

        self.own_half_counter = 0

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
        if is_in_own_half:  # and is_puck_slow:  # todo??
            w_offense, w_defense = 0.9, 0.2
            if puck_speed < self.puck_speed_threshold:
                self.own_half_counter += 0.05
        # self.logger.info("Offense")
        else:
            self.own_half_counter = 0
            w_offense, w_defense = 0.2, 0.9  # soft weight
        # self.logger.info("Defense")

        # Compute partial strategy rewards
        r_defense = self._defense_reward(p1_pos, p2_pos, puck_pos)
        r_offense = self._offense_reward(puck_pos, puck_vel, p2_pos)

        # Combine with adaptive weights
        reward += w_offense * (self.offense_weight * r_offense)
        reward += w_defense * (self.defense_weight * r_defense)

        reward -= (
            self.own_half_counter
        )  # punish for being long in own half when puck is slow

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

        goal_boarder_top = self.my_goal_center[1] - self.GOAL_SIZE / 2
        goal_boarder_bottom = self.my_goal_center[1] + self.GOAL_SIZE / 2

        reward = np.clip(dist_opponent_line - dist_agent_line, -5.0, 5.0)

        # punish beeing outside the boarders
        if p1_pos[1] < goal_boarder_top or p1_pos[1] > goal_boarder_bottom:
            reward -= 1
        return reward

    def _offense_reward(self, puck_pos, puck_vel, p2_pos):
        # Reward alignment puck->opponent_goal
        speed = np.linalg.norm(puck_vel)
        if speed < 1e-6:
            # No movement -> no reward
            return 0.0

        # 1) Reward for going in the direction of the opponent's goal

        # Use the x-component normalized by total speed
        # Opponent's goal is to the right, so positive x-direction is "good"
        direction_reward = puck_vel[0] / speed

        # Clip to [-1, 1] to avoid extreme values
        direction_reward = np.clip(direction_reward, -1.0, 1.0)

        # 2) Penalty if the puck is going in the direction of p2 (don't shoot the opponent because he will defend)
        # Calculate angle between puck_vel and vector from puck to p2
        vec_p2 = p2_pos - puck_pos
        dist_p2 = np.linalg.norm(vec_p2)
        penalty = 0.0

        if dist_p2 > 1e-6:
            # Cosine of angle between puck_vel and p2 vector
            cos_angle_p2 = np.dot(puck_vel, vec_p2) / (speed * dist_p2)

            # If cos_angle_p2 ~ 1, the puck is going directly toward p2
            if cos_angle_p2 > 0.8:
                penalty_strength = (cos_angle_p2 - 0.8) * 5.0
                penalty = np.clip(penalty_strength, 0.0, 2.0)

        # Combine: reward for going to the right, minus penalty
        return direction_reward - penalty

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
