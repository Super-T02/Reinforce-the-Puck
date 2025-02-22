"""
File: hockey_wrapper.py
Author: Jonathan Schwab and Tom Freudenmann
Content: This file contains the HokeyEnvWrapper class.
"""

import logging
import os

import hockey.hockey_env as h_env
import imageio
import numpy as np
from agents.base_agent import BaseAgent
from agents.moe_agent import MOEAgent
from environments.advanced_reward_calculator import TimedReward, Weights
from environments.base_wrapper import BaseEnvWrapper
from gymnasium.spaces.box import Box
from PIL import Image
from utils import workspace_dir


class HokeyEnvWrapper(BaseEnvWrapper):
    def __init__(
        self,
        max_steps: int,
        do_render: bool = False,
        agent: BaseAgent = None,
        opponent_agent: BaseAgent = None,
        mode: int = h_env.Mode.NORMAL,
        start_training_after_steps: int = 10000,
        train_both: bool = False,
        weights: Weights = Weights(),
    ):
        self._do_render = do_render
        self.env = h_env.HockeyEnv(mode=mode)
        self.env.render_mode = "rgb_array" if do_render else None
        self.agent = agent
        self.opponent_agent = opponent_agent
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.action_space = Box(
            self.action_space.low[:4],
            self.action_space.high[:4],
            (4,),
            self.action_space.dtype,
        )
        self._last_observation = (None, 0, False, False, {})
        self._last_opponent_observation = (None, 0, False, False, {})
        self._logger = logging.getLogger(__name__)
        self._max_steps = max_steps
        self._weights = weights
        self.record = False
        self.record_path = os.path.join(workspace_dir, "gifs")
        self.frames = []
        self.name = "Hockey-v0"
        self.train_both = train_both

        # self.reward_calculator = AdaptiveHockeyRewardCalculator()
        self.reward_calculator = TimedReward(max_steps, weights)
        self._start_training_after_steps = start_training_after_steps
        self._steps = 0
        self.reset()

    def reset(self):
        super().reset()
        state = self.env.obs_agent_two()
        self.reward_calculator.reset()
        self._last_opponent_observation = (state, 0, False, False, {})

    def render(self):
        """Render the environment."""
        frame = self.env.render("rgb_array" if self.record else "human")
        if self.record:
            frame = Image.fromarray(frame)
            self.frames.append(frame)

    def save_gif(self):
        """Save the captured frames as a GIF."""
        if self.record and self.frames and self.frames[0] is not None:
            imageio.mimsave(self.record_path, self.frames, fps=30)
            self._logger.info(f"GIF saved to {self.record_path}")
            print(f"GIF saved to {self.record_path}")
            self.frames = []
        else:
            print("No frames to save.")

    def step(self, save=True):
        # Get states
        state_agent = self._last_observation[0]
        state_opponent = self._last_opponent_observation[0]

        action1 = np.array(self.agent.act(state_agent))
        # Get actions
        if isinstance(self.agent, MOEAgent):
            moe_action = self.agent.last_action

        action2 = np.array(self.opponent_agent.act(state_opponent))

        # Agent 1
        actions = np.hstack([action1, action2])
        obs, r, d, t, info = self.env.step(actions)
        self._last_observation = (obs, r, d, t, info)

        # Agent 2
        info_opponent = self.env.get_info_agent_two()
        obs_opponent, r_opponent = (
            self.env.obs_agent_two(),
            self.env.get_reward_agent_two(info_opponent),
        )
        self._last_opponent_observation = (
            obs_opponent,
            r_opponent,
            d,
            t,
            info_opponent,
        )

        if save:
            if isinstance(self.agent, MOEAgent):
                self.agent.save_experience(
                    state_agent, moe_action, *self._last_observation
                )
                if self.agent.last_actor == "a":
                    self.agent.agent_a.save_experience(
                        state_agent, action1, *self._last_observation
                    )
                else:
                    self.agent.agent_b.save_experience(
                        state_agent, action1, *self._last_observation
                    )

            else:
                self.agent.save_experience(
                    state_agent, action1, *self._last_observation
                )
            if isinstance(self.opponent_agent, MOEAgent):
                self.opponent_agent.save_experience(
                    state_opponent,
                    self.opponent_agent.last_action,
                    *self._last_opponent_observation,
                )
            else:
                self.opponent_agent.save_experience(
                    state_opponent, action2, *self._last_opponent_observation
                )

        return self._last_observation

    def compute_reward_agent(self, obs, r, d, t, info, evaluate=False) -> float:
        return (
            self.reward_calculator.compute_reward(obs, info)
            if not evaluate
            else info.get("winner", 0.0) * self._weights.winner_weight
        )

    def run_eval(self) -> float:
        """Run one episode of the environment.

        Returns:
            float: The total reward received in the episode.
        """
        done = False
        reward_agent = 0
        reward_opponent = 0
        for _ in range(self._max_steps):
            self.step()
            reward_agent += self.compute_reward_agent(
                *self._last_observation, evaluate=True
            )
            reward_opponent += self.compute_reward_agent(
                *self._last_opponent_observation, evaluate=True
            )
            done = self._last_observation[2]
            if done:
                break
            self.render() if self._do_render else None
        self._logger.info("Episode finished. Total reward agent: %f", reward_agent)
        self._logger.info(
            "Episode finished. Total reward opponent: %f", reward_opponent
        )
        return reward_agent, reward_opponent

    def run_train_episode(self, i: int, train: bool = True) -> float:
        """Run a single episode of the training.

        Args:
            i (int): The episode number.

        Returns:
            float: The total reward of the episode.
        """
        with self.agent.train_context(), self.opponent_agent.train_context():
            reward_agent = 0
            reward_opponent = 0
            self.reset()
            self.agent.reset()
            self.opponent_agent.reset()
            done = False

            for _ in range(self._max_steps):
                self.step()
                done = self._last_observation[2]
                trunc = self._last_observation[3]
                reward_agent += self.compute_reward_agent(*self._last_observation)
                reward_opponent += self.compute_reward_agent(
                    *self._last_opponent_observation
                )
                self._steps += 1
                if done or trunc:
                    break

            if self.started_training and train:
                self.agent.train(
                    reward_agent
                    if isinstance(reward_agent, float)
                    else reward_agent.item()
                )  # backwards compatibility (some rewards are floats, some are tensors)

                if self.train_both:
                    self.opponent_agent.train(
                        reward_opponent
                        if isinstance(reward_opponent, float)
                        else reward_opponent.item()
                    )  # backwards compatibility (some rewards are floats, some are tensors)

            self._logger.info(
                "Episode %10d: Total reward agent: %4.2f", i, reward_agent
            )
            self._logger.info(
                "Episode %10d: Total reward opponent: %4.2f", i, reward_opponent
            )
        return reward_agent, reward_opponent

    def evaluate(self, n_episodes: int) -> list[float]:
        """Evaluate the agent in the environment.

        Args:
            n_episodes (int): The number of episodes to evaluate.

        Returns:
            list[float]: A list of rewards received in each episode.
        """
        with self.agent.evaluate_context(), self.opponent_agent.evaluate_context():
            rewards = []
            rewards_opponent = []
            for i in range(n_episodes):
                self.reset()
                self.agent.reset()
                self.opponent_agent.reset()
                r_agent, r_opponent = self.run_eval()
                rewards.append(r_agent), rewards_opponent.append(r_opponent)
        if self.record:
            self.save_gif()
        return rewards, rewards_opponent
