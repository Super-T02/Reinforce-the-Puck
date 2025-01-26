import logging
import os

import hockey.hockey_env as h_env
import imageio
import numpy as np
from agents.base_agent import BaseAgent
from environments.advanced_reward_calculator import AdaptiveHockeyRewardCalculator
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
        winner_weight: float = 100.0,
        closeness_puck_weight: float = 0.5,
        touch_puck_weight: float = 0.0,
        puck_direction_weight: float = 1.0,
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
        self.winner_weight = winner_weight
        self.closeness_puck_weight = closeness_puck_weight
        self.touch_puck_weight = touch_puck_weight
        self.puck_direction_weight = puck_direction_weight
        self.record = False
        self.record_path = os.path.join(workspace_dir, "gifs")
        self.frames = []
        self.name = "Hockey-v0"

        self.reward_calculator = AdaptiveHockeyRewardCalculator()

        self.reset()

    def reset(self):
        super().reset()
        state = self.env.obs_agent_two()
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
            self.frames = []

    def step(self, save=True):
        # Get states
        state_agent = self._last_observation[0]
        state_opponent = self._last_opponent_observation[0]

        # Get actions
        action1 = np.array(self.agent.act(state_agent))
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
            self.agent.save_experience(state_agent, action1, *self._last_observation)
            self.opponent_agent.save_experience(
                state_opponent, action2, *self._last_opponent_observation
            )

        return self._last_observation

    def compute_reward_agent(self, obs, r, d, t, info, evaluate=False) -> float:
        return (
            self.reward_calculator.compute_reward(obs, info)
            if not evaluate
            else info.get("winner", 0.0) * self.winner_weight
        )

    def _generic_reward(self, info) -> float:
        winner = info.get("winner", 0.0)  # 0: tie, -1: opponent, 1: agent
        closeness_puck = info.get("reward_closeness_to_puck", 0.0)
        touch_puck = info.get("reward_touch_puck", 0.0)
        puck_direction = info.get("reward_puck_direction", 0.0)
        r = (
            winner * self.winner_weight
            + closeness_puck * self.closeness_puck_weight
            + touch_puck * self.touch_puck_weight
            + puck_direction * self.puck_direction_weight
        )
        return r

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
        return reward_agent

    def run_train_episode(self, i: int) -> float:
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
                if done or trunc:
                    break

            self.agent.train(
                reward_agent if isinstance(reward_agent, float) else reward_agent.item()
            )  # backwards compatibility (some rewards are floats, some are tensors)
            # self.opponent_agent.train(
            #     reward_opponent
            #     if isinstance(reward_opponent, float)
            #     else reward_opponent.item()
            # )  # backwards compatibility (some rewards are floats, some are tensors)

            self._logger.info(
                "Episode %10d: Total reward agent: %4.2f", i, reward_agent
            )
            self._logger.info(
                "Episode %10d: Total reward opponent: %4.2f", i, reward_opponent
            )

        # todo: return reward_opponent (not compatible with the current implementation yet)
        return reward_agent

    def evaluate(self, n_episodes: int) -> list[float]:
        """Evaluate the agent in the environment.

        Args:
            n_episodes (int): The number of episodes to evaluate.

        Returns:
            list[float]: A list of rewards received in each episode.
        """
        with self.agent.evaluate_context(), self.opponent_agent.evaluate_context():
            rewards = []
            for i in range(n_episodes):
                self.reset()
                self.agent.reset()
                self.opponent_agent.reset()
                rewards.append(self.run_eval())
        if self._do_render and self.record:
            self.save_gif()
        return rewards
