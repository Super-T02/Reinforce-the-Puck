import logging
import os

import gymnasium as gym
from agents.base_agent import BaseAgent
from utils import model_dir


class BaseEnvWrapper:
    """
    A general-purpose environment wrapper for Gymnasium environments.
    """

    def __init__(
        self,
        env_name: str,
        max_steps: int,
        do_render: bool = False,
        agent: BaseAgent = None,
    ):
        """
        Initialize the environment wrapper.

        Args:
            env_name (str): Name of the Gymnasium environment.
            agent (callable): The agent class that interacts with the environment.
            kwargs_agent (dict): Keyword arguments for the agent.
            checkpoint (str): The path to the checkpoint file.
        """
        try:
            if do_render:
                self.env = gym.make(env_name, continuous=True, render_mode="human")
            else:
                self.env = gym.make(env_name, continuous=True)
        except:
            logging.error("Environment not compatible with the agent.")
            if do_render:
                self.env = gym.make(env_name, render_mode="human")
            else:
                self.env = gym.make(env_name)
        self.agent = agent
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._last_observation = (None, 0, False, False, {})
        self._logger = logging.getLogger(__name__)
        self._max_steps = max_steps
        self.name = env_name
        self.reset()

    @property
    def last_observation(self) -> tuple[any, float, bool, bool, dict[str, any]]:
        """
        Return the last observation from the environment.

        Returns:
            tuple[any, float, bool, bool, dict[str, any]]: The last observation from the environment.
        """
        return self._last_observation

    def step(self, save=True) -> tuple[any, float, bool, bool, dict[str, any]]:
        """
        Take an simulation step in the environment.

        Returns:
            tuple[any, float, bool, bool, dict[str, any]]: (next_state, reward, done, truncated, info)
        """
        state = self._last_observation[0]
        action = self.agent.act(state)
        self._last_observation = self.env.step(action)
        if save:
            self.agent.save_experience(state, action, *self._last_observation)
        return self._last_observation

    def reset(self):
        """
        Reset the environment and return the initial state.

        Returns:
            state: The initial state of the environment.
        """
        state, _ = self.env.reset()
        self._last_observation = (state, 0, False, False, {})
        return state

    def run_eval(self) -> float:
        """Run one episode of the environment.

        Returns:
            float: The total reward received in the episode.
        """
        done = False
        reward = 0
        for _ in range(self._max_steps):
            self.step()
            reward += self._last_observation[1]
            done = self._last_observation[2]
            if done:
                break
            self.render() if self._do_render else None
        self._logger.info("Episode finished. Total reward: %f", reward)
        return reward

    def run_train_episode(self, i: int) -> float:
        """Run a single episode of the training.

        Args:
            i (int): The episode number.

        Returns:
            float: The total reward of the episode.
        """
        with self.agent.train_context():
            reward = 0
            self.reset()
            self.agent.reset()
            done = False

            for _ in range(self._max_steps):
                self.step()
                done = self._last_observation[2]
                trunc = self._last_observation[3]
                reward += self._last_observation[1]
                if done or trunc:
                    break

            self.agent.train(
                reward if isinstance(reward, float) else reward.item()
            )  # backwards compatibility (some rewards are floats, some are tensors)
            self._logger.info("Episode %10d: Total reward: %4.2f", i, reward)
        return reward

    def evaluate(self, n_episodes: int) -> list[float]:
        """Evaluate the agent in the environment.

        Args:
            n_episodes (int): The number of episodes to evaluate.

        Returns:
            list[float]: A list of rewards received in each episode.
        """
        with self.agent.evaluate_context():
            rewards = []
            for i in range(n_episodes):
                self.reset()
                self.agent.reset()
                rewards.append(self.run_eval())
        return rewards

    def render(self):
        """Render the environment."""
        self.env.render()

    def close(self) -> "BaseEnvWrapper":
        """Close the environment.

        Returns:
            EnvWrapper: The environment wrapper object.
        """
        self.env.close()
        self._logger.info("Environment closed.")
        return self
