import logging

import gymnasium as gym
import numpy as np
from agents.base_agent import BaseAgent
from utils.config import global_config


class EnvWrapper:
    """
    A general-purpose environment wrapper for Gymnasium environments.
    """

    def __init__(self, env_name: str, agent_class: BaseAgent, kwargs_agent: dict = {}):
        """
        Initialize the environment wrapper.

        Args:
            env_name (str): Name of the Gymnasium environment.
            agent (callable): The agent class that interacts with the environment.
            kwargs_agent (dict): Keyword arguments for the agent.
        """
        self.env = gym.make_vec(
            env_name,
            num_envs=global_config.environment.num_envs,
            vectorization_mode=global_config.environment.vectorization_mode,
        )
        self.agent = agent_class(
            **kwargs_agent,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space
        )
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._last_observations = (None, 0, False, False, {})
        self._logger = logging.getLogger(__name__)

        self.reset()

    @property
    def last_observation(self) -> tuple[any, float, bool, bool, dict[str, any]]:
        """
        Return the last observation from the environment.

        Returns:
            tuple[any, float, bool, bool, dict[str, any]]: The last observation from the environment.
        """
        return self._last_observations

    def reset(self) -> any:
        """
        Reset the environment and return the initial state.

        Returns:
            state: The initial state of the environment.
        """
        state, _ = self.env.reset()
        return state

    def step(self) -> tuple[any, float, bool, bool, dict[str, any]]:
        """
        Take an simulation step in the environment.

        Returns:
            tuple[any, float, bool, bool, dict[str, any]]: (next_state, reward, done, truncated, info)
        """
        states = self._last_observations[0]
        actions = self.agent.act(states)
        self._last_observations = self.env.step(actions)
        for i in range(global_config.environment.num_envs):
            self.agent.save_experience(
                states[i],
                actions[i],
                self._last_observations[0][i],
                self._last_observations[1][i],
                self._last_observations[2][i],
                self._last_observations[3][i],
                self._last_observations[4],
            )
        return self._last_observations

    def reset(self):
        """
        Reset the environment and return the initial state.

        Returns:
            state: The initial state of the environment.
        """
        state, _ = self.env.reset()
        self._last_observations = (state, 0, False, False, {})
        return state

    def run(self) -> float:
        """Run one episode of the environment.

        Returns:
            float: The total reward received in the episode.
        """
        dones = np.zeros(global_config.environment.num_envs, dtype=bool)
        truncs = np.zeros(global_config.environment.num_envs, dtype=bool)
        self._logger.info("Running one episode...")
        rewards = 0
        while not np.all(np.logical_or(dones, truncs)):
            self.step()
            dones = self._last_observations[2]
            truncs = self._last_observations[3]
            rewards += self._last_observations[1]
        reward = rewards.mean()
        self._logger.info("Episode finished. Total reward: %f", reward)
        return reward

    def run_train_episode(self, i: int) -> float:
        """Run a single episode of the training.

        Args:
            i (int): The episode number.

        Returns:
            float: The total reward of the episode.
        """
        rewards = np.zeros(global_config.environment.num_envs, dtype=np.float32)
        self.reset()
        self.agent.reset()
        dones = np.zeros(global_config.environment.num_envs, dtype=bool)
        for _ in range(global_config.environment.max_steps):
            self.step()
            dones = self._last_observations[2]
            truncs = self._last_observations[3]
            rewards += self._last_observations[1]
            if np.all(np.logical_or(dones, truncs)):
                break

        reward = rewards.mean()
        self.agent.train(reward)
        self._logger.info("Episode %10d: Total reward: %4.2f", i, reward)
        return rewards.mean()

    def close(self) -> "EnvWrapper":
        """Close the environment.

        Returns:
            EnvWrapper: The environment wrapper object.
        """
        self.env.close()
        self._logger.info("Environment closed.")
        return self
