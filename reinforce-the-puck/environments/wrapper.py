import logging

import gymnasium as gym
from agents.base_agent import BaseAgent
from utils.config import global_config


class EnvWrapper:
    """
    A general-purpose environment wrapper for Gymnasium environments.
    """

    def __init__(
        self,
        env_name: str,
        agent_class: BaseAgent,
        max_steps: int,
        kwargs_agent: dict = {},
    ):
        """
        Initialize the environment wrapper.

        Args:
            env_name (str): Name of the Gymnasium environment.
            agent (callable): The agent class that interacts with the environment.
            kwargs_agent (dict): Keyword arguments for the agent.
        """
        self.env = gym.make(env_name)
        self.agent = agent_class(
            **kwargs_agent,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space
        )
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._last_observation = (None, 0, False, False, {})
        self._logger = logging.getLogger(__name__)
        self._max_steps = max_steps
        print("Running on env: ", env_name)
        self.reset()

    @property
    def last_observation(self) -> tuple[any, float, bool, bool, dict[str, any]]:
        """
        Return the last observation from the environment.

        Returns:
            tuple[any, float, bool, bool, dict[str, any]]: The last observation from the environment.
        """
        return self._last_observation

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
        state = self._last_observation[0]
        action = self.agent.act(state)
        self._last_observation = self.env.step(action)
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

    def run(self) -> float:
        """Run one episode of the environment.

        Returns:
            float: The total reward received in the episode.
        """
        done = False
        self._logger.info("Running one episode...")
        reward = 0
        while not done:
            self.step()
            done = self._last_observation[2]
            reward += self._last_observation[1]
        self._logger.info("Episode finished. Total reward: %f", reward)
        return reward

    def run_train_episode(self, i: int) -> float:
        """Run a single episode of the training.

        Args:
            i (int): The episode number.

        Returns:
            float: The total reward of the episode.
        """
        reward = 0
        self.reset()
        self.agent.reset()
        done = False
        for i in range(self._max_steps):
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

    def close(self) -> "EnvWrapper":
        """Close the environment.

        Returns:
            EnvWrapper: The environment wrapper object.
        """
        self.env.close()
        self._logger.info("Environment closed.")
        return self
