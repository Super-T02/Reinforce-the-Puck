import logging
import queue
import threading
import time

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
            action_space=self.env.action_space,
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
        rewards = 0
        for _ in range(global_config.environment.max_steps):
            self.step()
            dones = self._last_observations[2]
            truncs = self._last_observations[3]
            rewards += self._last_observations[1]
            if np.all(np.logical_or(dones, truncs)):
                break
        reward = rewards.mean()
        return reward

    def run_train_episode(self, i: int) -> float:
        """Run a single episode of the training.

        Args:
            i (int): The episode number.

        Returns:
            float: The total reward of the episode.
        """
        self.reset()
        self.agent.reset()
        reward = self.run()
        self.agent.train(reward)
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


class VectorizedEnvWrapper:
    """
    A vectorized environment wrapper to run multiple environments in parallel for a fixed duration.
    """

    def __init__(
        self,
        env_name: str,
        agent_class: BaseAgent,
        kwargs_agent: dict = {},
    ):
        """
        Initialize the vectorized environment wrapper.

        Args:
            env_name (str): Name of the Gymnasium environment.
            agent_class (BaseAgent): The agent class to use for all environments.
            num_envs (int): Number of environments to run in parallel.
            run_time (float): Time (in seconds) to run all environments.
            kwargs_agent (dict): Keyword arguments for the agent.
        """
        self.env_name = env_name
        self.num_envs = global_config.environment.num_envs
        self.run_time = global_config.environment.run_time
        self.envs = [gym.make(self.env_name) for _ in range(self.num_envs)]
        self.agent = agent_class(
            **kwargs_agent,
            observation_space=self.envs[0].observation_space,
            action_space=self.envs[0].action_space,
        )
        self.reward = []
        self.agent_lock = threading.Lock()
        self.rew_lock = threading.Lock()
        self._logger = logging.getLogger(__name__)

    def reset(self):
        """
        Reset all environments.
        """
        self.agent.reset()
        self.reward = []

    def _run_env(self, env: gym.Env, thread_id: int):
        """
        Run a single environment for the specified duration and collect experiences.

        Args:
            env (gym.Env): The environment to run.
            thread_id (int): The ID of the thread running this environment.
        """
        self._logger.debug(f"Thread {thread_id}: Starting environment.")
        start_time = time.time()
        while time.time() - start_time < self.run_time:
            done, trunc = False, False
            state, _ = env.reset()
            r = 0
            for _ in range(global_config.environment.max_steps):
                with self.agent_lock:
                    action = self.agent.act(state)
                new_state, reward, done, trunc, info = env.step(action)
                observation = (state, action, new_state, reward, done, trunc, info)
                with self.agent_lock:
                    self.agent.save_experience(*observation)
                r += reward
                if done or trunc:
                    break
            with self.rew_lock:
                self.reward.append(r)
        self._logger.debug(f"Thread {thread_id}: Finished running environment.")

    def run(self):
        """
        Run all environments in parallel for the specified duration and collect experiences.

        Returns:
            list: A list of all experiences collected from all environments.
        """
        self.reset()
        threads = []
        for i, env in enumerate(self.envs):
            thread = threading.Thread(target=self._run_env, args=(env, i))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        self._logger.debug("All environments finished. Collected experiences.")
        return sum(self.reward) / len(self.reward), len(self.reward)

    def run_train_episode(self, i: int) -> float:
        """
        Run a single episode of the training.

        Args:
            i (int): The episode number.

        Returns:
            float: The total reward of the episode.
        """
        self.reset()
        reward, num_iterations = self.run()
        self.agent.train(reward)
        self._logger.info(
            "Episode %10d: Total reward: %4.2f Simulation Its: %2d",
            i,
            reward,
            num_iterations,
        )
        return reward

    def close(self):
        """
        Close all environments.
        """
        for env in self.envs:
            env.close()
        self._logger.info("All environments closed.")
