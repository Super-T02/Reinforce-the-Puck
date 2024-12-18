import logging

import numpy as np
from agents.base_agent import BaseAgent
from environments.wrapper import EnvWrapper


def run_agent_on_environment(
    env: EnvWrapper, agent: BaseAgent, n_episodes=100
) -> tuple[list[any], list[any], list[float]]:
    """
    Runs a given agent on a specified environment for a number of episodes and collects observations, actions, and rewards.

    Args:
        env (EnvWrapper): The environment in which the agent will be run.
        agent (BaseAgent): The agent that will interact with the environment.
        n_episodes (int, optional): The number of episodes to run the agent. Defaults to 100.

    Returns:
        tuple[list[any], list[any], list[float]]: A tuple containing:
            - observations (list[any]): A list of states observed during the episodes.
            - actions (list[any]): A list of actions taken by the agent during the episodes.
            - rewards (list[float]): A list of total rewards received in each episode.
    """
    rewards = []
    observations = []
    actions = []
    for ep in range(1, n_episodes + 1):
        ep_reward = 0
        state, _info = env.reset()
        for t in range(2000):
            action = agent.act(state)
            state, reward, done, truncated, info = env.step(action)
            observations.append(state)
            actions.append(action)
            ep_reward += reward
            if done or truncated:
                break
        rewards.append(ep_reward)
        ep_reward = 0
    logging.getLogger(__name__).info(f"Mean reward: {np.mean(rewards)}")
    observations = np.asarray(observations)
    actions = np.asarray(actions)
    return observations, actions, rewards
