import argparse
import logging
import os

import numpy as np
from agents.agent_factory import AgentFactory
from environments.base_wrapper import BaseEnvWrapper
from environments.environment_factory import EnvironmentFactory


def run_agent_on_environment(
    env: BaseEnvWrapper, n_episodes=100, max_steps=2000
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
    for ep in range(1, n_episodes + 1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_steps):
            state, reward, done, truncated, info = env.step()
            observations.append(state)
            ep_reward += reward
            if done or truncated:
                break
        rewards.append(ep_reward)
        ep_reward = 0
    logging.getLogger(__name__).info(f"Mean reward: {np.mean(rewards)}")
    observations = np.asarray(observations)
    # actions = np.asarray(actions)
    actions = []
    return observations, actions, rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI zur Ausführung eines Agenten in einer bestimmten Umgebung."
    )

    parser.add_argument(
        "--env",
        type=str,
        required=False,
        default="LunarLander-v3",
        help="name of the environment to run the agent in.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        # default="models/checkpoints/checkpoint_2025-01-11_10-51-53_agent_ddpg_0-steps.pth",
        help="path to the agent c heckpoint file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        required=False,
        default=1000,
        help="number of episodes to run the agent.",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        required=True,
        # default="DDPG",
        help="agent type (SAC/TD3/DDPG).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        required=False,
        default=1000,
        help="maximum number of steps per episode.",
    )
    parser.add_argument(
        "--mode",
        type=int,
        required=False,
        default=None,
        help="Mode for the hockey environment (0:=NORMAL, 1:=Shooting, 2:=Defense).",
    )

    args = parser.parse_args()

    env = EnvironmentFactory.create_environment(
        args.env, max_steps=1000, do_render=True, mode=args.mode
    )

    try:
        env.agent = AgentFactory.create_agent_from_checkpoint(
            args.agent, args.agent_type, env.observation_space, env.action_space
        )
        env.reset()
        env.render()
    except Exception as e:
        print(f"Error loading agent: {e}")
        print(
            "Hint: Make sure that the agent is trained for this environment. Otherwise the layer input shapes (action space, observation space) might not match."
        )
        exit(1)

    run_agent_on_environment(env, n_episodes=args.episodes, max_steps=args.max_steps)
