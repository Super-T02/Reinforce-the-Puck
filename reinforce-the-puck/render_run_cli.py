import argparse
import datetime
import logging
import os

import numpy as np
from agents.agent_factory import AgentFactory
from environments.advanced_reward_calculator import AdaptiveHockeyRewardCalculator
from environments.base_wrapper import BaseEnvWrapper
from environments.environment_factory import EnvironmentFactory
from utils import workspace_dir
from utils.config import AgentConfig


def find_newest_file(directory):
    newest_file = None
    newest_time = 0

    # Durchlaufe das Verzeichnis und alle Unterverzeichnisse
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_time = os.path.getmtime(file_path)

            # Überprüfe, ob diese Datei neuer ist
            if file_time > newest_time:
                newest_time = file_time
                newest_file = file_path
    return newest_file


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
    rewards = env.evaluate(n_episodes)
    return rewards


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
        required=False,
        default=find_newest_file(os.path.join(workspace_dir, "models")),
        help="path to the agent checkpoint file.",
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
    parser.add_argument(
        "--opponent-checkpoint",
        type=str,
        required=False,
        default=None,
        help="Path to the opponent agent checkpoint file.",
    )
    parser.add_argument(
        "--opponent-type",
        type=str,
        required=False,
        default=None,
        help="Type of the opponent agent (SAC/TD3/DDPG/BASIC_OPPONENT_WEAK/BASIC_OPPONENT_STRONG).",
    )
    parser.add_argument("--gif", action="store_true", help="Create a gif of the run.")

    args = parser.parse_args()
    opponent_config = AgentConfig()
    if args.opponent_checkpoint is not None:
        opponent_config.checkpoint = args.opponent_checkpoint
    if args.opponent_type is not None:
        opponent_config.type = args.opponent_type

    env = EnvironmentFactory.create_environment(
        args.env, max_steps=1500, do_render=True, mode=args.mode
    )

    env.agent = AgentFactory.create_agent_from_checkpoint(
        args.agent, args.agent_type, env.observation_space, env.action_space
    )
    if env.name == "Hockey-v0":
        if opponent_config.checkpoint is not None:
            env.opponent_agent = AgentFactory.create_agent_from_checkpoint(
                opponent_config.checkpoint,
                opponent_config.type,
                env.observation_space,
                env.action_space,
            )
        else:
            env.opponent_agent = AgentFactory.create_agent_from_config(
                opponent_config, env.observation_space, env.action_space
            )
        print("Loaded opponent agent: ", env.opponent_agent)
    env.reset()
    if args.gif:
        env.record = True
        base = os.path.join(workspace_dir, "gifs")
        os.makedirs(base, exist_ok=True)
        env.record_path = os.path.join(
            base, f"{env.name}_{datetime.datetime.now()}.gif"
        )
    env.render()

    run_agent_on_environment(env, n_episodes=args.episodes, max_steps=args.max_steps)
