import argparse
import logging
import os

import numpy as np
import torch
from agents.base_agent import BaseAgent
from agents.ddpg import DDPGAgent
from agents.sac import SACAgent
from agents.td3 import TD3Agent
from environments.wrapper import EnvWrapper
from utils.config import AgentConfig, DDPGAgentConfig, SACAgentConfig, TD3AgentConfig


def run_agent_on_environment(
    env: EnvWrapper, n_episodes=100, max_steps=2000
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
    actions = np.asarray(actions)
    return observations, actions, rewards


def get_hidden_layer_sizes(state_dict):
    hidden_sizes = []
    for name, param in state_dict.items():
        if "weight" in name:
            hidden_sizes.append(param.shape[0])  # Output size (number of neurons)
    return hidden_sizes


def load_agent_config(path, agent_type) -> AgentConfig:
    """
    This fuctions loads an agent configuration by loading the checkpoint file and extracting the hidden layer sizes of the policy and critic networks.
    """
    checkpoint = torch.load(path)
    if agent_type == "SAC":
        q1 = checkpoint[0]
        q2 = checkpoint[1]
        policy = checkpoint[2]
        config = SACAgentConfig()
        actior_hidden_layer_sizes = get_hidden_layer_sizes(policy)
        critic_hidden_layer_sizes = get_hidden_layer_sizes(q1)
        config.actor_hidden_sizes = [
            actior_hidden_layer_sizes[0],
            actior_hidden_layer_sizes[1],
        ]
        config.critic_hidden_sizes = [
            critic_hidden_layer_sizes[0],
            critic_hidden_layer_sizes[1],
            critic_hidden_layer_sizes[2],
        ]
        return config

    elif agent_type == "TD3":
        q = checkpoint[0]
        policy = checkpoint[1]
        config = TD3AgentConfig()
        critic_hidden_layer_sizes = get_hidden_layer_sizes(q)
        actor_hidden_layer_sizes = get_hidden_layer_sizes(policy)
        config.actor_hidden_sizes = [
            actor_hidden_layer_sizes[0],
            actor_hidden_layer_sizes[1],
        ]
        config.critic_hidden_sizes = [
            critic_hidden_layer_sizes[0],
            critic_hidden_layer_sizes[1],
            critic_hidden_layer_sizes[2],
        ]
        return config
    elif agent_type == "DDPG":
        q = checkpoint[0]
        policy = checkpoint[1]
        config = DDPGAgentConfig()
        critic_hidden_layer_sizes = get_hidden_layer_sizes(q)
        actor_hidden_layer_sizes = get_hidden_layer_sizes(policy)
        config.actor_hidden_sizes = [
            actor_hidden_layer_sizes[0],
            actor_hidden_layer_sizes[1],
        ]
        config.critic_hidden_sizes = [
            critic_hidden_layer_sizes[0],
            critic_hidden_layer_sizes[1],
            critic_hidden_layer_sizes[2],
        ]
        return config


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

    args = parser.parse_args()

    AGENT_CLASSES = {"SAC": SACAgent, "TD3": TD3Agent, "DDPG": DDPGAgent}

    agent_config = load_agent_config(args.agent, args.agent_type)

    env = EnvWrapper(
        args.env,
        max_steps=1000,
        agent_class=AGENT_CLASSES[args.agent_type],
        do_render=True,
        kwargs_agent={"config": agent_config},
    )
    env.reset()
    env.render()

    try:
        env.agent.load(args.agent)
    except Exception as e:
        print(f"Error loading agent: {e}")
        print(
            "Hint: Make sure that the agent is trained for this environment. Otherwise the layer input shapes (action space, observation space) might not match."
        )
        exit(1)

    run_agent_on_environment(env, n_episodes=args.episodes, max_steps=args.max_steps)
