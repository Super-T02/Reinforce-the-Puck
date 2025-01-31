import os

import torch
from agents.base_agent import BaseAgent
from agents.basic_hokey_oponent import BasicHokeyOpponentWrapper
from agents.ddpg import DDPGAgent
from agents.sac import SACAgent
from agents.td3 import TD3Agent
from agents.td3_cross import TD3CrossQAgent
from gymnasium import spaces
from matplotlib.pylab import f
from utils.config import (
    AgentConfig,
    DDPGAgentConfig,
    SACAgentConfig,
    TD3AgentConfig,
    TD3CrossAgentConfig,
    model_dir,
)


def _get_hidden_layer_sizes(state_dict):
    hidden_sizes = []
    for name, param in state_dict.items():
        if "weight" in name:
            hidden_sizes.append(param.shape[0])  # Output size (number of neurons)
    return hidden_sizes


def _build_checkpoint_path(checkpoint):
    return os.path.join(model_dir, checkpoint)


class AgentFactory:
    @staticmethod
    def create_agent_from_checkpoint(
        path,
        agent_type,
        observation_space: spaces.box.Box,
        action_space: spaces.box.Box,
        config=None,
    ) -> BaseAgent:
        """
        This function creates an agent from a checkpoint file.
        """
        if config is None and path is not None:
            config = AgentFactory.create_adapted_agent_config_from_checkpoint(
                path, agent_type
            )

        agent = AgentFactory.create_agent_from_config(
            config, observation_space, action_space
        )
        agent.load(path)
        return agent

    @staticmethod
    def create_agent_from_config(
        config: AgentConfig,
        observation_space: spaces.box.Box,
        action_space: spaces.box.Box,
    ) -> BaseAgent:
        """
        This function creates an agent from a configuration object.
        """
        agent = None
        if isinstance(config, SACAgentConfig):
            agent = SACAgent(
                config=config,
                action_space=action_space,
                observation_space=observation_space,
            )
        elif isinstance(config, TD3CrossAgentConfig):
            agent = TD3CrossQAgent(
                config=config,
                action_space=action_space,
                observation_space=observation_space,
            )
        elif isinstance(config, TD3AgentConfig):
            agent = TD3Agent(
                config=config,
                action_space=action_space,
                observation_space=observation_space,
            )
        elif isinstance(config, DDPGAgentConfig):
            agent = DDPGAgent(
                config=config,
                action_space=action_space,
                observation_space=observation_space,
            )
        elif isinstance(config, AgentConfig):
            if config.type == "basic_opponent_weak":
                return BasicHokeyOpponentWrapper(weak=True)
            elif config.type == "basic_opponent_strong":
                return BasicHokeyOpponentWrapper(weak=False)
        else:
            raise ValueError("Invalid agent configuration type")
        if config.checkpoint is not None:
            path = _build_checkpoint_path(config.checkpoint)
            agent.load(path)
        return agent

    @staticmethod
    def create_adapted_agent_config_from_checkpoint(path, agent_type) -> AgentConfig:
        """
        This fuctions loads an agent configuration by loading the checkpoint file and extracting the hidden layer sizes of the policy and critic networks.
        """
        checkpoint = torch.load(path, weights_only=False)
        if agent_type.upper() == "SAC":
            q1 = checkpoint[0]
            q2 = checkpoint[1]
            policy = checkpoint[2]
            config = SACAgentConfig()
            actior_hidden_layer_sizes = _get_hidden_layer_sizes(policy)
            actior_hidden_layer_sizes = actior_hidden_layer_sizes[:-2]
            critic_hidden_layer_sizes = _get_hidden_layer_sizes(q1)
            config.actor_hidden_sizes = [*actior_hidden_layer_sizes[:-1]]
            config.critic_hidden_sizes = [*critic_hidden_layer_sizes[:-1]]

            return config

        elif agent_type.upper() == "TD3":
            q = checkpoint[0]
            policy = checkpoint[1]
            config = TD3AgentConfig()
            critic_hidden_layer_sizes = _get_hidden_layer_sizes(q)
            actor_hidden_layer_sizes = _get_hidden_layer_sizes(policy)
            config.actor_hidden_sizes = [*actor_hidden_layer_sizes[:-1]]
            config.critic_hidden_sizes = [*critic_hidden_layer_sizes[:-1]]
            return config
        elif agent_type.upper() == "DDPG":
            q = checkpoint[0]
            policy = checkpoint[1]
            config = DDPGAgentConfig()
            critic_hidden_layer_sizes = _get_hidden_layer_sizes(q)
            actor_hidden_layer_sizes = _get_hidden_layer_sizes(policy)
            config.actor_hidden_sizes = [*actor_hidden_layer_sizes[:-1]]
            config.critic_hidden_sizes = [*critic_hidden_layer_sizes[:-1]]
            return config
        elif agent_type.upper() == "TD3_CROSS":
            q = checkpoint[0]
            policy = checkpoint[1]
            config = TD3CrossAgentConfig()
            critic_hidden_layer_sizes = _get_hidden_layer_sizes(q)
            actor_hidden_layer_sizes = _get_hidden_layer_sizes(policy)
            important_critic = (len(critic_hidden_layer_sizes) - 1) // 2
            important_actor = (len(actor_hidden_layer_sizes) - 1) // 2
            config.actor_hidden_sizes = [
                *actor_hidden_layer_sizes[: -important_actor - 1]
            ]
            config.critic_hidden_sizes = [
                *critic_hidden_layer_sizes[: -important_critic - 1]
            ]
            return config
        else:
            config = AgentConfig()
            config.type = agent_type
            return config
