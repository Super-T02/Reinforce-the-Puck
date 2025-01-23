import os

import torch
from agents.base_agent import BaseAgent
from agents.basic_hokey_oponent import BasicHokeyOpponentWrapper
from agents.ddpg import DDPGAgent
from agents.sac import SACAgent
from agents.td3 import TD3Agent
from gymnasium import spaces
from matplotlib.pylab import f
from utils.config import (
    AgentConfig,
    DDPGAgentConfig,
    OpponentConfig,
    SACAgentConfig,
    TD3AgentConfig,
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
    ) -> BaseAgent:
        """
        This function creates an agent from a checkpoint file.
        """
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
        else:
            raise ValueError("Invalid agent configuration type")
        if config.checkpoint is not None:
            path = _build_checkpoint_path(config.checkpoint)
            agent.load(path)
        return agent

    @staticmethod
    def create_opponent_agent(
        opponent_config: OpponentConfig,
        observation_space: spaces.box.Box,
        action_space: spaces.box.Box,
    ):
        cfg = opponent_config.to_dict()
        if opponent_config.type == "basic_opponent":
            return BasicHokeyOpponentWrapper(cfg.get("weak", False))
        elif opponent_config.checkpoint is not None:
            return AgentFactory.create_agent_from_checkpoint(
                _build_checkpoint_path(cfg.get("checkpoint")),
                cfg.get("type"),
                observation_space,
                action_space,
            )
        else:
            raise ValueError("Invalid opponent configuration")

    @staticmethod
    def create_adapted_agent_config_from_checkpoint(path, agent_type) -> AgentConfig:
        """
        This fuctions loads an agent configuration by loading the checkpoint file and extracting the hidden layer sizes of the policy and critic networks.
        """
        checkpoint = torch.load(path)
        if agent_type == "SAC":
            q1 = checkpoint[0]
            q2 = checkpoint[1]
            policy = checkpoint[2]
            config = SACAgentConfig()
            actior_hidden_layer_sizes = _get_hidden_layer_sizes(policy)
            critic_hidden_layer_sizes = _get_hidden_layer_sizes(q1)
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
            critic_hidden_layer_sizes = _get_hidden_layer_sizes(q)
            actor_hidden_layer_sizes = _get_hidden_layer_sizes(policy)
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
            critic_hidden_layer_sizes = _get_hidden_layer_sizes(q)
            actor_hidden_layer_sizes = _get_hidden_layer_sizes(policy)
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
