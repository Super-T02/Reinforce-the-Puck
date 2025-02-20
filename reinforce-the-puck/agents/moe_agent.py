import os

import numpy as np
import torch
from agents.base_agent import BaseAgent
from agents.double_q_net import DoubleQLearningAgent
from agents.sac import SACAgent
from agents.td3 import TD3Agent
from components.memory import Batch, Memory
from gymnasium import spaces
from utils.config import AgentConfig, MoeAgentConfig, SACAgentConfig, TD3AgentConfig


class MOEAgent(BaseAgent):
    def __init__(
        self,
        config: MoeAgentConfig,
        agent_a: BaseAgent,
        agent_b: BaseAgent,
        observation_space: spaces.box.Box,
        action_space: spaces.box.Box,
    ):
        super().__init__("MOE", observation_space, action_space, config)
        self.agent_a = agent_a
        self.agent_b = agent_b

        self.router_agent = DoubleQLearningAgent(
            observation_space.shape[0],
            hidden_sizes=config.hidden_size,
            lr=config.trainer_config.learning_rate,
            gamma=config.gamma,
            device=self._config.specialized_config.device,
        )
        self.last_actor = "a"

    def act(self, state) -> any:
        action = self.router_agent.select_action(state)
        self.last_action = action
        if action == 0:
            self.last_actor = "a"
            with self.agent_a.evaluate_context():
                return self.agent_a.act(state)
        else:
            self.last_actor = "b"
            with self.agent_b.evaluate_context():
                return self.agent_b.act(state)

    def reset(self) -> "MOEAgent":
        """Reset the agent.

        Returns:
            SAC: The reset agent.
        """
        return self

    def train_step(self, batch: Batch) -> dict:
        """Train the agent.

        Args:
            batch (Batch): The batch of data.

        Returns:
            dict: The training metrics.
        """

        loss = self.router_agent.train_step(batch)
        loss["last_action"] = self.last_action

        with self.agent_a.train_context():
            agent_a_stats = self.agent_a.train_step(
                self.agent_a.sample(self._batch_size)
            )
        with self.agent_b.train_context():
            agent_b_stats = self.agent_b.train_step(
                self.agent_b.sample(self._batch_size)
            )

        # concat stats
        for key, value in agent_a_stats.items():
            loss[f"agent_a_{key}"] = value
        for key, value in agent_b_stats.items():
            loss[f"agent_b_{key}"] = value

        return loss

    def state(self) -> dict:
        return {
            "router_agent": {
                f"network_{i}": net_state
                for i, net_state in enumerate(self.router_agent.state())
            },
            "agent_a": {
                f"network_{i}": net_state
                for i, net_state in enumerate(self.agent_a.state())
            },
            "agent_b": {
                f"network_{i}": net_state
                for i, net_state in enumerate(self.agent_b.state())
            },
            "agent_a_type": self.agent_a._config.type.upper(),
            "agent_b_type": self.agent_b._config.type.upper(),
            "agent_a_config": self.agent_a._config.to_dict(),
            "agent_b_config": self.agent_b._config.to_dict(),
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state(), path)

    def load(self, path: str) -> None:
        state = torch.load(
            path,
            map_location=torch.device(self._config.specialized_config.device),
        )

        router_agent_states = tuple(
            state["router_agent"][f"network_{i}"]
            for i in range(len(state["router_agent"]))
        )

        self.router_agent.restore_state(router_agent_states)

        agent_type_mapping = {  # supported types
            "SAC": SACAgent,
            "TD3": TD3Agent,
        }
        agent_type_config_mapping = {
            "SAC": SACAgentConfig,
            "TD3": TD3AgentConfig,
        }

        agent_a_states = tuple(
            state["agent_a"][f"network_{i}"] for i in range(len(state["agent_a"]))
        )
        agent_a_type = state["agent_a_type"]
        agent_a_config = agent_type_config_mapping[agent_a_type]()
        agent_a_config.specialized_config.device = (
            self._config.specialized_config.device
        )
        agent_a_config.update_from_dict(state["agent_a_config"])
        self.agent_a = agent_type_mapping[agent_a_type](
            self._observation_space, self._action_space, agent_a_config
        )
        self.agent_a.restore_state(agent_a_states)

        agent_b_states = tuple(
            state["agent_b"][f"network_{i}"] for i in range(len(state["agent_b"]))
        )
        agent_b_type = state["agent_b_type"]
        agent_b_config = agent_type_config_mapping[agent_b_type]()
        agent_b_config.specialized_config.device = (
            self._config.specialized_config.device
        )
        agent_b_config.update_from_dict(state["agent_b_config"])
        self.agent_b = agent_type_mapping[agent_b_type](
            self._observation_space, self._action_space, agent_b_config
        )
        self.agent_b.restore_state(agent_b_states)
