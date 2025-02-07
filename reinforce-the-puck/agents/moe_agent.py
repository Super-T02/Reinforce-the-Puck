import numpy as np
from agents.base_agent import BaseAgent
from agents.double_q_net import DoubleQLearningAgent
from agents.sac import SACAgent
from components.memory import Batch, Memory
from gymnasium import spaces
from utils.config import MueAgentConfig, SACAgentConfig


class MOEAgent(BaseAgent):
    def __init__(
        self,
        config: MueAgentConfig,
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

    def save(self, path):
        pass  # todo

    def act(self, state) -> any:
        action = self.router_agent.select_action(state)
        self.last_action = action
        if action == 0:
            with self.agent_a.evaluate_context():
                return self.agent_a.act(state)
        else:
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
        return loss
