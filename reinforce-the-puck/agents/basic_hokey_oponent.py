"""
File: basic_hockey_opponent.py
Author: Jonathan Schwab and Tom Freudenmann
Content: This file contains the basic hockey opponent agent.
"""

from typing import List

import hockey.hockey_env as h_env
import numpy as np
from agents.base_agent import AgentMode, BaseAgent
from components.memory import Batch
from utils.config import AgentConfig


class BasicHokeyOpponentWrapper(BaseAgent):
    def __init__(self, weak=False):
        self.agent = h_env.BasicOpponent(weak=weak)
        self._mode = AgentMode.DEFAULT
        self._run_name = "BASIC_OPPONENT(WEAK)" if weak else "BASIC_OPPONENT(STRONG)"
        self._config = AgentConfig()
        self._config.type = "BasicOpponent_" + ("weak" if weak else "strong")
        self._config.name = self._run_name

    def act(self, observation):
        return self.agent.act(observation)

    def reset(self) -> "BaseAgent":
        return self

    def train_step(self, batch: Batch):
        pass  # Override this method to avoid exceptions, but pass because this agent is not trainable

    def train(self, last_reward: float = np.nan) -> List[dict]:
        pass  # Override this method to avoid exceptions, but pass because this agent is not trainable

    def save(self, path: str) -> None:
        pass  # Override this method to avoid exceptions, but pass because this agent is not trainable

    def save_experience(
        self,
        state: any,
        action: any,
        new_state: any,
        reward: float,
        done: bool,
        trunc: bool,
        info: dict[str, any],
    ) -> "BaseAgent":
        pass  # Override this method to avoid exceptions, but pass because this agent is not trainable

    def save_eval_result(self, rewards):
        pass  # Override this method to avoid exceptions, but pass because this agent is not trainable

    def save_checkpoint(self, checkpoint_name):
        pass  # Override this method to avoid exceptions, but pass because this agent is not trainable

    def save_statistics_async(self, statistics, filename):
        pass  # Override this method to avoid exceptions, but pass because this agent is not trainable
