import os

import numpy as np
import torch
from agents.base_agent import BaseAgent
from components.memory import Batch
from components.networks import Feedforward, QFunction
from components.noise import OUNoise
from gymnasium import spaces
from utils.config import AgentConfig, DDPGAgentConfig


class DDPGAgent(BaseAgent):
    """
    Agent implementing Q-learning with NN function approximation.
    """

    def __init__(
        self,
        trainer: callable,
        observation_space: spaces.box.Box,
        action_space: spaces.box.Box,
        config: DDPGAgentConfig,
    ):
        super().__init__("DDPG", trainer, observation_space, action_space, config)
        self._action_noise = OUNoise((self._action_n))

        self.Q = QFunction(
            input_size=self._obs_dim + self._action_n,
            output_size=1,
            hidden_sizes=config.critic_hidden_sizes,
            learning_rate=config.trainer_config.learning_rate_critic,
        )
        self.Q_target = QFunction(
            input_size=self._obs_dim + self._action_n,
            output_size=1,
            hidden_sizes=config.critic_hidden_sizes,
            learning_rate=0,
        )

        self.policy = self._create_policy_net()
        self.policy_target = self._create_policy_net()
        self._copy_nets()
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.trainer_config.learning_rate_actor,
            eps=0.000001,
        )
        self.Q_optimizer = torch.optim.Adam(
            self.Q.parameters(),
            lr=config.trainer_config.learning_rate_critic,
            eps=0.000001,
        )
        self.epoch = 0

    def _policy_activation(self) -> callable:
        """Activation function for the policy network.

        Returns:
            callable: The activation function.
        """
        high, low = torch.from_numpy(self._action_space.high), torch.from_numpy(
            self._action_space.low
        )
        output_activation = lambda x: (torch.nn.Tanh()(x) + 1) * (high - low) / 2 + low
        return output_activation

    def _create_policy_net(self) -> Feedforward:
        """Create the policy network.

        Returns:
            Feedforward: The policy network.
        """
        return Feedforward(
            input_size=self._obs_dim,
            hidden_sizes=self._config.actor_hidden_sizes,
            output_size=self._action_n,
            activation_fun=torch.nn.ReLU(),
            output_activation=self._policy_activation(),
        )

    def _copy_nets(self) -> None:
        """Copy the weights from the policy and Q networks to the target networks."""
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

    def act(self, state) -> any:
        """Select an action based on the given state.

        Args:
            state: The current state of the environment.

        Returns:
            action: The selected action.
        """
        action = self.policy.predict(state) + self._config.eps * self._action_noise()
        return action

    def state(self) -> tuple:
        """Get the state of the agent.

        Returns:
            tuple: The state of the agent.
        """
        return (self.Q.state_dict(), self.policy.state_dict())

    def restore_state(self, state: tuple) -> None:
        """Restore the state of the agent.

        Args:
            state (tuple): The state of the agent.
        """
        self.Q.load_state_dict(state[0])
        self.policy.load_state_dict(state[1])
        self._copy_nets()

    def reset(self) -> "DDPGAgent":
        """Reset the agent.

        Returns:
            DDPGAgent: The agent object.
        """
        self._action_noise.reset()
        return self

    def save(self, path: str) -> None:
        """Save the agent to a file.

        Args:
            path (str): The path to the file where the agent will be saved.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state(), path)

    def load(self, path: str) -> None:
        """Load the agent from a file.

        Args:
            path (str): The path to the file where the agent is saved.
        """
        self.restore_state(torch.load(path))

    def train_step(self, batch: Batch) -> dict:
        """Perform a single training step.

        Args:
            batch (Batch): The training batch.

        Returns:
            dict: The training statistics.
        """
        target_q = batch.rewards + self._config.discount * (
            1 - batch.dones
        ) * self.Q_target.Qvalues(
            batch.next_observations,
            self.policy_target.forward(batch.next_observations).detach(),
        )
        q_loss = self.Q.get_loss(
            torch.cat([batch.observations, batch.actions], dim=1), target_q
        )

        self.Q_optimizer.zero_grad()
        q_loss.backward()
        self.Q_optimizer.step()

        self.policy_optimizer.zero_grad()
        actions_pred = self.policy.forward(batch.observations)
        actor_loss = -self.Q.Qvalues(batch.observations, actions_pred).mean()
        actor_loss.backward()
        self.policy_optimizer.step()

        losses = {"loss": q_loss.item(), "actor_loss": actor_loss.item()}
        return losses
