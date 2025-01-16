import os
from itertools import chain

import numpy as np
import torch
from agents.base_agent import AgentMode, BaseAgent
from components.memory import Batch
from components.networks import Feedforward, QFunction
from components.noise import OUNoise
from gymnasium import spaces
from utils.config import DDPGAgentConfig


class DDPGAgent(BaseAgent):
    """
    Agent implementing Q-learning with NN function approximation.
    """

    def __init__(
        self,
        observation_space: spaces.box.Box,
        action_space: spaces.box.Box,
        config: DDPGAgentConfig,
        name: str = "DDPG",
    ):
        super().__init__(name, observation_space, action_space, config)
        self._config: DDPGAgentConfig = config
        self._action_noise = OUNoise((self._action_n))

        self.Q = self._create_q_net(1, config.trainer_config.learning_rate_critic)
        self.Q_target = self._create_q_net(1)

        self.policy = self._create_policy_net()
        self.policy_target = self._create_policy_net()
        self._copy_nets()
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.trainer_config.learning_rate_actor,
            eps=0.000001,
        )
        self.Q_optimizer = self._create_q_optim([self.Q])
        self.epoch = 0

    def _create_q_optim(self, q_nets: list[QFunction]) -> torch.optim.Optimizer:
        """Create the optimizer for the Q networks.

        Args:
            q_nets (list[QFunction]): The Q networks.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        return torch.optim.Adam(
            chain(*[q.parameters() for q in q_nets]),
            lr=self._config.trainer_config.learning_rate_critic,
            eps=0.000001,
        )

    def _create_q_net(self, out: int, lr: float = 0.0) -> QFunction:
        """Create the Q network.

        Args:
            out (int): The output size of the network.
            lr (float): The learning rate of the network

        Returns:
            QFunction: The Q network.
        """
        return QFunction(
            input_size=self._obs_dim + self._action_n,
            output_size=out,
            hidden_sizes=self._config.critic_hidden_sizes,
            learning_rate=lr,
            device=self._config.specialized_config.device,
        )

    def _policy_activation(self) -> callable:
        """Activation function for the policy network.

        Returns:
            callable: The activation function.
        """
        high, low = torch.from_numpy(self._action_space.high).to(
            self._config.specialized_config.device
        ), torch.from_numpy(self._action_space.low).to(
            self._config.specialized_config.device
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
            device=self._config.specialized_config.device,
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
        action = self.policy_target.predict(state)
        if self._mode in [AgentMode.TRAIN]:
            action += self._config.eps * self._action_noise()
        if type(action) == torch.Tensor:
            action = action.detach().numpy()
        action = np.clip(action, self._action_space.low, self._action_space.high)
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

    def train(self, last_reward=np.nan):
        """Train the agent."""
        if self._mode != AgentMode.TRAIN:
            raise ValueError("Agent is not in training mode.")

        d = self._config.update_target_every

        if d > 0 and self._train_iterations % d == 0:
            self._copy_nets()
        return super().train(last_reward)

    def train_step(self, batch: Batch) -> dict:
        """Perform a single training step.

        Args:
            batch (Batch): The training batch.

        Returns:
            dict: The training statistics.
        """
        if self._mode != AgentMode.TRAIN:
            raise ValueError("Agent is not in training mode.")
        critic_loss = self._optimize_critic(batch)
        actor_loss = self._optimize_actor(batch)

        losses = {"loss": critic_loss.item(), "actor_loss": actor_loss.item()}
        return losses

    def _compute_target_q(self, batch: Batch) -> torch.Tensor:
        """Calculate the target Q values.

        Args:
            batch (Batch): The training batch.

        Returns:
            torch.Tensor: The target Q values.
        """
        return batch.rewards + self._config.discount * (
            1 - batch.dones
        ) * self.Q_target.Qvalues(
            batch.next_observations,
            self.policy_target.forward(batch.next_observations).detach(),
        )

    def _compute_q_loss(self, batch: Batch) -> torch.Tensor:
        """Compute the Q loss.

        Args:
            batch (Batch): The training batch.

        Returns:
            torch.Tensor: The Q loss.
        """
        target_q = self._compute_target_q(batch)
        q_loss = self.Q.get_loss(
            torch.cat([batch.observations, batch.actions], dim=1), target_q
        )
        return q_loss

    def _optimize_critic(self, batch: Batch) -> torch.Tensor:
        """Optimize the Q network.

        Args:
            batch (Batch): The training batch.

        Returns:
            torch.Tensor: The critic loss.
        """
        q_loss = self._compute_q_loss(batch)

        self.Q_optimizer.zero_grad()
        q_loss.backward()
        self.Q_optimizer.step()
        return q_loss

    def _optimize_actor(self, batch: Batch) -> torch.Tensor:
        """Optimize the policy network.

        Args:
            batch (Batch): The training batch.

        Returns:
            torch.Tensor: The actor loss.
        """
        self.policy_optimizer.zero_grad()
        actions_pred = self.policy.forward(batch.observations)
        actor_loss = -self.Q.Qvalues(batch.observations, actions_pred).mean()
        actor_loss.backward()
        self.policy_optimizer.step()
        return actor_loss

    def __del__(self):
        return super().__del__()
