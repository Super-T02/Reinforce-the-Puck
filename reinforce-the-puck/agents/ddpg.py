"""
File: ddpg.py
Author: Tom Freudenmann
Content: This file contains the DDPG agent implementation.
"""

import os

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
        **kwargs,
    ):
        super().__init__(name, observation_space, action_space, config)

        self._config: DDPGAgentConfig = config

        # Action noise
        self._action_noise = OUNoise((self._action_n))

        # Create Q networks
        self.Q = self._create_q_net(
            1, config.trainer_config.learning_rate_critic, **kwargs
        )
        self.Q_target = self._create_q_net(1, **kwargs)

        # Create policy networks
        self.policy = self._create_policy_net(**kwargs)
        self.policy_target = self._create_policy_net(**kwargs)

        # Copy the networks and create optimizers
        self._copy_nets()
        self._create_optimizers()

        self.epoch = 0

    ### INITIALIZATION - START ###

    def _create_q_optim(self, q_net: QFunction) -> torch.optim.Optimizer:
        """Create the optimizer for the Q networks.

        Args:
            q_net (QFunction): The Q networks.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        return torch.optim.Adam(
            q_net.parameters(),
            lr=self._config.trainer_config.learning_rate_critic,
            betas=(
                self._config.trainer_config.beta1,
                self._config.trainer_config.beta2,
            ),
            eps=0.000001,
        )

    def _create_policy_optim(self) -> torch.optim.Optimizer:
        """Create the optimizer for the policy network.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        return torch.optim.Adam(
            self.policy.parameters(),
            lr=self._config.trainer_config.learning_rate_actor,
            eps=0.000001,
        )

    def _create_optimizers(self) -> None:
        """Create the optimizers for the networks."""
        self.Q_optimizer = self._create_q_optim(self.Q)
        self.policy_optimizer = self._create_policy_optim()

    def _create_q_net(self, out: int, lr: float = 0.0, **kwargs) -> QFunction:
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
            **kwargs,
        )

    def _policy_activation(self) -> callable:
        """Activation function for the policy network:

        `y = (tanh(x) + 1) * (high - low) / 2 + low`

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

    def _create_policy_net(self, **kwargs) -> Feedforward:
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
            **kwargs,
        )

    ### Target Networks - START ###

    def _copy_nets(self) -> None:
        """Copy the weights from the policy and Q networks to the target networks."""
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

    ###  - START ###

    def act(self, state) -> any:
        """Select an action based on the given state.
        If the agent is in training mode: Add noise to the action.

        Args:
            state: The current state of the environment.

        Returns:
            action: The selected action.
        """
        action = self._get_target_action(state)
        if self._mode in [AgentMode.TRAIN]:
            action += self._config.eps * self._action_noise()
        if type(action) == torch.Tensor:
            action = action.detach().numpy()
        action = np.clip(action, self._action_space.low, self._action_space.high)
        return action

    def _get_target_action(self, state) -> any:
        """Get the target action based on the given state.

        Args:
            state: The current state of the environment.

        Returns:
            action: The target action.
        """
        return self.policy_target.predict(state)

    ### SAVE/LOAD - START ###

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
        self._create_optimizers()

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
        self.restore_state(
            torch.load(
                path,
                weights_only=False,
                map_location=torch.device(self._config.specialized_config.device),
            )
        )

    ### TRAINING - START ###

    def train(self, last_reward=np.nan):
        """Train the agent. Update the target networks if necessary.

        Args:
            last_reward (float): The last reward received by the agent. Defaults to np.nan.
        """
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

        losses = {
            "loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
        }

        if self._config.buffer_type == "PER":
            losses["buffer/size"] = self._feedback_buffer._memory.size
            losses["buffer/beta"] = self._feedback_buffer._beta
            losses["buffer/alpha"] = self._feedback_buffer._alpha
            losses["buffer/total"] = self._feedback_buffer._memory.total()
            losses["buffer/max"] = self._feedback_buffer._memory.max()
        else:
            losses["buffer/size"] = float(self._feedback_buffer.size)

        return losses

    def _compute_target_q(self, batch: Batch) -> torch.Tensor:
        """Calculate the target Q values:

        `Q_target = r + gamma * (1 - done) * Q_target(s', pi_target(s'))`

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
        """Compute the Q loss:

        `Q_loss = Q(obs, act) - target_q`

        Args:
            batch (Batch): The training batch.

        Returns:
            torch.Tensor: The Q loss.
        """
        target_q = self._compute_target_q(batch)

        q_loss = self.Q.get_loss(
            torch.cat([batch.observations, batch.actions], dim=1), target_q
        )

        # Update the priorities in the buffer
        if self._config.buffer_type == "PER":
            q_loss = self._feedback_buffer.weight_loss(q_loss, batch.indices)
            self._feedback_buffer.update_priorities(
                batch.indices, target_q.cpu().numpy()
            )

        return q_loss

    def _optimize_critic(self, batch: Batch) -> torch.Tensor:
        """Optimize the Q network and handle the BPER buffer.

        Args:
            batch (Batch): The training batch.

        Returns:
            torch.Tensor: The critic loss.
        """
        q_loss = self._compute_q_loss(batch)

        # Update the priorities in the buffer
        if self._config.buffer_type == "BPER":
            self._feedback_buffer.update_priorities(
                batch.indices, batch.rewards.cpu().numpy()
            )
        elif self._config.buffer_type == "PER":
            self._feedback_buffer.anneal()

        # Backpropagate the loss
        self._backpropagate(q_loss)
        return q_loss

    def _backpropagate(self, loss: torch.Tensor) -> None:
        """Backpropagate the loss through the network.

        Args:
            loss (torch.Tensor): The loss to backpropagate.
        """
        self.Q_optimizer.zero_grad()
        loss.backward()
        self.Q_optimizer.step()

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

    ### REST - START ###

    def __del__(self):
        return super().__del__()
