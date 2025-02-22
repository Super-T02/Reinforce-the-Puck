"""
File: td3.py
Author: Tom Freudenmann
Content: This file contains the Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.
"""

import numpy as np
import torch
from agents.base_agent import AgentMode
from agents.ddpg import DDPGAgent
from components.memory import Batch
from components.networks import Feedforward
from components.noise import ClippedColoredNoise, ColoredNoise
from utils.config import TD3AgentConfig, global_config


class TD3Agent(DDPGAgent):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.
    """

    def __init__(
        self,
        observation_space: tuple,
        action_space: tuple,
        config: TD3AgentConfig,
        name: str = "TD3",
        **kwargs
    ):
        self.Q2, self.Q2_target = None, None
        super().__init__(observation_space, action_space, config, name, **kwargs)

        # Exploration noise
        self._action_noise = ColoredNoise(
            (self._action_n,), config.noise_sigma, config.noise_beta
        )

        # Smooths the target policy
        self._target_smoothing_noise = ClippedColoredNoise(
            (self._action_n,), config.noise_sigma, 0, config.noise_clip
        )  # Gaussian noise

        # Create 2nd Q network
        self.Q2 = self._create_q_net(
            1, config.trainer_config.learning_rate_critic, **kwargs
        )
        self.Q2_target = self._create_q_net(1, **kwargs)

        self._last_actor_loss = torch.nan
        self._copy_nets()
        self._create_optimizers()

    ### INITIALIZATION - START ###

    def _create_optimizers(self):
        """Create the optimizers for the networks."""
        super()._create_optimizers()
        if self.Q2 is not None:
            self.Q2_optimizer = self._create_q_optim(self.Q2)

    ### SAVE/LOAD - START ###

    def restore_state(self, state):
        """Restore the agent state.

        Args:
            state (State Dict): The state to restore.
        """
        self.Q2.load_state_dict(state[2])
        super().restore_state(state)

    def state(self):
        """Return the agent state.

        Returns:
            tuple: The agent state.
        """
        return (*super().state(), self.Q2.state_dict())

    ### TARGET NETWORKS - START ###

    def _copy_nets(self):
        """Copy the networks to the target networks."""
        super()._copy_nets()
        if self.Q2 is not None:
            self.Q2_target.load_state_dict(self.Q2.state_dict())

    def _update_target_nets(self):
        """Update all target networks with a soft update."""
        pairs = [
            (self.Q_target, self.Q),
            (self.policy_target, self.policy),
            (self.Q2_target, self.Q2),
        ]
        for target, source in pairs:
            self._update_target_net(target, source)

    def _update_target_net(self, target: Feedforward, source: Feedforward):
        """Update the target network. The update rule is:

        `target = tao * source + (1 - tao) * target`

        Args:
            target (Feedforward): The target network.
            source (Feedforward): The source network.
            tao (float): The tao value.
        """
        tao = self._config.tao
        result = target.state_dict()
        for target_param, source_param, name in zip(
            target.parameters(), source.parameters(), result.keys()
        ):
            result[name] = tao * source_param + (1 - tao) * target_param
        target.load_state_dict(result)

    ### TRAINING - START ###

    def train_step(self, batch):
        """Perform a single training step.

        1. Optimize the critic.
        2. If it's time to update the actor, optimize the actor:
            - Compute the actor loss.
            - Update the target networks.
        3. Return the training statistics.

        Args:
            batch (Batch): The training batch.

        Returns:
            dict: The training statistics.
        """
        if self._mode != AgentMode.TRAIN:
            raise ValueError("Agent is not in training mode.")
        critic_loss = self._optimize_critic(batch)

        # Update the target networks
        if self._train_iterations % self._config.policy_delay == 0:
            actor_loss = self._optimize_actor(batch)
            self._update_target_nets()
            self._last_actor_loss = actor_loss.item()

        losses = {
            "loss": critic_loss.item(),
            "actor_loss": self._last_actor_loss,
        }

        if self._config.buffer_type == "PER":
            losses["buffer/size"] = self._feedback_buffer._memory.size
            losses["buffer/beta"] = self._feedback_buffer._beta
            losses["buffer/alpha"] = self._feedback_buffer._alpha
            losses["buffer/total"] = self._feedback_buffer._memory.total()
            losses["buffer/max"] = self._feedback_buffer._memory.max()
        else:
            losses["buffer/size"] = self._feedback_buffer.size

        return losses

    def _compute_q_loss(self, batch):
        """Calculate the critic's loss:

        1. Compute the target Q values.
        2. q_1_loss = Q1(obs, act) - target_q
        3. q_2_loss = Q2(obs, act) - target_q
        4. Return q_1_loss + q_2_loss

        Args:
            batch (Batch): The training batch.

        Returns:
            torch.Tensor: The target Q values.
        """
        target_q = self._compute_target_q(batch)
        x = torch.cat((batch.observations, batch.actions), dim=1)
        q1_loss = self.Q.get_loss(x, target_q)
        q2_loss = self.Q2.get_loss(x, target_q)
        q_loss = q1_loss + q2_loss

        # Update the priorities in the buffer
        if self._config.buffer_type == "PER":
            q_loss = self._feedback_buffer.weight_loss(q_loss, batch.indices)
            self._feedback_buffer.update_priorities(
                batch.indices,
                target_q.cpu().detach().numpy(),
                batch.rewards.cpu().detach().numpy(),
            )
        return q_loss

    def _backpropagate(self, loss):
        """Backpropagate the loss through the network.

        Args:
            loss (Loss): The loss to backpropagate.
        """
        self.Q_optimizer.zero_grad()
        self.Q2_optimizer.zero_grad()
        loss.backward()
        self.Q_optimizer.step()
        self.Q2_optimizer.step()

    def _target_action(self, batch: Batch):
        """Compute the target action for the batch.

        1. Query the target policy network for the next action.
        2. Add noise Clipped[Colored|Gaussian]Noise.
        3. Clip the action to the action space.

        Args:
            batch (Batch): The training batch.

        Returns:
            torch.Tensor: The target action.
        """
        next_actions = (
            self.policy_target(batch.next_observations).detach().cpu().numpy()
        )
        noise = self._target_smoothing_noise()
        next_actions = np.clip(
            next_actions + noise, self._action_space.low, self._action_space.high
        )
        next_actions = torch.tensor(next_actions, dtype=global_config.base_config.dtype)
        return next_actions

    def _compute_target_q(self, batch):
        """Compute the target Q values for the batch.

        Args:
            batch (Batch): The training batch.

        Returns:
            torch.Tensor: The target Q values.
        """
        # Compute the target Q values
        next_actions = self._target_action(batch)
        q1 = self.Q_target.Qvalues(batch.next_observations, next_actions)
        q2 = self.Q2_target.Qvalues(batch.next_observations, next_actions)

        # Compute the target Q values
        target_q = batch.rewards + self._config.discount * (
            1 - batch.dones
        ) * torch.min(q1, q2)
        return target_q

    ### Rest - START ###

    def __del__(self):
        return super().__del__()
