import numpy as np
import torch
from agents.base_agent import AgentMode
from agents.td3 import TD3Agent
from components.networks import Feedforward
from components.noise import ClippedColoredNoise, ClippedGaussianNoise, ColoredNoise
from utils.config import TD3AgentConfig, global_config


class TD3CrossQAgent(TD3Agent):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) with cross q adaptation agent.
    """

    def __init__(
        self, observation_space: tuple, action_space: tuple, config: TD3AgentConfig
    ):
        super().__init__(observation_space, action_space, config, "TD3CrossQ")

        # Delete target networks
        del (
            self.Q_target,
            self.Q2_target,
            self.policy_target,
        )

    def _create_q_net(self, out: int, lr: int = 0, **kwargs):
        return super()._create_q_net(out, lr, use_batch_norm=True, **kwargs)

    def _create_policy_net(self, **kwargs):
        return super()._create_policy_net(use_batch_norm=True, **kwargs)

    def _get_target_action(self, state):
        return self.policy(state)

    def _copy_nets(self):
        pass

    def _update_target_nets(self):
        pass

    def _update_target_net(self, target: Feedforward, source: Feedforward):
        pass

    def train_step(self, batch):
        """Perform a single training step.

        Args:
            batch (Batch): The training batch.

        Returns:
            dict: The training statistics.
        """
        if self._mode != AgentMode.TRAIN:
            raise ValueError("Agent is not in training mode.")
        critic_loss = self._optimize_critic(batch)
        if self._train_iterations % self._config.policy_delay == 0:
            actor_loss = self._optimize_actor(batch)
            self._last_actor_loss = actor_loss.item()
        return {"loss": critic_loss.item(), "actor_loss": self._last_actor_loss}

    def _compute_q_loss(self, batch):
        """Compute the Q loss with a concatenated forward pass."""
        # "Target" Q values (without target networks)
        target_q = self._compute_target_q(batch)

        # Forward pass
        q1, q2, _, _ = self._joint_forward(
            batch.observations,
            batch.next_observations,
            batch.actions,
            self.act(batch.next_observations),
        )

        # Compute the Q loss
        q1_loss = self.Q.get_loss(q1, target_q)
        q2_loss = self.Q2.get_loss(q2, target_q)
        return q1_loss + q2_loss

    def _compute_target_q(self, batch):
        """Compute the target Q values for the batch without target networks."""
        next_actions = self.act(batch.next_observations)
        noise = self._target_smoothing_noise()
        next_actions = np.clip(
            next_actions + noise, self._action_space.low, self._action_space.high
        )
        next_actions = torch.tensor(next_actions, dtype=global_config.base_config.dtype)

        # Forward
        _, _, next_q1, next_q2 = self._joint_forward(
            batch.observations, batch.next_observations, batch.actions, next_actions
        )

        # Compute the target Q-values using the minimum of the two critics
        next_q = torch.min(next_q1, next_q2)
        target_q = batch.rewards + self._config.discount * (1 - batch.dones) * next_q
        return target_q

    def _joint_forward(self, states, next_states, actions, next_actions):
        """Perform a joint forward pass through both critics."""
        # Concatenate for a joint forward pass
        all_states = torch.cat([states, next_states], dim=0)
        all_actions = torch.cat([actions, next_actions], dim=0)

        # Joint Forward pass
        all_q1 = self.Q.forward(torch.cat([all_states, all_actions], dim=1))
        all_q2 = self.Q2.forward(torch.cat([all_states, all_actions], dim=1))

        # Split back the Q-values
        q1, next_q1 = torch.split(all_q1, states.shape[0])
        q2, next_q2 = torch.split(all_q2, states.shape[0])
        return q1, q2, next_q1, next_q2

    def __del__(self):
        return super().__del__()
