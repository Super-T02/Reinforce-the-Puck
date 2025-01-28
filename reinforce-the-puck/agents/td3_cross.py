import numpy as np
import torch
from agents.base_agent import AgentMode
from agents.td3 import TD3Agent
from components.networks import Feedforward
from utils.config import TD3AgentConfig, global_config


class TD3CrossQAgent(TD3Agent):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) with cross q adaptation agent.
    """

    def __init__(
        self, observation_space: tuple, action_space: tuple, config: TD3AgentConfig
    ):
        super().__init__(
            observation_space, action_space, config, "TD3CrossQ", use_batch_norm=True
        )

        # Delete target networks
        del (
            self.Q_target,
            self.Q2_target,
            self.policy_target,
        )

    def _get_target_action(self, state):
        state = self._create_tensor(state)
        return self.policy.predict(state)

    def _copy_nets(self):
        pass

    def _update_target_nets(self):
        pass

    def _update_target_net(self, target: Feedforward, source: Feedforward):
        pass

    def _compute_target_q(self, batch):
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
        # Select action with clipped noise
        next_actions = self.policy(batch.next_observations).detach().cpu().numpy()
        noise = self._target_smoothing_noise()
        next_actions = np.clip(
            next_actions + noise, self._action_space.low, self._action_space.high
        )
        next_actions = torch.tensor(next_actions, dtype=global_config.base_config.dtype)

        # Forward pass
        q1, q2, next_q1, next_q2 = self._joint_forward(
            batch.observations,
            batch.next_observations,
            batch.actions,
            next_actions,
        )

        # Compute the target Q-values using the minimum of the two critics
        next_q = torch.min(next_q1, next_q2).detach().cpu()
        q_hat = batch.rewards + self._config.discount * (1 - batch.dones) * next_q
        q_hat = self._create_tensor(q_hat)

        # Compute the Q loss
        q1_loss = self.Q.get_loss(q1, q_hat, False)
        q2_loss = self.Q2.get_loss(q2, q_hat, False)
        return q1_loss + q2_loss

    def _joint_forward(self, states, next_states, actions, next_actions):
        """Perform a joint forward pass through both critics."""
        # Create Tensors if not already
        states = self._create_tensor(states)
        next_states = self._create_tensor(next_states)
        actions = self._create_tensor(actions)
        next_actions = self._create_tensor(next_actions)

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

    def _create_tensor(self, data):
        return (
            torch.tensor(data, dtype=global_config.base_config.dtype)
            if not isinstance(data, torch.Tensor)
            else data
        )

    def __del__(self):
        return super().__del__()
