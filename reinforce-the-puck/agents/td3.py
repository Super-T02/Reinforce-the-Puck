import numpy as np
import torch
from agents.base_agent import AgentMode
from agents.ddpg import DDPGAgent
from components.networks import Feedforward
from components.noise import ClippedGaussianNoise
from utils.config import TD3AgentConfig, global_config


class TD3Agent(DDPGAgent):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.
    """

    def __init__(
        self, observation_space: tuple, action_space: tuple, config: TD3AgentConfig
    ):
        self.Q2, self.Q2_target = None, None
        super().__init__(observation_space, action_space, config, "TD3")
        self._action_noise = ClippedGaussianNoise(
            (self._action_n,), config.noise_sigma, config.noise_clip
        )
        self._target_smoothing_noise = ClippedGaussianNoise(
            (self._action_n,), 0.1, config.noise_clip
        )

        # Create 2nd Q network
        self.Q2 = self._create_q_net(1, config.trainer_config.learning_rate_critic)
        self.Q2_target = self._create_q_net(1)
        self._last_actor_loss = torch.nan
        self._copy_nets()
        self.Q_optimizer = self._create_q_optim([self.Q, self.Q2])

    def _copy_nets(self):
        super()._copy_nets()
        if self.Q2 is not None:
            self.Q2_target.load_state_dict(self.Q2.state_dict())

    def _update_target_nets(self):
        """Update the target networks."""
        pairs = [
            (self.Q_target, self.Q),
            (self.policy_target, self.policy),
            (self.Q2_target, self.Q2),
        ]
        for target, source in pairs:
            self._update_target_net(target, source)

    def _update_target_net(self, target: Feedforward, source: Feedforward):
        """Update the target network.

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

    def restore_state(self, state):
        self.Q2.load_state_dict(state[2])
        super().restore_state(state)

    def state(self):
        return (*super().state(), self.Q2.state_dict())

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
            self._update_target_nets()
            self._last_actor_loss = actor_loss.item()
        return {"loss": critic_loss.item(), "actor_loss": self._last_actor_loss}

    def _compute_q_loss(self, batch):
        """Calculate the target Q values.

        Args:
            batch (Batch): The training batch.

        Returns:
            torch.Tensor: The target Q values.
        """
        target_q = self._compute_target_q(batch)
        x = torch.cat((batch.observations, batch.actions), dim=1)
        q1_loss = self.Q.get_loss(x, target_q)
        q2_loss = self.Q2.get_loss(x, target_q)
        return q1_loss + q2_loss

    def _compute_target_q(self, batch):
        """Compute the target Q values for the batch.

        Args:
            batch (Batch): The training batch.

        Returns:
            torch.Tensor: The target Q values.
        """
        next_actions = self.act(batch.next_observations)
        noise = self._target_smoothing_noise()
        next_actions = np.clip(
            next_actions + noise, self._action_space.low, self._action_space.high
        )
        next_actions = torch.tensor(next_actions, dtype=global_config.base_config.dtype)
        q1 = self.Q_target.Qvalues(batch.next_observations, next_actions)
        q2 = self.Q2_target.Qvalues(batch.next_observations, next_actions)
        target_q = batch.rewards + self._config.discount * (
            1 - batch.dones
        ) * torch.min(q1, q2)
        return target_q

    def __del__(self):
        return super().__del__()
