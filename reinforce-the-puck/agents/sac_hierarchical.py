import os

import numpy as np
import torch
from agents.base_agent import AgentMode, BaseAgent
from components.memory import Batch
from components.networks import QFunction, StochasticPolicyNetwork
from gymnasium import spaces
from utils.config import AgentConfig, HierarchicalAgentConfig, global_config


class SACHierarchicalAgent(BaseAgent):
    """
    Agent implementing Q-learning with NN function approximation.
    """

    def __init__(
        self,
        observation_space: spaces.box.Box,
        action_space: spaces.box.Box,
        config: HierarchicalAgentConfig,
    ):
        self._config: HierarchicalAgentConfig = config
        print("Config: ", self._config.trainer_config.learning_rate_actor)
        self.device = self._config.specialized_config.device
        super().__init__("SAC", observation_space, action_space, config)

        # Alpha Tuning
        if self._config.alpha_tuning:
            self._log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self._target_entropy = -np.prod(action_space.shape)
            self.alpha = self._log_alpha.exp().item()
        else:
            self.alpha = config.alpha

        self.highlevel_policy = self._create_highlevel_policy_net()
        self.high_level_policy_optimizer = torch.optim.Adam(
            self.highlevel_policy.parameters(),
            lr=config.trainer_config.learning_rate_actor,
            eps=0.0001,
        )

        # High-Level Critic Networks
        self.Q1_high = QFunction(
            input_size=self._obs_dim + self._action_n,
            output_size=1,
            hidden_sizes=config.critic_hidden_sizes,
            loss_fn=torch.nn.MSELoss(),
            device=self.device,
            learning_rate=config.trainer_config.learning_rate_critic,
        )
        self.Q2_high = QFunction(
            input_size=self._obs_dim + self._action_n,
            output_size=1,
            hidden_sizes=config.critic_hidden_sizes,
            loss_fn=torch.nn.MSELoss(),
            device=self.device,
            learning_rate=config.trainer_config.learning_rate_critic,
        )
        self.Q1_high_target = QFunction(
            input_size=self._obs_dim + self._action_n,
            output_size=1,
            hidden_sizes=config.critic_hidden_sizes,
            loss_fn=torch.nn.MSELoss(),
            device=self.device,
            learning_rate=config.trainer_config.learning_rate_critic,
        )
        self.Q2_high_target = QFunction(
            input_size=self._obs_dim + self._action_n,
            output_size=1,
            hidden_sizes=config.critic_hidden_sizes,
            loss_fn=torch.nn.MSELoss(),
            device=self.device,
            learning_rate=config.trainer_config.learning_rate_critic,
        )

        self.lowlevel_policy = self._create_policy_net(sub_goal_dim=self._action_n)

        self.Q1 = QFunction(
            input_size=self._obs_dim + self._action_n,
            output_size=1,
            hidden_sizes=config.critic_hidden_sizes,
            loss_fn=torch.nn.MSELoss(),
            device=self.device,
            learning_rate=config.trainer_config.learning_rate_critic,
        )

        self.Q2 = QFunction(
            input_size=self._obs_dim + self._action_n,
            output_size=1,
            hidden_sizes=config.critic_hidden_sizes,
            loss_fn=torch.nn.MSELoss(),
            device=self.device,
            learning_rate=config.trainer_config.learning_rate_critic,
        )

        self.Q1_target = QFunction(
            input_size=self._obs_dim + self._action_n,
            output_size=1,
            hidden_sizes=config.critic_hidden_sizes,
            loss_fn=torch.nn.MSELoss(),
            device=self.device,
            learning_rate=config.trainer_config.learning_rate_critic,
        )

        self.Q2_target = QFunction(
            input_size=self._obs_dim + self._action_n,
            output_size=1,
            hidden_sizes=config.critic_hidden_sizes,
            loss_fn=torch.nn.MSELoss(),
            device=self.device,
            learning_rate=config.trainer_config.learning_rate_critic,
        )

        self.create_optimizer()

        self._copy_nets()

        self.epoch = 0

    def create_optimizer(self):
        config = self._config
        self.Q1_high_optimizer = torch.optim.Adam(
            self.Q1_high.parameters(),
            lr=config.trainer_config.learning_rate_critic,
            eps=0.000001,
        )
        self.Q2_high_optimizer = torch.optim.Adam(
            self.Q2_high.parameters(),
            lr=config.trainer_config.learning_rate_critic,
            eps=0.000001,
        )

        self.low_level_policy_optimizer = torch.optim.Adam(
            self.lowlevel_policy.parameters(),
            lr=config.trainer_config.learning_rate_actor,
            eps=0.000001,
        )

        self.Q1_optimizer = torch.optim.Adam(
            self.Q1.parameters(),
            lr=config.trainer_config.learning_rate_critic,
            eps=0.000001,
        )
        self.Q2_optimizer = torch.optim.Adam(
            self.Q2.parameters(),
            lr=config.trainer_config.learning_rate_critic,
            eps=0.000001,
        )
        if self._config.alpha_tuning:
            self.alpha_optimizer = torch.optim.Adam(
                [self._log_alpha], lr=self._config.alpha_lr
            )

    def _policy_activation(self) -> callable:
        return torch.nn.Tanh()

    def _create_policy_net(self, sub_goal_dim=0) -> StochasticPolicyNetwork:
        input_size = self._obs_dim + sub_goal_dim
        return StochasticPolicyNetwork(
            input_size=input_size,
            hidden_sizes=self._config.actor_hidden_sizes,
            output_size=self._action_n,
            activation=torch.nn.ReLU,
            device=self.device,
            output_activation=self._policy_activation(),
            log_std_min=self._config.log_std_min,
            log_std_max=self._config.log_std_max,
        )

    def _create_highlevel_policy_net(self) -> StochasticPolicyNetwork:
        return StochasticPolicyNetwork(
            input_size=self._obs_dim,
            hidden_sizes=self._config.actor_hidden_sizes,
            output_size=self._action_n,
            activation=torch.nn.ReLU,
            device=self.device,
            output_activation=self._policy_activation(),
            log_std_min=self._config.log_std_min,
            log_std_max=self._config.log_std_max,
        )

    def _copy_nets(self) -> None:
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())
        self.Q1_high_target.load_state_dict(self.Q1_high.state_dict())
        self.Q2_high_target.load_state_dict(self.Q2_high.state_dict())

    def _soft_update(self, source, target, tau: float) -> None:
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def act(self, state) -> any:
        if isinstance(state, np.ndarray):
            state_torch = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            )
        else:
            state_torch = state

        if len(state_torch.shape) == 1:
            state_torch = state_torch.unsqueeze(0)

        sub_goal, _ = self.highlevel_policy.sample(state_torch)
        combined_input = torch.cat([state_torch, sub_goal], dim=-1)
        action, _ = self.lowlevel_policy.sample(combined_input)
        action = action.detach().cpu().numpy()[0]
        action = np.clip(action, self._action_space.low, self._action_space.high)
        return action

    def state(self) -> tuple:
        return (
            self.Q1.state_dict(),
            self.Q2.state_dict(),
            self.lowlevel_policy.state_dict(),
            self.highlevel_policy.state_dict(),
            self.Q1_high.state_dict(),
            self.Q2_high.state_dict(),
            self._log_alpha,
        )

    def restore_state(self, state: tuple) -> None:
        self.Q1.load_state_dict(state[0])
        self.Q2.load_state_dict(state[1])
        self.lowlevel_policy.load_state_dict(state[2])
        self.highlevel_policy.load_state_dict(state[3])
        self.Q1_high.load_state_dict(state[4])
        self.Q2_high.load_state_dict(state[5])
        self._log_alpha = state[6].to(self.device)

        self.create_optimizer()
        self._copy_nets()

    def reset(self) -> "HierarchicalAgent":
        return self

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state(), path)

    def load(self, path: str) -> None:
        self.restore_state(torch.load(path, map_location=self.device))

    def update_alpha(self, batch):
        sub_goals, _ = self.highlevel_policy.sample(batch.observations)
        combined_obs = torch.cat(
            [batch.observations.to(self.device), sub_goals], dim=-1
        )
        _, log_probs = self.lowlevel_policy.sample(combined_obs)
        temp = (log_probs + self._target_entropy).to(self.device).detach()
        alpha_loss = -(self._log_alpha * temp).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self._log_alpha.exp().item()
        return alpha_loss

    def update_lowlevel_policy(self, batch):
        sub_goals, _ = self.highlevel_policy.sample(batch.observations)
        combined_obs = torch.cat(
            [batch.observations.to(self.device), sub_goals], dim=-1
        )
        actions_pred, log_probs = self.lowlevel_policy.sample(combined_obs)
        q1_pred = self.Q1.Qvalues(batch.observations, actions_pred)
        q2_pred = self.Q2.Qvalues(batch.observations, actions_pred)
        q_pred_min = torch.min(q1_pred, q2_pred).to(self.device)
        policy_loss = -(q_pred_min - self.alpha * log_probs).mean()
        self.low_level_policy_optimizer.zero_grad()
        policy_loss.backward()
        self.low_level_policy_optimizer.step()
        return policy_loss

    def update_highlevel_q_values(self, batch):
        # Sample current sub-goals for the current observations
        subgoals_current, _ = self.highlevel_policy.sample(batch.observations)
        critic_input = torch.cat(
            [batch.observations.to(self.device), subgoals_current], dim=1
        )

        # Compute target Q-value without computing gradients
        with torch.no_grad():
            next_subgoals, next_log_probs = self.highlevel_policy.sample(
                batch.next_observations.to(self.device)
            )
            target_q1_high = self.Q1_high_target.Qvalues(
                batch.next_observations.to(self.device), next_subgoals
            )
            target_q2_high = self.Q2_high_target.Qvalues(
                batch.next_observations.to(self.device), next_subgoals
            )
            target_q_min_high = torch.min(target_q1_high, target_q2_high).to(
                self.device
            )
            target_q_high = batch.rewards.to(self.device) + self._config.discount * (
                1 - batch.dones.to(self.device)
            ) * (target_q_min_high - self.alpha * next_log_probs)

        # Compute individual losses for high-level critics
        q1_high_loss = self.Q1_high.get_loss(critic_input, target_q_high).mean()
        q2_high_loss = self.Q2_high.get_loss(critic_input, target_q_high).mean()

        # Sum the losses to perform a single backward pass
        total_loss = q1_high_loss + q2_high_loss

        self.Q1_high_optimizer.zero_grad()
        self.Q2_high_optimizer.zero_grad()
        total_loss.backward()
        self.Q1_high_optimizer.step()
        self.Q2_high_optimizer.step()

        return q1_high_loss, q2_high_loss

    def update_highlevel_policy(self, batch: Batch) -> torch.Tensor:
        subgoals, log_probs = self.highlevel_policy.sample(
            batch.observations.to(self.device)
        )
        q1_high_pred = self.Q1_high.Qvalues(
            batch.observations.to(self.device), subgoals
        )
        q2_high_pred = self.Q2_high.Qvalues(
            batch.observations.to(self.device), subgoals
        )
        q_high_pred_min = torch.min(q1_high_pred, q2_high_pred).to(self.device)
        policy_loss = -(q_high_pred_min - self.alpha * log_probs).mean()
        self.high_level_policy_optimizer.zero_grad()
        policy_loss.backward()
        self.high_level_policy_optimizer.step()
        return policy_loss

    def update_q_values(self, batch):
        with torch.no_grad():
            sub_goals, _ = self.highlevel_policy.sample(
                batch.next_observations.to(self.device)
            )
            combined_next_obs = torch.cat(
                [batch.next_observations.to(self.device), sub_goals], dim=-1
            )
            next_actions, next_log_probs = self.lowlevel_policy.sample(
                combined_next_obs
            )
            target_q1 = self.Q1_target.Qvalues(
                batch.next_observations.to(self.device), next_actions
            )
            target_q2 = self.Q2_target.Qvalues(
                batch.next_observations.to(self.device), next_actions
            )
            target_q_min = torch.min(target_q1, target_q2).to(self.device)
            target_q = batch.rewards.to(self.device) + self._config.discount * (
                1 - batch.dones.to(self.device)
            ) * (target_q_min - self.alpha * next_log_probs)
        q1_loss = self.Q1.get_loss(
            torch.cat([batch.observations, batch.actions], dim=1), target_q
        ).mean()
        q2_loss = self.Q2.get_loss(
            torch.cat([batch.observations, batch.actions], dim=1), target_q
        ).mean()
        self.Q1_optimizer.zero_grad()
        q1_loss.backward()
        self.Q1_optimizer.step()
        self.Q2_optimizer.zero_grad()
        q2_loss.backward()
        self.Q2_optimizer.step()
        return q1_loss, q2_loss

    def train_step(self, batch: Batch) -> dict:
        if self._mode != AgentMode.TRAIN:
            raise ValueError("Agent is not in training mode.")
        q1_loss, q2_loss = self.update_q_values(batch)
        q1_high_loss, q2_high_loss = self.update_highlevel_q_values(batch)
        policy_loss_low = self.update_lowlevel_policy(batch)
        policy_loss_high = self.update_highlevel_policy(batch)
        if self._config.alpha_tuning:
            alpha_loss = self.update_alpha(batch)
        else:
            alpha_loss = torch.zeros(1, device=self.device)
        self._soft_update(self.Q1, self.Q1_target, self._config.tau)
        self._soft_update(self.Q2, self.Q2_target, self._config.tau)
        self._soft_update(self.Q1_high, self.Q1_high_target, self._config.tau)
        self._soft_update(self.Q2_high, self.Q2_high_target, self._config.tau)
        losses = {
            "loss": q1_loss.item()
            + q2_loss.item()
            + q1_high_loss.item()
            + q2_high_loss.item(),
            "actor_loss_low": policy_loss_low.item(),
            "actor_loss_high": policy_loss_high.item(),
            "alpha_loss": alpha_loss.item(),
        }
        return losses

    def __del__(self):
        return super().__del__()
