import os

import numpy as np
import torch
from agents.base_agent import AgentMode, BaseAgent
from components.memory import Batch
from components.networks import QFunction, StochasticPolicyNetwork
from components.noise import ClippedColoredNoise
from gymnasium import spaces
from utils.config import AgentConfig, SACAgentConfig, global_config


class SACHierarchicalAgent(BaseAgent):
    """
    Agent implementing Q-learning with NN function approximation.
    """

    def __init__(
        self,
        observation_space: spaces.box.Box,
        action_space: spaces.box.Box,
        config: SACAgentConfig,
    ):
        self._config: SACAgentConfig = config
        print("Config: ", self._config.trainer_config.learning_rate_actor)
        self.device = self._config.specialized_config.device
        super().__init__("SAC", observation_space, action_space, config)

        """enhancement of original paper which uses fixed alpha (hyperparameter)

        https://arxiv.org/pdf/1812.05905
        We extend SAC to incorporate a
        number of modifications that accelerate training and improve stability with respect
        to the hyperparameters, including a constrained formulation that automatically
        tunes the temperature hyperparameter
        """

        self._action_noise = ClippedColoredNoise(
            (self._action_n,), sigma=0.1, beta=1, clip=0.8  # pink noise
        )

        if self._config.alpha_tuning:
            self._log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

            self.alpha_optimizer = torch.optim.Adam(
                [self._log_alpha], lr=self._config.alpha_lr
            )
            # use the hyperparameter from the original paper: https://arxiv.org/pdf/1812.05905 (See D Hyperparameters)
            self._target_entropy = -np.prod(action_space.shape)
            self.alpha = self._log_alpha.exp().item()
        else:
            self.alpha = config.alpha

        self.highlevel_policy = self._create_highlevel_policy_net()
        self.high_level_policy_optimizer = torch.optim.Adam(
            self.highlevel_policy.parameters(),
            lr=config.trainer_config.learning_rate_actor,
            eps=0.0003,
        )

        self.lowlevel_policy = self._create_policy_net(sub_goal_dim=self._action_n)

        self.Q1 = QFunction(
            input_size=self._obs_dim + self._action_n,  # Standard: obs_dim + action_dim
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
            device=self.device,
            hidden_sizes=config.critic_hidden_sizes,
            loss_fn=torch.nn.MSELoss(),
            learning_rate=config.trainer_config.learning_rate_critic,
        )

        self._copy_nets()  # directly copy to minimize the difference between the target and the policy network

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
        self.epoch = 0

    def _policy_activation(self) -> callable:
        """Activation function for the policy network.

        Returns:
            callable: The activation function.
        """
        return torch.nn.Tanh()

    def _create_policy_net(self, sub_goal_dim=0) -> StochasticPolicyNetwork:
        """Create the LOW-level policy network (optionally with sub-goal inputs).

        Args:
            sub_goal_dim (int): Dimension der Sub-Ziele (High-Level-Ausgabe).
                               0 = Keine Sub-Ziel-Inputs.
        Returns:
            StochasticPolicyNetwork: The low-level policy network.
        """
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
        """Create the HIGH-level policy network.

        Returns:
            StochasticPolicyNetwork: The high-level policy network.
        """
        # High-Level gibt "sub-goals" mit derselben Dimension wie action_n zurück
        return StochasticPolicyNetwork(
            input_size=self._obs_dim,
            hidden_sizes=self._config.actor_hidden_sizes,
            output_size=self._action_n,  # "Sub-Ziele" = Gleiche Dim. wie Actions
            activation=torch.nn.ReLU,
            device=self.device,
            output_activation=self._policy_activation(),
            log_std_min=self._config.log_std_min,
            log_std_max=self._config.log_std_max,
        )

    def _copy_nets(self) -> None:
        """Copy the policy network to the target policy network."""
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())

    def _soft_update(self, source, target, tau: float) -> None:
        """Soft update the target network parameters.
        Soft Updates vs Hard Updates see See Appendix E. Additional Baseline Results in https://arxiv.org/pdf/1801.01290
        """
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def act(self, state) -> any:
        if isinstance(state, np.ndarray):
            state_torch = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            )
        else:
            state_torch = state  # falls schon Tensor

        # Da in vielen Envs state 1D ist, transformieren wir es zur Batch-Dimension
        if len(state_torch.shape) == 1:
            state_torch = state_torch.unsqueeze(0)

        sub_goal, _ = self.highlevel_policy.predict(state_torch)

        combined_input = torch.cat([state_torch, sub_goal], dim=-1)

        action, _ = self.lowlevel_policy.predict(combined_input)

        action = action + self._action_noise()

        action = action.detach().cpu().numpy()[0]
        action = np.clip(action, self._action_space.low, self._action_space.high)
        return action

    def state(self) -> tuple:
        """Get the state of the agent.

        Returns:
            tuple: The state of the agent.
        """
        return (
            self.Q1.state_dict(),
            self.Q2.state_dict(),
            self.lowlevel_policy.state_dict(),
            self.highlevel_policy.state_dict(),
            self._log_alpha,
        )

    def restore_state(self, state: tuple) -> None:
        """Restore the state of the agent.

        Args:
            state (tuple): The state of the agent.
        """
        self.Q1.load_state_dict(state[0])
        self.Q2.load_state_dict(state[1])
        self.lowlevel_policy.load_state_dict(state[2])
        self.highlevel_policy.load_state_dict(state[3])
        self._log_alpha = state[3]
        self._copy_nets()

    def reset(self) -> "SACAgent":
        """Reset the agent.

        Returns:
            SAC: The reset agent.
        """
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
        self.restore_state(torch.load(path, weights_only=False))

    def train_step(self, batch: Batch) -> dict:
        """Perform a single training step.

        Args:
            batch (Batch): The training batch.

        Returns:
            dict: The training statistics.
        """
        if self._mode != AgentMode.TRAIN:
            raise ValueError("Agent is not in training mode.")

        q1_loss, q2_loss = self.update_q_values(batch)

        # Low-Level Policy-Update (inkl. Sub-Goal)
        policy_loss_low = self.update_lowlevel_policy(batch)

        # High-Level Policy Update (wie gehabt)
        policy_loss_high = self.update_highlevel_policy(batch)

        if self._config.alpha_tuning:
            alpha_loss = self.update_alpha(batch)
        else:
            alpha_loss = torch.zeros(1)

        self._soft_update(self.Q1, self.Q1_target, self._config.tau)
        self._soft_update(self.Q2, self.Q2_target, self._config.tau)

        losses = {
            "loss": q1_loss.item() + q2_loss.item(),
            "actor_loss_low": policy_loss_low.item(),
            "actor_loss_high": policy_loss_high.item(),
            "alpha_loss": alpha_loss.item(),
        }
        return losses

    def update_alpha(self, batch):
        sub_goals, _ = self.highlevel_policy.sample(batch.observations)
        combined_obs = torch.cat([batch.observations, sub_goals], dim=-1)
        _, log_probs = self.lowlevel_policy.sample(
            combined_obs
        )  # ACHTUNG: hier OHNE subgoals (Batch)
        temp = (log_probs + self._target_entropy).to(self._log_alpha.device).detach()
        alpha_loss = -(self._log_alpha * temp).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self._log_alpha.exp().item()
        return alpha_loss

    def update_lowlevel_policy(self, batch):
        """
        1) High-Level Policy generiert subgoals für jede State-Observation in batch
        2) Kombiniere observation + subgoal
        3) Low-Level Policy wählt finale Aktionen
        4) Standard-SAC-Loss
        """
        # 1) Sub-Goals vom High-Level
        sub_goals, _ = self.highlevel_policy.sample(batch.observations)
        batch.observations = batch.observations.to(self.device)

        # 2) Combined Input = (obs, sub_goals)
        combined_obs = torch.cat([batch.observations, sub_goals], dim=-1)

        # 3) Aktionen aus Low-Level Policy + LogProbs
        actions_pred, log_probs = self.lowlevel_policy.sample(combined_obs)
        log_probs = log_probs.cpu()

        # Qs werden nach wie vor nur aus (obs, action) berechnet
        # (Minimaländerung: hier ignorieren wir sub_goals beim Q-Eingang.)
        q1_pred = self.Q1.Qvalues(batch.observations, actions_pred)
        q2_pred = self.Q2.Qvalues(batch.observations, actions_pred)
        q_pred_min = torch.min(q1_pred, q2_pred)

        # 4) SAC Policy Loss
        # E[-(Q(s, π(s)) - α * log π(s))]
        policy_loss = -(q_pred_min - self.alpha * log_probs).mean()

        self.low_level_policy_optimizer.zero_grad()
        policy_loss.backward()
        self.low_level_policy_optimizer.step()
        return policy_loss

    def update_highlevel_policy(self, batch: Batch) -> torch.Tensor:
        subgoals, log_probs = self.highlevel_policy.sample(batch.observations)
        log_probs = log_probs.cpu()

        # Naiv: High-Level kriegt die "Environment Rewards" für alle Subgoals
        policy_loss = -torch.mean(log_probs * batch.rewards)

        """
        Example for Intuition:

        log_prob, reward
        100%    ->   +10
        20%     ->   -2
        2%      ->   -20
        """

        self.high_level_policy_optimizer.zero_grad()
        policy_loss.backward()
        self.high_level_policy_optimizer.step()

        return policy_loss

    def update_q_values(self, batch):
        with torch.no_grad():
            sub_goals, _ = self.highlevel_policy.sample(batch.next_observations)
            batch.next_observations = batch.next_observations.to(self.device)
            combined_next_obs = torch.cat([batch.next_observations, sub_goals], dim=-1)

            next_actions, next_log_probs = self.lowlevel_policy.sample(
                combined_next_obs
            )
            next_log_probs = next_log_probs.cpu()

            target_q1 = self.Q1_target.Qvalues(batch.next_observations, next_actions)
            target_q2 = self.Q2_target.Qvalues(batch.next_observations, next_actions)
            target_q_min = torch.min(target_q1, target_q2)

            """"
            The soft Q-function parameters can be trained to minimize the soft Bellman residual
            In (Haarnoja et al., 2018c) we introduced an additional function approximator for the value function,
            but later found it to be unnecessary
            --> no value function
            https://arxiv.org/pdf/1812.05905
            """
            target_q = batch.rewards + self._config.discount * (1 - batch.dones) * (
                target_q_min - self.alpha * next_log_probs  # entropy term
            )

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

    def __del__(self):
        return super().__del__()
