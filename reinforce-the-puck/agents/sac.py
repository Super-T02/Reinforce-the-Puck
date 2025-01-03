import os

import numpy as np
import torch
from agents.base_agent import BaseAgent
from components.memory import Batch
from components.networks import QFunction, StochasticPolicyNetwork
from components.noise import OUNoise
from gymnasium import spaces
from utils.config import AgentConfig, SACAgentConfig


class SAC(BaseAgent):
    """
    Agent implementing Q-learning with NN function approximation.
    """

    def __init__(
        self,
        observation_space: spaces.box.Box,
        action_space: spaces.box.Box,
        config: SACAgentConfig,
    ):
        super().__init__("SAC", observation_space, action_space, config)
        self._config: SACAgentConfig = config
        self._action_noise = OUNoise((self._action_n))

        self.policy = self._create_policy_net()  # Actor network
        self.policy_target = self._create_policy_net()

        self.Q1 = QFunction(
            input_size=self._obs_dim + self._action_n,
            output_size=1,
            hidden_sizes=config.critic_hidden_sizes,
            learning_rate=config.trainer_config.learning_rate_critic,
        )

        self.Q2 = QFunction(
            input_size=self._obs_dim + self._action_n,
            output_size=1,
            hidden_sizes=config.critic_hidden_sizes,
            learning_rate=config.trainer_config.learning_rate_critic,
        )

        self.Q1_target = QFunction(
            input_size=self._obs_dim + self._action_n,
            output_size=1,
            hidden_sizes=config.critic_hidden_sizes,
            learning_rate=config.trainer_config.learning_rate_critic,
        )

        self.Q2_target = QFunction(
            input_size=self._obs_dim + self._action_n,
            output_size=1,
            hidden_sizes=config.critic_hidden_sizes,
            learning_rate=config.trainer_config.learning_rate_critic,
        )

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
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

    def _create_policy_net(self) -> StochasticPolicyNetwork:
        """Create the policy network.

        Returns:
            Feedforward: The policy network.
        """
        return StochasticPolicyNetwork(
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

    #  def act(self, state, memory):
    #         state = torch.from_numpy(state).float().to(device)
    #         action_probs = self.action_layer(state)
    #         dist = Categorical(action_probs)
    #         action = dist.sample()

    #         memory.states.append(state)
    #         memory.actions.append(action)
    #         memory.logprobs.append(dist.log_prob(action))

    #         return action.item()

    def act(self, state) -> any:
        """Select an action based on the given state.
        Args:
            state: The current state of the environment.

        Returns:
            action: The selected action.
        """
        action = self.policy.sample(state)
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

    def reset(self) -> "SAC":
        """Reset the agent.

        Returns:
            SAC: The reset agent.
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
        return super().train(last_reward)

    def train_step(self, batch: Batch) -> dict:
        """Perform a single training step.

        Args:
            batch (Batch): The training batch.

        Returns:
            dict: The training statistics.
        """

        # compute the target Q value
        with torch.no_grad():
            next_actions, next_log_probs = self.policy_target.predict(
                batch.next_observations
            )

            # select minimum Q value
            target_q1 = self.Q1_target.Qvalues(batch.next_observations, next_actions)
            target_q2 = self.Q2_target.Qvalues(batch.next_observations, next_actions)
            target_q_min = torch.min(target_q1, target_q2)
            # create bellman backup: r + gamma * (1 - done) * (Q - alpha * logπ)
            target_q = batch.rewards + self._config.discount * (1 - batch.dones) * (
                target_q_min - self.alpha * next_log_probs
            )

        # update Q networks
        current_q1 = self.Q1.Qvalues(batch.observations, batch.actions)
        current_q2 = self.Q2.Qvalues(batch.observations, batch.actions)
        q1_loss = self.Q1.get_loss(current_q1, target_q)
        q2_loss = self.Q2.get_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss  # update both Q networks

        self.Q1_optimizer.zero_grad()
        self.Q2_optimizer.zero_grad()
        q_loss.backward()
        self.Q1_optimizer.step()
        self.Q2_optimizer.step()

        # policy update
        actions_pred, log_probs = self.policy.sample(batch.observations)
        q1_pred = self.Q1.Qvalues(batch.observations, actions_pred)
        q2_pred = self.Q2.Qvalues(batch.observations, actions_pred)
        q_pred_min = torch.min(q1_pred, q2_pred)

        policy_loss = (self.alpha * log_probs - q_pred_min).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # alpha update
        if self._auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_probs.detach() + self._target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # update alpha
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.0)

        losses = {
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item() if self._auto_alpha else 0.0,
        }
        return losses

    def __del__(self):
        return super().__del__()
