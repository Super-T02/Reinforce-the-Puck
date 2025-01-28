import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.base_agent import BaseAgent
from components.memory import Batch
from components.networks import StochasticPolicyNetwork
from gymnasium import spaces
from utils.config import SACAgentConfig


class JointQFunction(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        activation: callable = nn.ReLU,
        use_batch_norm: bool = True,
        bn_momentum: float = 0.99,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList() if dropout_rate > 0 else None

        prev_size = input_size
        for size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(size, momentum=bn_momentum))
            if dropout_rate > 0:
                self.dropouts.append(nn.Dropout(dropout_rate))
            prev_size = size

        self.q1_head = nn.Linear(prev_size, 1)
        self.q2_head = nn.Linear(prev_size, 1)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.batch_norms:
                x = self.batch_norms[i](x)
            x = self.activation(x)
            if self.dropouts and self.training:
                x = self.dropouts[i](x)
        return self.q1_head(x), self.q2_head(x)


# Hauptagent-Klasse
class CrossQAgent(BaseAgent):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        config: SACAgentConfig,
    ):
        self.device = config.specialized_config.device
        # Hyperparameter-Anpassungen
        config.tau = 1.0
        config.critic_hidden_sizes = [2048, 2048]
        config.specialized_config.bn_momentum = 0.99
        config.dropout_rate = (
            0.1 if not hasattr(config, "dropout_rate") else config.dropout_rate
        )

        super().__init__(
            name="CrossQ",
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )

        # Zustands- und Aktionsdimensionen
        self._obs_dim = observation_space.shape[0]
        self._action_n = action_space.shape[0]

        # Alpha-Tuning
        self._log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam(
            [self._log_alpha], lr=config.alpha_lr, eps=1e-6
        )
        self._target_entropy = -np.prod(action_space.shape)
        self.alpha = self._log_alpha.exp().item()

        # Netzwerke
        self.policy = self._create_policy_net(config)
        self.Q = self._create_q_network(config)
        self.Q_target = self._create_q_network(config)
        self._hard_update()

        # Optimizer
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.trainer_config.learning_rate_actor,
            betas=(0.5, 0.999),
            eps=1e-6,
        )
        self.Q_optimizer = torch.optim.Adam(
            self.Q.parameters(),
            lr=config.trainer_config.learning_rate_critic,
            betas=(0.5, 0.999),
            eps=1e-6,
        )

    def _create_policy_net(self, config) -> StochasticPolicyNetwork:
        return StochasticPolicyNetwork(
            input_size=self._obs_dim,
            hidden_sizes=config.actor_hidden_sizes,
            output_size=self._action_n,
            activation=nn.ReLU,
            output_activation=nn.Tanh,
            device=self.device,
            log_std_min=config.log_std_min,
            log_std_max=config.log_std_max,
            use_batch_norm=True,
            bn_momentum=config.specialized_config.bn_momentum,
            dropout_rate=config.dropout_rate,
        )

    def _create_q_network(self, config) -> JointQFunction:
        return JointQFunction(
            input_size=self._obs_dim + self._action_n,
            hidden_sizes=config.critic_hidden_sizes,
            activation=nn.ReLU,
            use_batch_norm=True,
            bn_momentum=config.specialized_config.bn_momentum,
            dropout_rate=config.dropout_rate,
        ).to(self.device)

    def _hard_update(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

    def act(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.policy.sample(state_tensor)
        return (
            action.squeeze(0)
            .cpu()
            .numpy()
            .clip(self._action_space.low, self._action_space.high)
        )

    def train_step(self, batch: Batch) -> dict:
        q_loss = self.update_q_values(batch)
        policy_loss = self.update_policy(batch)
        alpha_loss = self.update_alpha(batch)

        return {
            "total_loss": q_loss + policy_loss + alpha_loss,
            "loss": q_loss,
            "policy_loss": policy_loss,
            "alpha_loss": alpha_loss,
            "alpha_value": self.alpha,
        }

    def update_q_values(self, batch: Batch) -> float:
        self.Q.train()
        self.Q_target.eval()

        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(batch.next_observations)
            q1_t, q2_t = self.Q_target(
                torch.cat([batch.next_observations, next_actions], 1)
            )
            target_q = batch.rewards + self._config.discount * (1 - batch.dones) * (
                torch.min(q1_t, q2_t) - self.alpha * next_log_probs
            )

        current_q1, current_q2 = self.Q(
            torch.cat([batch.observations, batch.actions], 1)
        )

        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.Q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Q.parameters(), 1.0)
        self.Q_optimizer.step()
        self._hard_update()

        return q_loss.item()

    def update_policy(self, batch: Batch) -> float:
        actions, log_probs = self.policy.sample(batch.observations)
        q1, q2 = self.Q(torch.cat([batch.observations, actions], 1))
        policy_loss = -(torch.min(q1, q2) - self.alpha * log_probs).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()

        return policy_loss.item()

    def update_alpha(self, batch: Batch) -> float:
        with torch.no_grad():
            _, log_probs = self.policy.sample(batch.observations)

        alpha_loss = -(self._log_alpha * (log_probs + self._target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self._log_alpha.exp().item()

        return alpha_loss.item()

    def state_dict(self) -> dict:
        return {
            "Q": self.Q.state_dict(),
            "Q_target": self.Q_target.state_dict(),
            "policy": self.policy.state_dict(),
            "log_alpha": self._log_alpha,
            "Q_optimizer": self.Q_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def reset(self) -> "CrossQAgent":
        """Reset the agent.

        Returns:
            SAC: The reset agent.
        """
        return self

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))

    def load_state_dict(self, state: dict):
        # Laden der Netzwerkparameter
        self.Q.load_state_dict(state["Q"])
        self.Q_target.load_state_dict(state["Q_target"])
        self.policy.load_state_dict(state["policy"])

        # Alpha-Parameter mit Gradiententracking
        self._log_alpha = state["log_alpha"].clone().detach().requires_grad_(True)
        self.alpha = self._log_alpha.exp().item()

        # Optimizer-Zustände laden
        self.Q_optimizer.load_state_dict(state["Q_optimizer"])
        self.policy_optimizer.load_state_dict(state["policy_optimizer"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])

        # Parameter zu richtigem Device verschieben
        for param in self.Q.parameters():
            param.data = param.data.to(self.device)
        for param in self.Q_target.parameters():
            param.data = param.data.to(self.device)
        for param in self.policy.parameters():
            param.data = param.data.to(self.device)

        # Target Networks synchronisieren
        self._hard_update()

        # Optimizer-Device-Korrektur
        for state in self.Q_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        for state in self.policy_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        for state in self.alpha_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
