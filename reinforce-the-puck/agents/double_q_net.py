import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from components.memory import Batch
from components.networks import QFunction


class DoubleQLearningAgent:
    def __init__(
        self,
        state_dim: int,
        hidden_sizes: list[int],
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.device = device

        self.input_size = state_dim + 1

        self.q_net = QFunction(
            input_size=self.input_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            device=device,
        )
        self.q_target_net = QFunction(
            input_size=self.input_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            device=device,
        )
        self.q_target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def select_action(self, state: np.ndarray) -> int:
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        q_values = []

        for action in [0, 1]:
            action_tensor = torch.tensor(
                [[float(action)]], dtype=torch.float32, device=self.device
            )

            q_val = self.q_net.Qvalues(state_tensor, action_tensor)
            q_values.append(q_val.item())
        best_action = int(np.argmax(q_values))
        return best_action

    def train_step(self, batch: Batch) -> dict:
        # Gather batch data
        states = batch.observations.to(self.device)
        actions = batch.actions.to(self.device).float()
        rewards = batch.rewards.to(self.device).squeeze(-1)
        next_states = batch.next_observations.to(self.device)
        dones = batch.dones.to(self.device).squeeze(-1)
        B = states.size(0)

        current_q = self.q_net.Qvalues(states, actions).squeeze(-1)

        q_values_next = []
        for action_val in [0, 1]:
            action_tensor = torch.full((B, 1), float(action_val), device=self.device)
            q_val = self.q_net.Qvalues(next_states, action_tensor)
            q_values_next.append(q_val.squeeze(-1))
        q_values_next = torch.stack(q_values_next, dim=1)  # Shape: [B, 2]
        best_next_actions = q_values_next.argmax(dim=1).float().unsqueeze(-1)

        with torch.no_grad():
            target_q_values = self.q_target_net.Qvalues(
                next_states, best_next_actions
            ).squeeze(-1)

        td_target = rewards + self.gamma * (1 - dones) * target_q_values

        loss = F.smooth_l1_loss(current_q, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def update_target(self):
        self.q_target_net.load_state_dict(self.q_net.state_dict())
