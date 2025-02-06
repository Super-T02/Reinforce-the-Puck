import numpy as np
import torch
from components.data_structures import SumTree
from utils.config import global_config


class Batch:
    def __init__(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        indices: np.ndarray = None,
        priority_weights: torch.Tensor = None,
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.next_observations = next_observations
        self.dones = dones
        self.indices = indices
        self.priority_weights = priority_weights


class MemoryInterface:
    """Interface for the Memory buffer."""

    def __init__(self, max_size: int):
        pass

    def add_transition(self, transitions_new: list) -> "MemoryInterface":
        pass

    def sample(self, batch: int = 1, indices: np.ndarray = None) -> np.ndarray:
        pass

    def get_all_transitions(self):
        pass


class Memory(MemoryInterface):
    """Memory buffer to store transitions."""

    def __init__(self, max_size=global_config.base_config.max_memory_size):
        """Initialize the Memory object.

        Args:
            max_size (int, optional): Maximum size of the memory buffer. Defaults to config.MAX_MEMORY_SIZE.
        """
        self.transitions = np.asarray([])
        self.size = 0  # Size of the memory buffer
        self.current_idx = 0  # Current index to add new transitions
        self.max_size = max_size  # Maximum size of the memory buffer

    def add_transition(self, transitions_new: list) -> "Memory":
        """Add a new transition to the memory.

        Args:
            transitions_new (list): List of transitions to be added.

        Returns:
            self: Memory object.
        """
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        # Add new transition to the memory buffer
        self.transitions[self.current_idx, :] = np.asarray(
            transitions_new, dtype=object
        )

        # Update size and current index
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size
        return self

    def sample(self, batch: int = 1, indices: np.ndarray = None) -> np.ndarray:
        """Sample a batch of transitions from the memory.

        Args:
            batch (int, optional): Batch size. Defaults to 1.
            indices (np.ndarray, optional): Indices of transitions to sample. Defaults to None.

        Returns:
            np.ndarray: Batch of transitions.
        """
        if batch > self.size:
            batch = self.size
        if indices is not None:
            self.inds = indices
        else:
            self.inds = np.random.choice(range(self.size), size=batch, replace=False)
        return self.transitions[self.inds, :]

    def get_all_transitions(self):
        """Get all transitions from the memory.

        Returns:
            np.ndarray: All transitions.
        """
        return self.transitions[0 : self.size]


class PrioritizedMemory(MemoryInterface):
    """Prioritized Memory buffer based on:
    Schaul, Tom. "Prioritized Experience Replay." arXiv preprint arXiv:1511.05952 (2015).
    """

    def __init__(
        self,
        max_size=global_config.base_config.max_memory_size,
        alpha: int = 0.6,
        beta: int = 0.4,
        decay_steps: int = 100,
    ):
        self._max_size = max_size
        self._memory = SumTree(max_size)
        self._alpha = alpha  # Controls how much prioritization is used
        self._beta = beta  # Importance sampling weight
        self._decay_steps = decay_steps  # Steps after which beta is incremented
        self._beta_increment = self._compute_increment(beta, decay_steps)
        self._alpha_increment = self._compute_increment(alpha, decay_steps)

    def _compute_increment(self, value: float, train_steps: int) -> float:
        """Compute the increment for the value.

        Args:
            value (float): Initial value.
            train_steps (int): Training steps.

        Returns:
            float: Increment for the value.
        """
        return (1.0 - value) / (train_steps - 0.05 * train_steps)

    def add_transition(self, transitions_new: list) -> "PrioritizedMemory":
        """Add a new transition to the buffer.

        Args:
            transitions_new (list): New experiences.

        Returns:
            PrioritizedMemory: Own object
        """
        priority = self._memory.max() ** self._alpha  # Initial priority
        self._memory.add(priority, transitions_new)
        return self

    def sample(self, batch_size: int) -> tuple:
        """Sample a batch of transitions from the prioritized memory.

        Args:
            batch_size (int): Batch size.

        Returns:
            tuple: Tuple containing the batch, indices and importance weights
        """
        indices, priorities, batch = self._memory.sample_batch(batch_size, self._alpha)
        return np.asarray(batch), np.asarray(indices), np.asarray(priorities)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update the priorities of the transitions and scale the priorities by alpha.

        >>> new_priority = abs(td_error) ** alpha

        Args:
            indices (np.ndarray): Indices of the transitions.
            td_errors (np.ndarray): TD errors of the transitions.
        """
        for idx, error in zip(indices, np.abs(td_errors)):
            self._memory.update(idx, error**self._alpha)

    def importance_sampling_weights(self, indices: np.ndarray) -> np.ndarray:
        """Compute the importance sampling weights for the transitions.

        >>> w_i = (N * P(i)) ** -beta / max(w_i)

        Args:
            indices (np.ndarray): Indices of the transitions.

        Returns:
            np.ndarray: Importance sampling weights.
        """
        prios, _ = self._memory.get_leafs(indices)
        probs = prios / self._memory.total()
        weights = (self._memory.size * probs) ** -self._beta
        weights /= weights.max()
        return weights

    def weight_loss(self, loss: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
        """Weight the loss with the importance sampling weights.

        Args:
            loss (torch.Tensor): Loss to weight.
            indices (np.ndarray): Indices of the transitions.

        Returns:
            torch.Tensor: Weighted loss.
        """
        weights = self.importance_sampling_weights(indices)
        weights = torch.from_numpy(weights).float().to(loss.device)
        return torch.mean(loss * weights)

    def anneal(self):
        """Anneal the alpha and beta values by the defined increment."""
        self._beta = min(1.0, self._beta + self._beta_increment)
        self._alpha = min(1.0, self._alpha + self._alpha_increment)

    def get_all_transitions(self):
        return super().get_all_transitions()


class BalancedPrioritizedMemory(Memory):
    def __init__(
        self, max_size, alpha=0.6, beta=0.4, beta_increment=0.001, decay_steps=100
    ):
        super().__init__(max_size)
        self.alpha = alpha  # Controls how much prioritization is used
        self.beta = beta  # Importance sampling weight
        self.beta_increment = beta_increment  # Increment for beta
        self.decay_steps = decay_steps  # Steps after which memory strength decays
        self.priorities = np.zeros((max_size,), dtype=np.float32)  # Priority values
        self.memory_strength = np.ones((max_size,), dtype=np.float32)  # Memory strength
        self.max_priority = 1.0  # Initial max priority

    def add_transition(self, transitions_new: list) -> "BalancedPrioritizedMemory":
        """Add a new transitition to the buffer

        Args:
            transitions_new (list): New experiences.

        Returns:
            BalancedPrioritizedMemory: Own object
        """
        idx = self.current_idx
        super().add_transition(transitions_new)

        # Normalize reward to [0, 100] for prioritization
        reward = transitions_new[3]  # Reward is at index 3
        normalized_reward = self.normalize_reward(reward)

        # Set initial priority based on normalized reward
        self.priorities[idx] = (normalized_reward + 1e-5) ** self.alpha
        self.memory_strength[idx] = 1.0  # Reset memory strength for new transition
        self.max_priority = max(self.max_priority, self.priorities[idx])
        return self

    def sample(self, batch_size: int) -> tuple:
        """Sample from the balanced prioritized experience replay buffer.
        Therefore, the following steps are performed:

        1. Compute the effective priorities: prio x memory_strength
        2. Calculate probabilities: effective priorities / sum(effective priorities)
        3. Sample with probabilities
        4. Compute the importance weights: (size * p**-beta) / max(weights)
        5. Increase Beta
        6. Return the Batch

        Args:
            batch_size (int): The batch size to sample.

        Returns:
            tuple: Tuple containing the batch, indices and importance weights
        """
        effective_priorities = (
            self.priorities[: self.size] * self.memory_strength[: self.size]
        )
        probs = effective_priorities / effective_priorities.sum()
        indices = np.random.choice(self.size, batch_size, p=probs)

        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** -self.beta
        weights /= weights.max()

        # Increment beta for stability
        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = super().sample(batch_size, indices=indices)
        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, rewards: np.ndarray):
        """Update the priorities

        Args:
            indices (np.ndarray): Indices to update
            rewards (np.ndarray): Reward of the indices
        """
        # Scale Rewards to [0,100]
        normalized_rewards = self.normalize_reward(rewards).squeeze()
        self.priorities[indices] = (normalized_rewards + 1e-5) ** self.alpha
        self.max_priority = max(self.max_priority, self.priorities.max())

    def normalize_reward(self, rewards: np.ndarray) -> np.ndarray:
        """Normalizes the reward to [0, 100]

        Args:
            rewards (np.ndarray): Input Rewards

        Returns:
            np.ndarray: Normalized rewards
        """
        return (
            (rewards - np.min(self.transitions[:, 3]))
            / (np.max(self.transitions[:, 3]) - np.min(self.transitions[:, 3]) + 1e-5)
            * 100
        )

    def decay_memory_strength(self):
        """Decay the memory strength to balance early and late experiences."""
        self.memory_strength[: self.size] -= 1.0 / self.decay_steps
        self.memory_strength = np.clip(self.memory_strength, 0, 1)
