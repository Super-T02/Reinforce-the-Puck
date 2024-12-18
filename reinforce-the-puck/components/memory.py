import numpy as np
from utils.config import global_config


class Memory:
    """Memory buffer to store transitions."""

    def __init__(self, max_size=global_config.base_config.max_memory_size):
        """Initialize the Memory object.

        Args:
            max_size (int, optional): Maximum size of the memory buffer. Defaults to config.MAX_MEMORY_SIZE.
        """
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size = max_size

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

        self.transitions[self.current_idx, :] = np.asarray(
            transitions_new, dtype=object
        )
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size
        return self

    def sample(self, batch: int = 1):
        """Sample a batch of transitions from the memory.

        Args:
            batch (int, optional): Batch size. Defaults to 1.

        Returns:
            np.ndarray: Batch of transitions.
        """
        if batch > self.size:
            batch = self.size
        self.inds = np.random.choice(range(self.size), size=batch, replace=False)
        return self.transitions[self.inds, :]

    def get_all_transitions(self):
        """Get all transitions from the memory.

        Returns:
            np.ndarray: All transitions.
        """
        return self.transitions[0 : self.size]
