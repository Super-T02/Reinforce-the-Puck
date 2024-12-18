import numpy as np
import torch
from components.memory import Batch, Memory
from training.trainer import Trainer
from utils.config import AgentConfig


class BaseAgent:
    """Base class for all agents."""

    def __init__(self, name: str, trainer: callable, config: AgentConfig):
        """
        Initialize the agent.

        Args:
            name (str): Name of the agent.
            trainer (callable): The trainer class.
        """
        self._name = name
        self._trainer: Trainer = trainer(config.trainer_config)
        self._config = config
        self._feedback_buffer = Memory(self._config.memory_size)

    def act(self, state) -> any:
        """
        Select an action based on the given state.

        Args:
            state: The current state of the environment.

        Returns:
            action: The selected action.
        """
        raise NotImplementedError

    def reset(self) -> "BaseAgent":
        """
        Reset the agent.

        Returns:
            BaseAgent: The agent object.
        """
        self._feedback_buffer = Memory(self._config.memory_size)
        return self

    def save_experience(
        self,
        state: any,
        action: any,
        new_state: any,
        reward: float,
        done: bool,
        trunc: bool,
        info: dict[str, any],
    ) -> "BaseAgent":
        """Save the experience tuple

        Args:
            state (any): The current state of the environment.
            action (any): The action taken in the current state.
            new_state (any): The new state of the environment.
            reward (float): The reward received for the action.
            done (bool): Whether the episode has ended.
            trunc (bool): Whether the episode was truncated.
            info (dict[str, any]): Additional information about the

        Returns:
            Agent: The agent object.
        """
        self._feedback_buffer.add_transition(
            [state, action, new_state, reward, done, trunc]
        )
        return self

    def train(self, last_reward: float = np.nan) -> "BaseAgent":
        """
        Trains the agent using the specified trainer.

        This method initiates the training process for the agent by calling the
        `train` method of the `_trainer` attribute. The training process involves
        running for a specified number of iterations, using the agent's sample
        method and train_step method.

        With the `last_reward` parameter, the reward can also be passed to the statistics.

        Args:
            last_reward (float, optional): The reward from the last episode. Defaults to np.nan.

        Returns:
            BaseAgent: The trained agent instance.
        """
        self._trainer.train(
            self,
            self._config.epochs,
            self.sample,
            self.train_step,
            {"reward": last_reward},
        )

    def sample(self, batch_size: int) -> Batch:
        """
        Sample a batch of experiences from the memory.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            Batch: The batch of experiences.
        """
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32))
        sample = self._feedback_buffer.sample(batch_size)
        return Batch(*[to_torch(x) for i, x in enumerate(sample) if i < 4])

    def train_step(self) -> dict:
        """
        Perform a single training step.

        Returns:
            dict: The training statistics.
        """
        raise NotImplementedError
