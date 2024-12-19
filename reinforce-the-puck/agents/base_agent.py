import numpy as np
import torch
from components.memory import Batch, Memory
from gymnasium import spaces
from training.trainer import Trainer
from utils.config import AgentConfig


class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible"""

    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)


class BaseAgent:
    """Base class for all agents."""

    def __init__(
        self,
        name: str,
        trainer: callable,
        observation_space: spaces.box.Box,
        action_space: spaces.box.Box,
        config: AgentConfig,
    ):
        """
        Initialize the agent.

        Args:
            name (str): Name of the agent.
            trainer (callable): The trainer class.
        """
        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace(
                "Observation space {} incompatible "
                "with {}. (Require: Box)".format(observation_space, self)
            )
        if not isinstance(action_space, spaces.box.Box):
            raise UnsupportedSpace(
                "Action space {} incompatible with {}."
                " (Require Box)".format(action_space, self)
            )

        self._name = name
        self._trainer: Trainer = trainer(config.trainer_config)
        self._config = config

        self._feedback_buffer = Memory(self._config.memory_size)
        self._observation_space = observation_space
        self._action_space = action_space
        self._obs_dim = self._observation_space.shape[0]
        self._action_n = self._action_space.shape[0]

    def save(self, path: str) -> None:
        """
        Save the agent to a file.

        Args:
            path (str): The path to the file where the agent will be saved.

        Returns:
            None
        """
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load the agent from a file.

        Args:
            path (str): The path to the file where the agent is saved.

        Returns:
            None
        """
        raise NotImplementedError

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
        raise NotImplementedError

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

    def to_torch(self, x: np.ndarray) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.

        Args:
            x (np.ndarray): The numpy array to convert.

        Returns:
            torch.Tensor: The PyTorch tensor.
        """
        return torch.from_numpy(x.astype(np.float32))

    def sample(self, batch_size: int) -> Batch:
        """
        Sample a batch of experiences from the memory.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            Batch: The batch of experiences.
        """
        sample = self._feedback_buffer.sample(batch_size)
        state = self.to_torch(np.vstack(sample[:, 0]))
        action = self.to_torch(np.vstack(sample[:, 1]))
        next_state = self.to_torch(np.vstack(sample[:, 2]))
        reward = self.to_torch(np.vstack(sample[:, 3]))
        done = self.to_torch(np.vstack(sample[:, 4]))
        return Batch(state, action, next_state, reward, done)

    def train_step(self, batch: Batch) -> dict:
        """
        Perform a single training step.

        Args:
            batch (Batch): The batch of experiences.

        Returns:
            dict: The training statistics.
        """
        raise NotImplementedError
