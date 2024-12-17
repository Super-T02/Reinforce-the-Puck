from training.trainer import BaseTrainer


class BaseAgent:
    """Base class for all agents."""

    def __init__(self, name: str, trainer: BaseTrainer, config: dict):
        """
        Initialize the agent.

        Args:
            name (str): Name of the agent.
            trainer (object): The trainer object that trains the agent.
            config (dict): Configuration parameters for the agent.
        """
        self._name = name
        self._config = config
        self._trainer = trainer

    def act(self, state) -> any:
        """
        Select an action based on the given state.

        Args:
            state: The current state of the environment.

        Returns:
            action: The selected action.
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
        # Todo: Implement when memory is merged
        raise NotImplementedError

    def train(self) -> "BaseAgent":
        """Learn from the last iteration.

        Returns:
            Agent: The agent object.
        """
        raise NotImplementedError
