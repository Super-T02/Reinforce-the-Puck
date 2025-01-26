import logging
import os

from agents.base_agent import BaseAgent
from utils import model_dir
from utils.config import AgentConfig


class Checkpoint:
    def __init__(
        self, agent: BaseAgent, environment: str, avg_eval_reward: float
    ) -> "Checkpoint":
        """Initializes the checkpoint

        Args:
            agent (BaseAgent): Agent the checkpoint should be represent.
            environment (str): Environment Name.
            avg_eval_reward (float): Reward this checkpoint earned at evaluation.

        Returns:
            Checkpoint: Checkpoint obj.
        """
        self.agent = agent
        self.environment = environment
        self.avg_eval_reward = avg_eval_reward

    def save(self, path) -> None:
        """Saves the checkpoint to the base path

        Args:
            path (str): Path to save the checkpoint.
        """
        path = self.get_path() if path is None else path
        os.makedirs(os.sep.join(path.split(os.sep)[:-1]), exist_ok=True)
        self.agent.save(f"{path}.pth")
        self.agent.get_config().to_yaml(f"{path}_config.yaml")

    def get_path(self) -> str:
        """Returns the path of the checkpoint

        Returns:
            str: Path of the checkpoint.
        """
        return os.path.join(
            model_dir,
            self.environment,
            self.agent.get_config().type,
            self.agent.get_name(),
            "checkpoint",
        )


class CheckpointManager:
    """
    Checkpoint manager for automatic saving and loading of model checkpoints.
    This includes the saving of the best model on a given type + environment.
    """

    def __init__(
        self, environment_name: str, max_checkpoints: int = 5, best_path: str = "best"
    ):
        self._checkpoints: list[Checkpoint] = []
        self._best_checkpoint: dict[str, Checkpoint] = {}
        self._max_checkpoints = max_checkpoints
        self._best_path = best_path
        self._environment_name = environment_name
        self._logger = logging.getLogger(__name__)

    def add_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Adds a checkpoint to the manager

        Args:
            checkpoint (Checkpoint): Checkpoint to add.
        """
        if len(self._checkpoints) >= self._max_checkpoints:
            self._checkpoints.pop(0)
        self._checkpoints.append(checkpoint)
        self._add_best_checkpoint(checkpoint)

    def _add_best_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Adds a checkpoint to the best checkpoint list

        Args:
            checkpoint (Checkpoint): Checkpoint to add.
        """
        agent_type = checkpoint.agent.get_config().type
        if agent_type not in self._best_checkpoint:
            self._best_checkpoint[agent_type] = checkpoint
        elif (
            checkpoint.avg_eval_reward
            > self._best_checkpoint[agent_type].avg_eval_reward
        ):
            self._best_checkpoint[agent_type] = checkpoint

    def save_last_checkpoint(self) -> None:
        """Saves the last checkpoint to the base path"""
        if len(self._checkpoints) == 0:
            self._logger.warning("No checkpoints to save.")
            return
        to_save: Checkpoint = self._checkpoints[-1]
        path = to_save.get_path()
        self._logger.info(f"Saving last checkpoint to {path}")
        to_save.save(path + "_last")

    def save_best_checkpoint(self) -> None:
        """Saves the best checkpoint to the best path"""
        agent_types = list(set([c.agent.get_config().type for c in self._checkpoints]))
        for t in agent_types:
            best_checkpoint = max(
                [c for c in self._checkpoints if c.agent.get_config().type == t],
                key=lambda x: x.avg_eval_reward,
            )
            path = best_checkpoint.get_path() + "_best"
            self._logger.info(
                f"Saving best checkpoint for {t}: {best_checkpoint.avg_eval_reward} to: {path}"
            )
            best_checkpoint.save(path)

    def get_best_path(self, agent_type: str) -> str:
        """Generates the path for the best checkpoint.

        Args:
            agent_type (str): Type of the agent.

        Returns:
            str: Path for the best Checkpoint.
        """
        return os.path.join(
            model_dir, self._environment_name, agent_type, self._best_path, "best"
        )

    def get_best_config(self, agent_type: str) -> AgentConfig:
        """Get the best config for the agent type based on the **RUN**

        Args:
            agent_type (str): Type of the agent.

        Returns:
            AgentConfig: Config of the best agent.
        """
        agent = self.get_best_agent(agent_type)
        if agent is None:
            return None
        return agent.get_config()

    def get_best_agent(self, agent_type: str) -> BaseAgent:
        """Get the best agent for the agent type based on the **RUN**

        Args:
            agent_type (str): Type of the agent.

        Returns:
            BaseAgent: Best agent.
        """
        if agent_type not in self._best_checkpoint:
            return None
        return self._best_checkpoint[agent_type].agent
