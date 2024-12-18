import logging
import os
import pickle
from typing import Callable, List

import torch
from agents.base_agent import BaseAgent
from components.memory import Batch
from evaluation.tensorboard_statistics import TensorboardStatistics
from utils import logs_dir


class Trainer:
    def __init__(
        self,
        checkpoint_dir: str,
        learning_rate: float,
        batch_size: int,
        log_freq: int,
        save_checkpoint_freq: int,
        max_checkpoints: int = 5,
    ):
        self._checkpoint_dir = checkpoint_dir
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._log_freq = log_freq
        self._save_checkpoint_freq = save_checkpoint_freq
        self._max_checkpoints = max_checkpoints
        self._checkpoints = []
        self._logger = logging.getLogger(__name__)

    def save_checkpoint(self, agent: BaseAgent, checkpoint_name: str):
        """
        Saves the agent's state to a checkpoint file.

        Args:
            agent (BaseAgent): The agent to save.
            checkpoint_name (str): The name of the checkpoint file.

        Returns:
            None
        """
        checkpoint_path = os.path.join(self._checkpoint_dir, checkpoint_name + ".pth")
        agent.save(checkpoint_path)

        # Add the new checkpoint to the list
        self._checkpoints.append(checkpoint_path)

        # Remove old checkpoints if exceeding max_checkpoints
        if len(self._checkpoints) > self._max_checkpoints:
            oldest_checkpoint = self._checkpoints.pop(0)
            if os.path.exists(oldest_checkpoint):
                try:
                    os.remove(oldest_checkpoint)
                except OSError as e:
                    self._logger.error(
                        f"Error deleting checkpoint {oldest_checkpoint}: {e}"
                    )

    def save_statistics(self, statistics, filename: str):
        """
        Save the training statistics to a file.

        Args:
            statistics (dict): The statistics data to be saved.
            filename (str): The path to the file where the statistics will be saved.

        Returns:
            None
        """

        with open(filename, "w") as f:
            pickle.dump(statistics, f)

    def train(
        self,
        agent: BaseAgent,
        iter_fit: int,
        sample_batch: Callable[[int], Batch],
        training_step: Callable[[BaseAgent, Batch], dict],
        env_stats: dict = {},
    ) -> List[dict]:
        """
        Trains the agent.

        Args:
            agent (BaseAgent): The agent to train.
            iter_fit (int): The number of iterations to fit the agent.
            sample_batch (Callable[[int], Batch]):
            A function that samples a batch of data for training (e.g. from replay buffer).
            training_step (Callable[[BaseAgent, Batch], dict]): A function that performs a single training step.
            env_stats (dict, optional): The last statistics from the previous environmment run. Defaults to {}.
        Returns:
            List[dict]: A list of dictionaries containing the loss values for each iteration.
        """
        statistics: List[dict] = []
        tensorboard: TensorboardStatistics = TensorboardStatistics(
            os.path.join(logs_dir, self.__generate_training_name(iter_fit))
        )

        for iteration in range(iter_fit):
            batch = sample_batch(self._batch_size)
            statistic = training_step(agent, batch)
            env_stats.update(statistic)
            statistic = env_stats

            if "loss" not in statistic:
                raise ValueError(
                    "The training step must return a dictionary with a 'loss' key."
                )

            statistics.append(statistic)
            if iteration % self._log_freq == 0:
                self.save_statistics(
                    self.__convert_dicts_to_lists(statistics),
                    f"loss_{self.__generate_training_name(iteration)}",
                )
                self._logger.info(f"Iteration {iteration}: {statistic}")
                tensorboard.write_tensorboard_statistics(iteration, statistic)

            if iteration % self._save_checkpoint_freq == 0:
                self.save_checkpoint(
                    agent, f"checkpoint_{self.__generate_training_name(iteration)}"
                )

        tensorboard.close()
        return self.__convert_dicts_to_lists(statistics)["loss"]

    def __convert_dicts_to_lists(data):
        """
        Convert a list of dictionaries into a dictionary of lists.
        This function takes a list of dictionaries and converts it into a dictionary
        where each key maps to a list of values corresponding to that key from the
        input dictionaries.
        Args:
            data (list of dict): A list of dictionaries to be converted.
        Returns:
            dict: A dictionary where each key maps to a list of values from the input dictionaries.
                  If the input list is empty, an empty dictionary is returned.
        """
        if not data:
            return {}

        # Dictionary comprehension: Erstellt Listen für jeden Key aus den Dictionaries
        result = {key: [d[key] for d in data] for key in data[0]}
        return result

    def __generate_training_name(self, iter: int) -> str:
        """
        Helper function that generates a training name based on the current training parameters and iteration.

        Args:
            iter (int): The current iteration number.

        Returns:
            str: A string representing the training name, formatted with the learning rate,
                 batch size, log frequency, save checkpoint frequency, and iteration steps.
        """
        return f"{self._learning_rate}_{self._batch_size}_{self._log_freq}_{self._save_checkpoint_freq}_{iter}-steps"
