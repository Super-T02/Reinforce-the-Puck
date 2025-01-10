import datetime
import logging
import multiprocessing
import os
import pickle
from typing import List

import aiofiles
import numpy as np
from components.memory import Batch
from evaluation.tensorboard_statistics import TensorboardStatistics
from utils import logs_dir
from utils.config import TrainerConfig


class BaseTrainer:
    def __init__(self, trainer_config: TrainerConfig):
        self._checkpoint_dir = trainer_config.checkpoint_dir
        self._learning_rate = trainer_config.learning_rate
        self._batch_size = trainer_config.batch_size
        self._log_freq = trainer_config.log_freq
        self._save_checkpoint_freq = trainer_config.save_checkpoint_freq
        self._max_checkpoints = trainer_config.max_checkpoints
        self._epochs = trainer_config.epochs
        self._log_name = trainer_config.log_name
        self._checkpoints = []
        self._logger = logging.getLogger(__name__)
        self._train_iterations = 0
        self._is_config_logged = False

        self._tensorboard: TensorboardStatistics = TensorboardStatistics(
            os.path.join(
                os.path.join(logs_dir, "tensorboard"),
                self.__generate_training_name(self._epochs),
            )
        )

    def save_checkpoint(self, checkpoint_name: str):
        """
        Saves the agent's state to a checkpoint file.

        Args:
            agent (BaseAgent): The agent to save.
            checkpoint_name (str): The name of the checkpoint file.

        Returns:
            None
        """
        checkpoint_path = os.path.join(self._checkpoint_dir, checkpoint_name + ".pth")
        self.save(checkpoint_path)

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

    async def save_statistics_async(self, statistics: dict, filename: str):
        """
        Save the training statistics to a file.

        Args:
            statistics (dict): The statistics data to be saved.
            filename (str): The path to the file where the statistics will be saved.

        Returns:
            None
        """

        file_path = os.path.join(logs_dir, filename)
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(pickle.dumps(statistics))

    def save_eval_result(self, rewards: List[float]):
        """Save the evaluation results to the TensorBoard."""
        self._tensorboard.write_tensorboard_statistics_async(
            self._train_iterations,
            {
                "eval/reward": np.mean(rewards),
                "eval/max_reward": np.max(rewards),
                "eval/min_reward": np.min(rewards),
            },
        )

    def train(self, last_reward: float = np.nan) -> List[dict]:
        """
        Train the agent for a specified number of iterations.
        """
        self._train_iterations += 1
        statistics: List[dict] = []
        env_stats = {"reward": last_reward}

        if not self._is_config_logged:
            config = self._get_train_config()
            self._tensorboard.save_hyper_parameters(config)
            self._is_config_logged = True

        for iteration in range(self._epochs):
            batch = self.sample(self._batch_size)
            statistic = self.train_step(batch)
            env_stats.update(statistic)
            statistic = env_stats

            if "loss" not in statistic:
                raise ValueError(
                    "The training step must return a dictionary with a 'loss' key."
                )

            statistics.append(statistic)
            if iteration % self._log_freq == 0:
                # self.save_statistics_async(
                #     self.__convert_dicts_to_lists(statistics),
                #     f"stats_{self.__generate_training_name(iteration)}",
                # )

                self._tensorboard.write_tensorboard_statistics_async(
                    iteration, statistic
                )

            if iteration % self._save_checkpoint_freq == 0:
                self.save_checkpoint(
                    f"checkpoint_{self.__generate_training_name(iteration)}"
                )

        return self.__convert_dicts_to_lists(statistics)["loss"]

    def __convert_dicts_to_lists(self, data) -> dict:
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

    def __generate_training_name(self, i: int) -> str:
        """
        Helper function that generates a training name based on the current training parameters, iteration and timestamp.

        Args:
            iter (int): The current iteration number.

        Returns:
            str: A string representing the training name, formatted with the timestamp, learning rate,
                 batch size, log frequency, save checkpoint frequency, and iteration steps.
        """
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{time_stamp}_{self._log_name}_{i}-steps"

    def get_logging_dir(self) -> str:
        """Get the directory where the logs are stored."""
        return self._tensorboard.writer.log_dir

    def train_step(self, batch: Batch):
        """
        Perform a single training step.

        Args:
            batch (Batch): The batch of experiences.

        Returns:
            dict: The training statistics.
        """
        raise NotImplementedError

    def sample(self, batch_size: int) -> Batch:
        """
        Sample a batch of experiences for training.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            Batch: A batch of sampled experiences.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """
        Save the current state to the specified path.

        Args:
            path (str): The file path where the state should be saved.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def _get_train_config(self) -> dict:
        """Get the training configuration as a dictionary."""
        self._logger.warning(
            "The get_train_config method is not overwritten by the subclass. Not all training parameters are saved."
        )
        return {
            "learning_rate": self._learning_rate,
            "batch_size": self._batch_size,
            "log_freq": self._log_freq,
            "save_checkpoint_freq": self._save_checkpoint_freq,
            "max_checkpoints": self._max_checkpoints,
            "epochs": self._epochs,
            "log_name": self._log_name,
        }

    def __del__(self):
        self._tensorboard.close()
