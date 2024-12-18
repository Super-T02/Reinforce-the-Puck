"""This file contains the CLI for the training process. Input is a yaml file with the configuration for the training."""

import argparse
import logging
import os

import numpy as np
from agents.base_agent import BaseAgent
from environments.wrapper import EnvWrapper
from training.trainer import Trainer
from utils import config_dir, logger
from utils.config import global_config


class TrainCLI:
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._trainer = None
        self._agent = None
        self._environment = None
        self._parser = argparse.ArgumentParser(description="Train the agent.")

        self._parser.add_argument(
            "-e",
            "--env",
            dest="environment",
            type=str,
            default="Pendulum-v1",
            required=False,
            help="Name of the environment to train the agent.",
        )

        self._parser.add_argument(
            "-i",
            "--num_episodes",
            type=int,
            default=None,
            required=False,
            help="Number of episodes to train the agent.",
        )

        self._parser.add_argument(
            "-c",
            "--config_path",
            type=str,
            required=False,
            default=os.path.join(config_dir, "config.yaml"),
            help="Path to the configuration file.",
        )

        self._args = self._parser.parse_args()

    def load_config(self):
        """Load the configuration from a YAML file."""
        global_config.from_yaml(self._args.config_path)
        self._logger.info("Loaded configuration from %s", self._args.config_path)
        self._logger.info("Configuration: \n%s", global_config.to_dict())

    def load_classes(self):
        """Load the trainer, agent, and environment classes."""
        self._trainer = Trainer(
            global_config.trainer.checkpoint_dir,
            global_config.trainer.learning_rate,
            global_config.trainer.batch_size,
            global_config.trainer.log_freq,
            global_config.trainer.save_checkpoint_freq,
            global_config.trainer.max_checkpoints,
        )
        # TODO: Add support for multiple agents
        # TODO: Use implemented agent
        self._agent = BaseAgent(
            global_config.agent1.name, self._trainer, global_config.agent1
        )
        self._environment = EnvWrapper(self._args.environment, self._agent)
        self._episodes = self._args.num_episodes

    def setup(self):
        """Setup the CLI."""
        self.load_config()
        self.load_classes()

    def run_train_episode(self, i: int) -> int:
        """Run a single episode of the training.

        Args:
            i (int): The episode number.

        Returns:
            int: The total reward of the episode.
        """
        reward = 0
        self._environment.reset()
        done = False
        while not done:
            ret = self._environment.step()
            done = ret[2]
            reward += ret[1]
        self._logger.info("Episode %10d: Total reward: %4.2f", i, reward)
        return reward

    def run(self):
        """
        Run the CLI.
        """
        self.setup()
        self._logger.info(
            "Starting training [%d]...", global_config.base_config.num_episodes
        )
        rewards = np.array(
            [
                self.run_train_episode(i)
                for i in range(global_config.base_config.num_episodes)
            ]
        )
        self._logger.info("Training finished.")
        self._logger.info("Mean reward: %4.2f", rewards.mean())
        self._logger.info(
            "Max reward: %4.2f [Episode: %10d]", rewards.max(), rewards.argmax()
        )
        self._logger.info(
            "Min reward: %4.2f [Episode: %10d]", rewards.min(), rewards.argmin()
        )


if __name__ == "__main__":
    logger.init_logger(os.path.join(config_dir, "logging.yaml"))
    cli = TrainCLI()
    cli.run()
