"""This file contains the CLI for the training process. Input is a yaml file with the configuration for the training."""

import argparse
import logging
import os

import numpy as np
from agents.base_agent import BaseAgent
from agents.ddpg import DDPGAgent
from environments.wrapper import EnvWrapper, VectorizedEnvWrapper
from utils import config_dir, logger
from utils.config import global_config


class TrainCLI:
    def __init__(self):
        self._logger = logging.getLogger(__name__)
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
        """Load the agent, and environment classes."""
        # TODO: Add support for multiple agents
        type2agent = {
            "ddpg": DDPGAgent,
        }
        # @Jona: Kp ob das hier so schön ist
        self._environment = VectorizedEnvWrapper(
            self._args.environment,
            type2agent[global_config.agent1.type],
            {
                "config": global_config.agent1,
            },
        )
        self._agent = self._environment.agent
        self._episodes = (
            global_config.training.num_episodes
            if self._args.num_episodes is None
            else self._args.num_episodes
        )

    def setup(self):
        """Setup the CLI."""
        self.load_config()
        self.load_classes()

    def run(self):
        """
        Run the CLI.
        """
        self.setup()
        self._logger.info("Starting training [%d]...", self._episodes)
        rewards = np.array(
            [self._environment.run_train_episode(i) for i in range(self._episodes)]
        )
        self._logger.info("Training finished.")
        self._logger.info("Mean reward: %4.2f", rewards.mean())
        self._logger.info(
            "Max reward: %4.2f [Episode: %10d]", rewards.max(), rewards.argmax()
        )
        self._logger.info(
            "Min reward: %4.2f [Episode: %10d]", rewards.min(), rewards.argmin()
        )
        self._environment.close()


if __name__ == "__main__":
    logger.init_logger(os.path.join(config_dir, "logging.yaml"))
    cli = TrainCLI()
    cli.run()
