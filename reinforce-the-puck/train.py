"""This file contains the CLI for the training process. Input is a yaml file with the configuration for the training."""

import argparse
import copy
import logging
import os

import numpy as np
from agents.ddpg import DDPGAgent
from agents.sac import SACAgent
from agents.td3 import TD3Agent
from environments.wrapper import EnvWrapper
from utils import config_dir, logger, model_dir
from utils.config import AgentConfig, global_config


class TrainingRun:
    def __init__(
        self,
        environment: EnvWrapper,
        agent: AgentConfig,
        num_episodes: int,
    ):
        self._environment = environment
        self._agent_config = agent
        self._num_episodes = num_episodes
        self._logger = logging.getLogger(__name__)
        self._best_agent = None
        self._best_reward = -np.inf

    def run(self):
        self._logger.info("Starting training [%d]...", self._num_episodes)
        rewards = []
        for i in range(self._num_episodes):
            rewards += [self._environment.run_train_episode(i)]
            if i % self._agent_config.eval_freq == 0:
                self.evaluate()
        rewards = np.array(rewards)
        self._logger.info("Training finished.")
        self._logger.info("Mean reward: %4.2f", rewards.mean())
        self._logger.info(
            "Max reward: %4.2f [Episode: %10d]", rewards.max(), rewards.argmax()
        )
        self._logger.info(
            "Min reward: %4.2f [Episode: %10d]", rewards.min(), rewards.argmin()
        )
        self.evaluate()
        self.save_best_agent()

    def evaluate(self):
        """Evaluate the agent in the environment."""
        self._logger.info("Starting evaluation...")
        rewards = self._environment.evaluate(self._agent_config.eval_episodes)
        self._logger.info("Evaluation finished.")
        self._environment.agent.save_eval_result(rewards)
        mean_reward = np.mean(rewards)
        if mean_reward > self._best_reward:
            self._best_reward = mean_reward
            self._best_agent = copy.copy(self._environment.agent)
            self.save_best_agent()

    def save_best_agent(self):
        """Save the best agent to a file."""
        self._best_agent.save(
            os.path.join(
                model_dir,
                self._environment.name,
                self._agent_config.type,
                self._best_agent.get_name() + "_best_agent.pth",
            )
        )


class TrainCLI:
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self.training_runs: list[TrainingRun] = []
        self._parser = argparse.ArgumentParser(description="Train the agent.")

        self._parser.add_argument(
            "-e",
            "--env",
            dest="environment",
            type=str,
            default=None,
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
            "-m",
            "--max_steps",
            type=int,
            default=None,
            required=False,
            help="Maximum number of steps per episode.",
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

        type2agent = {"ddpg": DDPGAgent, "sac": SACAgent, "td3": TD3Agent}

        for agent_config in global_config.get_agents():
            env = next(
                (
                    env
                    for env in global_config.get_environments()
                    if agent_config.env_id == env.id
                ),
                None,
            )
            env_name = (
                env.env_name
                if self._args.environment is None
                else self._args.environment
            )
            max_steps = (
                env.max_steps if self._args.max_steps is None else self._args.max_steps
            )
            env = EnvWrapper(
                env_name=env_name,
                max_steps=max_steps,
                checkpoint=agent_config.checkpoint,
                agent_class=type2agent[agent_config.type],
                kwargs_agent={
                    "config": agent_config,
                },
            )
            training_run = TrainingRun(
                environment=env,
                agent=agent_config,
                num_episodes=(
                    agent_config.specialized_config.num_episodes
                    if self._args.num_episodes is None
                    else self._args.num_episodes
                ),
            )
            # self.training_runs.append(training_run)
            training_run.run()

    def setup(self):
        """Setup the CLI."""
        self.load_config()
        # self.load_classes()

    def run(self):
        """
        Run the CLI.
        """
        self.setup()
        self.load_classes()  # Aka run

        # for training_run in self.training_runs:
        #     rewards = training_run.run()


if __name__ == "__main__":
    logger.init_logger(os.path.join(config_dir, "logging.yaml"))
    cli = TrainCLI()
    cli.run()
