"""This file contains the CLI for the training process. Input is a yaml file with the configuration for the training."""

import argparse
import logging
import os
from calendar import c

import numpy as np
from agents.agent_factory import AgentFactory
from environments.base_wrapper import BaseEnvWrapper
from environments.environment_factory import EnvironmentFactory
from utils import config_dir, logger, model_dir
from utils.config import AgentConfig, global_config


class TrainingRun:
    def __init__(
        self,
        environment: BaseEnvWrapper,
        agent_config: AgentConfig,
        num_episodes: int,
    ):
        self._environment = environment
        self._agent_config = agent_config
        self._num_episodes = num_episodes
        self._logger = logging.getLogger(__name__)
        self._best_config = None
        self._best_agent = None
        self._best_reward = -np.inf
        self.reset()

    def reset(self):
        self._best_agent_current_run = None
        self._best_reward_current_run = -np.inf

    def run(self, num_runs: int = 1):
        """Run the training process.
        1. Train the agent.
        2. Evaluate the agent.
        3. Save the best agent.
        4. Mutate the agent.
        """
        for i in range(num_runs):
            self.train()
            self.evaluate()
            self.save_run_best_agent()
            self.save_best_agent()
            if not self._agent_config.mutation_config.enabled:
                break
            self._mutate() if i < num_runs - 1 else None

    def train(self):
        """Train the agent in the environment."""
        self._logger.info("Starting training [%d]...", self._num_episodes)
        rewards = []
        for i in range(self._num_episodes):
            rewards += [self._environment.run_train_episode(i)]
            if i % self._agent_config.eval_freq == 0:
                self.evaluate()
        rewards = np.array(rewards)
        self._logger.info("Training finished.")

    def evaluate(self):
        """Evaluate the agent in the environment."""
        self._logger.info("Starting evaluation...")
        rewards = self._environment.evaluate(self._agent_config.eval_episodes)
        self._logger.info("Evaluation finished.")
        self._environment.agent.save_eval_result(rewards)
        mean_reward = np.mean(rewards)
        if mean_reward > self._best_reward:
            self._best_reward = mean_reward
            self._best_agent = self._environment.agent
            self.save_best_agent()
        if mean_reward > self._best_reward_current_run:
            self._best_reward_current_run = mean_reward
            self._best_agent_current_run = self._environment.agent
            self.save_run_best_agent()

    def save_run_best_agent(self):
        """Save the best agent to a file."""
        cfg = self._best_agent_current_run.get_config()
        self._best_agent_current_run.save(
            os.path.join(
                model_dir,
                self._environment.name,
                cfg.type,
                self._best_agent_current_run.get_name(),
                "best_agent_run.pth",
            )
        )
        cfg.to_yaml(
            os.path.join(
                model_dir,
                self._environment.name,
                cfg.type,
                self._best_agent_current_run.get_name(),
                "best_config_run.yaml",
            )
        )

    def save_best_agent(self):
        """Save the best agent to a file."""
        self._best_config = self._best_agent.get_config()
        self._best_agent.save(
            os.path.join(
                model_dir,
                self._environment.name,
                self._best_config.type,
                self._best_agent.get_name(),
                "best_agent_mutation.pth",
            )
        )
        self._best_config.to_yaml(
            os.path.join(
                model_dir,
                self._environment.name,
                self._best_config.type,
                self._best_agent.get_name(),
                "best_config_mutation.yaml",
            )
        )

    def _mutate(self):
        """Mutate the agent."""
        self._logger.info("Mutating the agent...")
        self._agent_config.mutate()
        self._environment.agent = AgentFactory.create_agent_from_config(
            self._agent_config,
            self._environment.observation_space,
            self._environment.action_space,
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
            env = EnvironmentFactory.create_environment(
                env_name=env_name,
                max_steps=max_steps,
                do_render=agent_config.specialized_config.do_render,
            )

            env.agent = AgentFactory.create_agent_from_config(
                agent_config, env.observation_space, env.action_space
            )

            training_run = TrainingRun(
                environment=env,
                agent_config=agent_config,
                num_episodes=(
                    agent_config.specialized_config.num_episodes
                    if self._args.num_episodes is None
                    else self._args.num_episodes
                ),
            )
            # self.training_runs.append(training_run)
            training_run.run(agent_config.num_runs)

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
