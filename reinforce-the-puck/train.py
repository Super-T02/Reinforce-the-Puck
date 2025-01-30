"""This file contains the CLI for the training process. Input is a yaml file with the configuration for the training."""

import argparse
import logging
import os
import time

import numpy as np
from agents.agent_factory import AgentFactory
from environments.base_wrapper import BaseEnvWrapper
from environments.environment_factory import EnvironmentFactory
from environments.hokey_wrapper import HokeyEnvWrapper
from utils import config_dir, logger
from utils.checkpoint import Checkpoint, CheckpointManager
from utils.config import AgentConfig, global_config


class TrainingRun:
    """Training Run class which trains, evaluates, saves the best and mutates the best agent."""

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
        self._checkpoint_manager_agent = CheckpointManager(environment.name, 5, "best")
        if isinstance(self._environment, HokeyEnvWrapper):
            self._checkpoint_manager_opponent = CheckpointManager(
                environment.name, 5, "best"
            )

    def run(self, num_runs: int = 1):
        """Run the training process.
        1. Train the agent.
        2. Evaluate the agent.
        3. Save the best agent.
        4. Mutate the best agent.
        """
        for i in range(num_runs):
            self.train()
            self.evaluate()
            if not self._agent_config.mutation_config.enabled:
                break
            self._mutate() if i < num_runs - 1 else None

    def train(self):
        """Train the agent in the environment."""
        self._logger.info("Starting training [%d]...", self._num_episodes)
        for i in range(self._num_episodes):
            self._environment.run_train_episode(i)
            if i % self._agent_config.eval_freq == 0 and i != 0:
                if not self._environment.started_training:
                    self._logger.info(
                        f"Currently: {self._environment._steps} of {self._environment._start_training_after_steps} steps ({self._environment._steps/self._environment._start_training_after_steps * 100} %)"
                    )
                    continue
                self.evaluate()
        self._logger.info("Training finished.")

    def evaluate(self):
        """Evaluate the agent in the environment."""
        if isinstance(self._environment, HokeyEnvWrapper):
            self._eval_agent_and_opponent()
        else:
            self._eval_agent()

    def _eval_agent(self) -> None:
        self._logger.info("Starting evaluation...")
        rewards = self._environment.evaluate(self._agent_config.eval_episodes)
        self._logger.info("Evaluation finished.")
        self._environment.agent.save_eval_result(rewards)
        mean_reward = np.mean(rewards)
        self._add_checkpoint(mean_reward)
        self._checkpoint_manager_agent.save_last_checkpoint()
        self._checkpoint_manager_agent.save_best_checkpoint()

    def _eval_agent_and_opponent(self) -> None:
        self._logger.info("Starting evaluation...")
        rewards_agent, rewards_opponent = self._environment.evaluate(
            self._agent_config.eval_episodes
        )
        self._logger.info(
            f"Evaluation finished. Agent: {np.mean(rewards_agent)}, Opponent: {np.mean(rewards_opponent)}"
        )
        self._environment.agent.save_eval_result(rewards_agent)
        self._environment.opponent_agent.save_eval_result(rewards_opponent)
        mean_agent, mean_opponent = np.mean(rewards_agent), np.mean(rewards_opponent)
        self._add_checkpoint(mean_agent, mean_opponent)
        self._checkpoint_manager_agent.save_last_checkpoint(), self._checkpoint_manager_opponent.save_last_checkpoint()
        self._checkpoint_manager_agent.save_best_checkpoint(), self._checkpoint_manager_opponent.save_best_checkpoint()

    def _add_checkpoint(self, mean_reward: float, mean_reward_opponent: float = None):
        """Adds a checkpoint to the Manager."""
        checkpoint = Checkpoint(
            self._environment.agent,
            self._environment.name,
            mean_reward,
        )
        self._checkpoint_manager_agent.add_checkpoint(checkpoint)

        # Add opponent
        if mean_reward_opponent is not None:
            checkpoint_opponent = Checkpoint(
                self._environment.opponent_agent,
                self._environment.name,
                mean_reward_opponent,
            )
            self._checkpoint_manager_opponent.add_checkpoint(checkpoint_opponent)

    def _mutate(self):
        """Mutate the best agent."""
        self._logger.info("Mutating the agent...")
        best_agent_config = self._checkpoint_manager_agent.get_best_config(
            self._agent_config.type
        )
        self._agent_config = best_agent_config.mutate()
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

        for agent_config in global_config.get_agent_configs():
            env = next(
                (
                    env
                    for env in global_config.get_environments()
                    if agent_config.env_id == env.id
                ),
                None,
            )
            if env is None:
                self._logger.error(
                    "No environment found for agent %s", agent_config.name
                )
                continue
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
                mode=env.mode,
                start_training_after_steps=env.start_training_after_steps,
            )

            env.agent = AgentFactory.create_agent_from_config(
                agent_config, env.observation_space, env.action_space
            )

            if env.name == "Hockey-v0":
                env.opponent_agent = AgentFactory.create_agent_from_config(
                    global_config.get_opponent_config(agent_config.opponent_name),
                    env.observation_space,
                    env.action_space,
                )
                logging.info(
                    "Created opponent agent. Type: %s",
                    type(env.opponent_agent).__name__,
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
