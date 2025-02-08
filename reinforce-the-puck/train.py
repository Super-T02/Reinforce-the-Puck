"""This file contains the CLI for the training process. Input is a yaml file with the configuration for the training."""

import argparse
import logging
import os

import numpy as np
from agents.agent_factory import AgentFactory
from agents.base_agent import BaseAgent
from agents.basic_hokey_oponent import BasicHokeyOpponentWrapper
from environments.advanced_reward_calculator import Weights
from environments.base_wrapper import BaseEnvWrapper
from environments.environment_factory import EnvironmentFactory
from environments.hokey_wrapper import HokeyEnvWrapper
from utils import config_dir, logger
from utils.checkpoint import Checkpoint, CheckpointManager
from utils.config import AgentConfig, EnvironmentConfig, global_config


class TrainingRun:
    """Training Run class which trains, evaluates, saves the best and mutates the best agent."""

    def __init__(
        self,
        environment: BaseEnvWrapper,
        agent_config: AgentConfig,
        num_episodes: int,
        agents: list[BaseAgent],
        new_agents_after_eval: int = 0,
        train_all: bool = False,
    ):
        self._environment = environment
        self._agent_config = agent_config
        self._num_episodes = num_episodes
        self._logger = logging.getLogger(__name__)
        self._checkpoint_manager_agent = CheckpointManager(environment.name, 5, "best")
        self._agents = agents  # At 0 is the agent, at 1 ... n are the opponents
        self._new_agents_after_eval = new_agents_after_eval
        self._train_all = train_all
        self._last_eval_results = []
        self._keep_last_n_evals = 3
        if self.use_opponent:
            self._checkpoint_manager_opponent = CheckpointManager(
                environment.name, 5, "best"
            )

    @property
    def use_opponent(self) -> bool:
        """Use opponent for training."""
        return (
            isinstance(self._environment, HokeyEnvWrapper)
            and self._environment.train_both
        )

    def run(self, num_runs: int = 1):
        """Run the training process.
        1. Train the agent.
        2. Evaluate the agent.
        3. Save the best agent.
        4. Mutate the best agent.
        """
        for i in range(num_runs):
            self.warmup()
            self.train()
            self.evaluate()
            if not self._agent_config.mutation_config.enabled:
                break
            self._mutate() if i < num_runs - 1 else None

    def train(self):
        """Train the agent in the environment."""
        self._logger.info("Starting training [%d]...", self._num_episodes)
        evals = 0
        # Start training
        for i in range(self._num_episodes):
            self._environment.run_train_episode(i)

            if i % self._agent_config.eval_freq == 0 and i != 0:
                self.evaluate()
                evals += 1
                if isinstance(self._environment, HokeyEnvWrapper):
                    if (
                        self._new_agents_after_eval > 0
                        and evals % self._new_agents_after_eval == 0
                    ):
                        self.change_agents()
                    elif self._check_convergence():
                        self.change_agents()
                        evals = 0

        self._logger.info("Training finished.")

    def _check_convergence(self):
        """Check if the agent has converged."""
        if len(self._last_eval_results) == self._keep_last_n_evals:
            mean_rewards = np.array(self._last_eval_results)
            mean_rewards = mean_rewards[:, 0]
            if np.std(mean_rewards) < 0.1:
                self._logger.info("Agent has converged.")
                return True
        return False

    def warmup(self):
        """Warmup the agent in the environment."""
        while not self._environment.started_training:
            self._environment.run_train_episode(-1, train=False)

            if isinstance(self._environment, HokeyEnvWrapper):
                (
                    self._environment.agent,
                    self._environment.opponent_agent,
                ) = self._select_random_agent_and_opponent(self._agents)

        if isinstance(self._environment, HokeyEnvWrapper):
            self._environment.agent = self._agents[0]
            self._environment.opponent_agent = self._agents[1]

    def evaluate(self):
        """Evaluate the agent in the environment."""
        if isinstance(self._environment, HokeyEnvWrapper):
            self._eval_hockey()
        else:
            self._eval_default()

    def change_agents(self):
        """Change the agents."""
        (
            self._environment.agent,
            self._environment.opponent_agent,
        ) = self._select_random_agent_and_opponent(self._agents)
        self._last_eval_results = []
        self._logger.info(
            f"Changed agents. Agent={self._environment.agent.get_name()}, Opponent={self._environment.opponent_agent.get_name()}"
        )

    def _eval_default(self) -> None:
        self._logger.info("Starting evaluation...")
        rewards = self._environment.evaluate(self._agent_config.eval_episodes)
        self._logger.info("Evaluation finished.")
        self._environment.agent.save_eval_result(rewards)
        mean_reward = np.mean(rewards)
        self._add_checkpoint(mean_reward)
        self._checkpoint_manager_agent.save_last_checkpoint()
        self._checkpoint_manager_agent.save_best_checkpoint()
        self._add_eval_result(mean_reward)

    def _eval_hockey(self) -> None:
        self._logger.info("Starting evaluation...")
        rewards_agent, rewards_opponent = self._environment.evaluate(
            self._agent_config.eval_episodes
        )
        self._logger.info(
            f"Evaluation finished. Agent: {np.mean(rewards_agent)}, Opponent: {np.mean(rewards_opponent)}"
        )
        self._environment.agent.save_eval_result(rewards_agent)
        mean_agent, mean_opponent = np.mean(rewards_agent), np.mean(rewards_opponent)
        self._add_checkpoint(mean_agent, mean_opponent)
        self._checkpoint_manager_agent.save_last_checkpoint()
        self._checkpoint_manager_agent.save_best_checkpoint()

        if self.use_opponent and not isinstance(
            self._environment.opponent_agent, BasicHokeyOpponentWrapper
        ):
            self._environment.opponent_agent.save_eval_result(rewards_opponent)
            self._checkpoint_manager_opponent.save_last_checkpoint()
            self._checkpoint_manager_opponent.save_best_checkpoint()

        self._add_eval_result(mean_agent, mean_opponent)

    def _add_eval_result(self, mean_reward: float, mean_reward_opponent: float = None):
        """Add the evaluation result to the list."""
        self._last_eval_results.append((mean_reward, mean_reward_opponent))
        if len(self._last_eval_results) > self._keep_last_n_evals:
            self._last_eval_results.pop(0)

    def _add_checkpoint(self, mean_reward: float, mean_reward_opponent: float = None):
        """Adds a checkpoint to the Manager."""
        if not isinstance(self._environment.agent, BasicHokeyOpponentWrapper):
            checkpoint = Checkpoint(
                self._environment.agent,
                self._environment.name,
                mean_reward,
            )
            self._checkpoint_manager_agent.add_checkpoint(checkpoint)

        # Add opponent
        if mean_reward_opponent is not None and self.use_opponent:
            if not isinstance(
                self._environment.opponent_agent, BasicHokeyOpponentWrapper
            ):
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

    def _select_random_agent_and_opponent(self, agents: list[BaseAgent]):
        """Select a random agent and opponent."""
        # If we train only the agent, select a random opponent
        num_agents = len(agents)
        agent = agents[0]
        opponent = np.random.choice(agents[1:])

        # If we train all agents, select a random agent and opponent
        if self._train_all:
            agent_i = np.random.choice(range(num_agents))
            probs = [1 / (num_agents - 1)] * num_agents
            probs[agent_i] = 0
            opponent_i = np.random.choice(num_agents, p=probs)
            if agent_i == opponent_i:
                raise ValueError("Agent and opponent should not be the same.")
            agent = agents[agent_i]
            opponent = agents[opponent_i]

            # If the agent is a BasicHokeyOpponentWrapper, swap the agents or select base agent
            if isinstance(agent, BasicHokeyOpponentWrapper) and isinstance(
                opponent, BasicHokeyOpponentWrapper
            ):
                agent = agents[0]
            elif isinstance(agent, BasicHokeyOpponentWrapper):
                agent, opponent = opponent, agent
        return agent, opponent


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
            env_config: EnvironmentConfig = next(
                (
                    env
                    for env in global_config.get_environments()
                    if agent_config.env_id == env.id
                ),
                None,
            )
            if env_config is None:
                self._logger.error(
                    "No environment found for agent %s", agent_config.name
                )
                continue
            env_name = (
                env_config.env_name
                if self._args.environment is None
                else self._args.environment
            )
            max_steps = (
                env_config.max_steps
                if self._args.max_steps is None
                else self._args.max_steps
            )
            env = EnvironmentFactory.create_environment(
                env_name=env_name,
                max_steps=max_steps,
                do_render=agent_config.specialized_config.do_render,
                mode=env_config.mode,
                start_training_after_steps=env_config.start_training_after_steps,
                weights=Weights.from_config(env_config.weights),
            )

            env.agent = AgentFactory.create_agent_from_config(
                agent_config, env.observation_space, env.action_space
            )
            agents = [env.agent]
            if env.name == "Hockey-v0":
                agents = self.generate_opponents(
                    env.agent,
                    agent_config.opponent_names,
                    env,
                )
                env.opponent_agent = agents[1]
                env.train_both = env_config.train_both
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
                agents=agents,
                new_agents_after_eval=env_config.new_agents_after_eval,
                train_all=env_config.train_all,
            )
            # self.training_runs.append(training_run)
            training_run.run(agent_config.num_runs)

    def generate_opponents(
        self, agent: BaseAgent, opponent_names: list[str], env: HokeyEnvWrapper
    ) -> list[BaseAgent]:
        """Generate agents for the training.

        Args:
            agent (BaseAgent): The agent to train.
            opponent_names (list[str]): List of opponent names.
            env (HokeyEnvWrapper): The environment.

        Returns:
            list[BaseAgent]: List of agents.
        """
        if not isinstance(env, HokeyEnvWrapper):
            raise ValueError("Environment should be a subclass of HokeyEnvWrapper")
        if not type(opponent_names) == list:
            raise ValueError("Opponent names should be a list")

        agents = [agent]
        for opponent_name in opponent_names:
            print("Opponent Name: ", opponent_name)
            agent = AgentFactory.create_agent_from_config(
                global_config.get_opponent_config(opponent_name),
                env.observation_space,
                env.action_space,
            )
            agents.append(agent)
        return agents

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
