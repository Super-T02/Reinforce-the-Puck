"""This file contains the tournament logic, where agents can compete against each other to determine the best one."""

import argparse
import datetime
import logging
import os
import random
import time

import numpy as np
import pandas as pd
import tabulate
import yaml
from agents.agent_factory import AgentFactory
from agents.base_agent import BaseAgent
from agents.basic_hokey_oponent import BasicHokeyOpponentWrapper
from environments.hokey_wrapper import HokeyEnvWrapper
from openskill.models import PlackettLuce, PlackettLuceRating
from utils import config_dir, logger, logs_dir


class Player:
    def __init__(self, name: str, agent: BaseAgent, rate_obj: PlackettLuceRating):
        self.name: str = name
        self.agent: BaseAgent = agent
        try:
            self.type = agent.get_config().type
        except:
            self.type = "Unknown"  # basic hokey opponent have no config
        self.rate_obj: PlackettLuceRating = rate_obj


class Tournament:
    def __init__(self, n_epochs: int, do_render: bool = False):
        self._model = PlackettLuce()
        self._agents: dict[str, Player] = {}
        self._logger = logging.getLogger(__name__)
        self._stats = pd.DataFrame(
            columns=["game", "p1", "p2", "p1_score", "p2_score", "winner"]
        )

        self._do_render: bool = do_render
        self._n_epochs: int = n_epochs

    def from_yaml(self, path: str) -> None:
        """Loads the agents from a yaml file.

        Args:
            path (str): The path to the yaml file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist.")

        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        o_space, a_space = (
            HokeyEnvWrapper(1).observation_space,
            HokeyEnvWrapper(1).action_space,
        )

        for agent_name, c in cfg.items():
            self._logger.info(
                f"Loading agent {agent_name} from checkpoint {c['checkpoint']}"
            )
            agent = AgentFactory.create_agent_from_checkpoint(
                c["checkpoint"], c["type"], o_space, a_space
            )
            agent.set_name(agent_name)
            self.add_agent(agent)

    def add_agent(self, agent: BaseAgent) -> None:
        if agent.get_name() in self._agents:
            raise ValueError(f"Agent with name {agent.get_name()} already exists.")
        name = agent.get_name()
        self._agents[name] = Player(name, agent, self._model.rating(name=name))

    def add_benchmarks(self) -> None:
        """Adds the benchmark agents to the tournament."""
        weak, strong = BasicHokeyOpponentWrapper(True), BasicHokeyOpponentWrapper(False)
        self.add_agent(weak)
        self.add_agent(strong)

    def remove_agent(self, name: str) -> None:
        if name not in self._agents:
            raise ValueError(f"Agent with name {name} does not exist.")
        del self._agents[name]

    def single_simulation(self, p1: Player, p2: Player) -> tuple[Player, Player, str]:
        """Given two players, simulates a single game between them and returns the updated players.

        Args:
            p1 (Player): First player.
            p2 (Player): Second player.

        Returns:
            tuple[Player, Player, str]: Rerated players. The winner of the game.
        """
        env = HokeyEnvWrapper(
            max_steps=100000,
            do_render=self._do_render,
            agent=p1.agent,
            opponent_agent=p2.agent,
            winner_weight=1,
            closeness_puck_weight=0,
            touch_puck_weight=0,
            puck_direction_weight=0,
        )
        env._logger.setLevel(logging.ERROR)
        p1_score, p2_score = env.evaluate(self._n_epochs)
        p1_score, p2_score = int(np.sum(p1_score)), int(np.sum(p2_score))
        p1_new, p2_new = self._model.rate(
            [[p1.rate_obj], [p2.rate_obj]], scores=[p1_score, p2_score]
        )
        p1.rate_obj = p1_new[0]
        p2.rate_obj = p2_new[0]
        won = (
            p1.name
            if p1_score > p2_score
            else p2.name
            if p2_score > p1_score
            else "Tie"
        )
        self._stats.loc[len(self._stats)] = [
            len(self._stats),
            p1.name,
            p2.name,
            p1_score,
            p2_score,
            won,
        ]
        return p1, p2, won

    def simulate(self, n_games: int = 100, max_skill_gap: int = 25) -> None:
        """Simulates the tournament with the given number of parallel games.

        Args:
            n_games (int, optional): Number of games to simulate. Defaults to 100.
            max_skill_gap (int, optional): Maximum skill gap between two players. Defaults to 25.
        """
        for i in range(n_games):
            p1, p2 = self.sample_players(int(len(self._agents) * 0.3), max_skill_gap)
            self._logger.info(
                f"Starting game {i + 1}/{n_games}. {p1.name} vs {p2.name}"
            )
            p1, p2, won = self.single_simulation(p1, p2)
            self._logger.info(
                f"Game {i + 1}/{n_games} finished. Won {won}. {p1.name} rating: {p1.rate_obj.mu}, {p2.name} rating: {p2.rate_obj.mu}"
            )
            self._agents[p1.name] = p1
            self._agents[p2.name] = p2

    def sample_players(
        self, tournament_size: int = 4, max_skill_gap: int = 25
    ) -> tuple[Player, Player]:
        """Samples two players from the matchmaking pool. Selects the players based on their skill level.
        A player with a higher skill level is more likely to be selected:

        1. Sample `tournament_size` players from the pool.
        2. Calculate the probability of selecting each player based on their skill level. (softmax)
        3. Sample two players in the pool based on the probabilities from step 2.
        4. Ensure that the skill gap between the two players is less than `max_skill_gap`.

        If for some reason the skill gap is too large, repeat the process up to 100 times. (This is to avoid infinite loops)

        Args:
            tournament_size (int, optional): The number of players to sample from. Defaults to 4. If it is equal to 2, it is equivalent to uniformly sampling two players.
            max_skill_gap (int, optional): The maximum skill gap between the two players. Defaults to 25.

        Returns:
            tuple[Player, Player]: Two players.
        """
        if len(self._agents) < 2:
            raise ValueError("Not enough agents to sample from.")
        max_skill_gap = max(0, max_skill_gap)
        tournament_size = max(2, tournament_size)
        for _ in range(100):
            pool = random.sample([a for a in self._agents.values()], tournament_size)
            mus = [a.rate_obj.mu for a in pool]
            probs = np.exp(mus) / np.sum(np.exp(mus))
            idx = np.random.choice(len(pool), size=2, p=probs, replace=False)
            selected = [pool[i] for i in idx]

            # Ensure that the skill gap is not too large
            if abs(selected[0].rate_obj.mu - selected[1].rate_obj.mu) <= max_skill_gap:
                return selected[0], selected[1]

        raise ValueError(
            "Could not find two players with a skill gap less than the maximum skill gap."
        )

    def get_rating(self, name: str) -> float:
        """Returns the rating of the agent with the given name.

        Args:
            name (str): The name of the agent.

        Returns:
            float: The rating of the agent.
        """
        if name not in self._agents:
            raise ValueError(f"Agent with name {name} does not exist.")
        return self._agents[name].rate_obj.mu, self._agents[name].rate_obj.sigma

    def show_result_table(self):
        """Prints the result table of the tournament."""
        players = self._agents.values()
        players = sorted(players, key=lambda x: x.rate_obj.mu, reverse=True)

        table = []
        for player in players:
            table.append(
                [
                    player.name,
                    player.type,
                    player.rate_obj.mu,
                    player.rate_obj.sigma,
                    self.get_player_avg_score(player.name),
                ]
            )

        print(
            tabulate.tabulate(
                table, headers=["Name", "Type", "Rating", "Sigma", "Avg. Score"]
            )
        )

    def get_player_avg_score(self, name: str) -> float:
        """Returns the average score of the player with the given name.

        Args:
            name (str): The name of the player.

        Returns:
            float: The average score of the player.
        """
        return (
            self._stats[self._stats["p1"] == name]["p1_score"].mean()
            + self._stats[self._stats["p2"] == name]["p2_score"].mean()
        )

    def save_stats(self, path: str) -> None:
        """Save the statistics of the tournament to a file.

        Args:
            path (str): The path to the file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._stats.to_csv(path, index=False)
        self._logger.info(f"Saved the tournament statistics to {path}.")


if __name__ == "__main__":
    logger.init_logger(os.path.join(config_dir, "logging.yaml"))

    parser = argparse.ArgumentParser(description="Tournament runner.")
    parser.add_argument("config", type=str, help="Path to the config file.")
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of epochs to simulate per game.",
    )
    parser.add_argument("--n_games", type=int, default=100, help="Number of games.")
    parser.add_argument("--render", action="store_true", help="Render the games.")

    start = time.perf_counter()
    args = parser.parse_args()
    tournament = Tournament(args.n_epochs, args.render)
    tournament.from_yaml(args.config)
    tournament.add_benchmarks()
    tournament.simulate(n_games=args.n_games)
    tournament.show_result_table()
    tournament.save_stats(
        os.path.join(
            logs_dir,
            "tournaments",
            f"tournament_results{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
        )
    )
    end = time.perf_counter()
    print(f"Finished in {end - start:.2f} seconds.")
