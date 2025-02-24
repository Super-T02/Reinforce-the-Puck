"""
File: final_evaluation.py
Author: Tom Freudenmann
Content: This file contains final evaluation of the agents.
"""

import argparse
import datetime
import logging
import os
import time

import numpy as np
import pandas as pd
import tabulate
import yaml
from agents.agent_factory import AgentFactory
from agents.base_agent import BaseAgent
from agents.basic_hokey_oponent import BasicHokeyOpponentWrapper
from environments.advanced_reward_calculator import Weights
from environments.hokey_wrapper import HokeyEnvWrapper
from utils import config_dir, logger, logs_dir, workspace_dir


class Player:
    def __init__(self, name: str, agent: BaseAgent):
        self.name: str = name
        self.agent: BaseAgent = agent
        self.type = agent.get_config().type


class FinalEvaluation:
    def __init__(self, n_epochs: int, do_render: bool = False):
        self._agents: dict[str, Player] = {}
        self._logger = logging.getLogger(__name__)
        self._stats = pd.DataFrame(
            columns=["game", "p1", "p2", "p1_score", "p2_score", "game-type"]
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
        self._agents[name] = Player(name, agent)

    def single_simulation(self, p1: Player, p2: Player) -> tuple[Player, Player, str]:
        """Given two players, simulates a single game between them and returns the updated players.

        Args:
            p1 (Player): First player.
            p2 (Player): Second player.

        Returns:
            tuple[Player, Player, str]: Rerated players. The winner of the game.
        """
        weights = Weights(
            winner_weight=1.0,
            closeness_puck_weight=0.0,
            touch_puck_weight=0.0,
            puck_direction_weight=0.0,
            no_touch_penalty=0.0,
        )

        env = HokeyEnvWrapper(
            max_steps=100000,
            do_render=self._do_render,
            agent=p1.agent,
            opponent_agent=p2.agent,
            weights=weights,
        )
        env._logger.setLevel(logging.ERROR)
        p1_score, p2_score = env.evaluate(self._n_epochs)
        p1_score, p2_score = int(np.sum(p1_score)), int(np.sum(p2_score))
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
            "$".join([p1.name, p2.name]),
        ]
        return p1, p2, won

    def simulate(self, p1: Player, p2: Player, n_games: int = 100) -> None:
        """Simulates the epoch with the given number of games.

        Args:
            n_games (int, optional): Number of games to simulate. Defaults to 100.
        """
        for i in range(n_games):
            self._logger.info(
                f"Starting game {i + 1}/{n_games}. {p1.name} vs {p2.name}"
            )
            p1, p2, won = self.single_simulation(p1, p2)
            self._logger.info(f"Game {i + 1}/{n_games} finished. Won {won}.")

    def run(self, n_games: int = 100) -> None:
        """Runs the tournament with the given number of games.

        Args:
            n_games (int, optional): Number of games to simulate. Defaults to 100.
        """
        weak = Player("weak", BasicHokeyOpponentWrapper(True))
        strong = Player("strong", BasicHokeyOpponentWrapper(False))
        opponents = [weak, strong]
        for opponent in opponents:
            for a in self._agents.values():
                if a == opponent:
                    continue
                self.simulate(a, opponent, n_games)

    def show_result_table(self):
        """Prints the result table of the tournament."""
        players = self._agents.values()

        table = []
        for player in players:
            table.append(
                [
                    player.name,
                    player.type,
                    self.get_player_avg_score(player.name),
                ]
            )

        print(
            tabulate.tabulate(
                table, headers=["Name", "Type", "Average Score"], tablefmt="pretty"
            )
        )

    def get_player_avg_score(self, name: str) -> float:
        """Returns the average score of the player with the given name.

        Args:
            name (str): The name of the player.

        Returns:
            float: The average score of the player, or 0 if no scores are available.
        """
        p1_scores = self._stats[self._stats["p1"] == name]["p1_score"]
        p2_scores = self._stats[self._stats["p2"] == name]["p2_score"]
        total_scores = pd.concat([p1_scores, p2_scores])

        if total_scores.empty:
            return np.nan  # Return 0 if no scores are available

        return total_scores.mean()

    def generate_report_table(self, path: str, input_path: str = None) -> None:
        """Generate the latex table for the report.

        Args:
            path (str): The path to the file.
        """
        if input_path is not None:
            self._stats = pd.read_csv(input_path)

        result_df = pd.DataFrame(columns=["Agent", "Opponent", "Win-Rate (%)"])

        for game in self._stats["game-type"].unique():
            p1, p2 = game.split("$")
            game_data = self._stats[self._stats["game-type"] == game]
            p1_wins = len(game_data[game_data["p1_score"] > game_data["p2_score"]])
            p2_wins = len(game_data[game_data["p2_score"] > game_data["p1_score"]])
            ties = len(game_data[game_data["p1_score"] == game_data["p2_score"]])
            total_games = p1_wins + p2_wins + ties
            win_rate = (p1_wins / total_games) * 100
            # tie_rate = (ties / total_games) * 100
            result_df.loc[len(result_df)] = [p1, p2, win_rate]

        # Generate opponent | agent table
        result_df = result_df.pivot(
            columns="Agent", index="Opponent", values=["Win-Rate (%)"]
        )
        result_df = result_df.sort_index(axis=0)

        print(result_df)
        result_df.to_latex(
            path,
            float_format="%.2f",
            bold_rows=True,
            multicolumn_format="c",
            multicolumn=True,
            escape=True,
            caption="Performance between the agents and the baseline opponents.} \\centering {",
            column_format="l" + "c" * len(result_df.columns.levels[1]),
            label="tab:performance_opponents",
            na_rep="-",
        )

    def save_stats(self, path: str) -> None:
        """Save the statistics of the evaluation to a file.

        Args:
            path (str): The path to the file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._stats.to_csv(path, index=False)
        self._logger.info(f"Saved the results statistics to {path}.")


if __name__ == "__main__":
    # Initialize and run the tournament
    logger.init_logger(os.path.join(config_dir, "logging.yaml"))

    parser = argparse.ArgumentParser(description="Evaluation runner.")
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
    eval_instance = FinalEvaluation(args.n_epochs, args.render)
    eval_instance.from_yaml(args.config)
    eval_instance.run(n_games=args.n_games)
    eval_instance.show_result_table()
    eval_instance.save_stats(
        os.path.join(
            logs_dir,
            "finale_evaluation",
            f"final_evaluation_results{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
        )
    )
    eval_instance.generate_report_table(
        os.path.join(
            workspace_dir,
            "report",
            "tables",
            f"performance_opponents.tex",
        )
    )
    end = time.perf_counter()
    print(f"Finished in {end - start:.2f} seconds.")
