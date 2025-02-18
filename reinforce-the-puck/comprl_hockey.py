from __future__ import annotations

import argparse
import uuid

import hockey.hockey_env as h_env
import numpy as np
from agents.agent_factory import AgentFactory
from comprl.client import Agent, launch_client
from gymnasium.spaces.box import Box


class RandomAgent(Agent):
    """A hockey agent that simply uses random actions."""

    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class HockeyAgent(Agent):
    """A hockey agent that can be weak or strong."""

    def __init__(self, weak: bool) -> None:
        super().__init__()

        self.hockey_agent = h_env.BasicOpponent(weak=weak)

    def get_step(self, observation: list[float]) -> list[float]:
        # NOTE: If your agent is using discrete actions (0-7), you can use
        # HockeyEnv.discrete_to_continous_action to convert the action:
        #
        # from hockey.hockey_env import HockeyEnv
        # env = HockeyEnv()
        # continuous_action = env.discrete_to_continous_action(discrete_action)

        action = self.hockey_agent.act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class NewAgent(Agent):
    def __init__(self, checkpoint_path: str, agent_type: str) -> None:
        super().__init__()

        observation_space = h_env.HockeyEnv().observation_space
        action_space = h_env.HockeyEnv().action_space
        action_space = Box(
            action_space.low[:4],
            action_space.high[:4],
            (4,),
            action_space.dtype,
        )
        self.agent = AgentFactory.create_agent_from_checkpoint(
            checkpoint_path, agent_type, observation_space, action_space
        )
        print(f"Agent loaded from {checkpoint_path}")

    def get_step(self, observation: list[float]) -> list[float]:
        with self.agent.evaluate_context():
            action = self.agent.act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent-type",
        type=str,
        choices=["weak", "strong", "random", "td3", "sac", "moe"],
        default="weak",
        help="Which agent to use.",
    )
    parser.add_argument(
        "--agent-checkpoint", type=str, help="Path to the agent checkpoint."
    )
    args = parser.parse_args(agent_args)

    # Initialize the agent based on the arguments.
    agent: Agent
    if args.agent_type == "weak":
        agent = HockeyAgent(weak=True)
    elif args.agent_type == "strong":
        agent = HockeyAgent(weak=False)
    elif args.agent_type == "random":
        agent = RandomAgent()
    else:
        agent = NewAgent(args.agent_checkpoint, args.agent_type)
    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
