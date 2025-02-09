from agents.basic_hokey_oponent import BasicHokeyOpponentWrapper
from environments.advanced_reward_calculator import Weights
from environments.base_wrapper import BaseEnvWrapper
from environments.hokey_wrapper import HokeyEnvWrapper
from hockey.hockey_env import Mode
from utils.config import EnvironmentConfig


class EnvironmentFactory:
    @staticmethod
    def create_environment(
        env_name,
        max_steps: int,
        do_render: bool = False,
        mode: int = None,
        start_training_after_steps: int = 10000,
        weights: Weights = Weights(),
    ):
        if env_name == "Hockey-v0":
            mode = Mode.NORMAL if mode is None else mode
            env = HokeyEnvWrapper(
                max_steps,
                do_render,
                mode=mode,
                start_training_after_steps=start_training_after_steps,
                weights=weights,
            )
            return env
        else:
            return BaseEnvWrapper(
                env_name,
                max_steps,
                do_render,
                start_training_after_steps=start_training_after_steps,
            )
