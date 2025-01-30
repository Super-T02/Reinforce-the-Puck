from agents.basic_hokey_oponent import BasicHokeyOpponentWrapper
from environments.base_wrapper import BaseEnvWrapper
from environments.hokey_wrapper import HokeyEnvWrapper
from hockey.hockey_env import Mode


class EnvironmentFactory:
    @staticmethod
    def create_environment(
        env_name,
        max_steps: int,
        do_render: bool = False,
        mode: int = None,
        start_training_after_steps: int = 10000,
    ):
        if env_name == "Hockey-v0":
            mode = Mode.NORMAL if mode is None else mode
            env = HokeyEnvWrapper(
                max_steps,
                do_render,
                mode=mode,
                start_training_after_steps=start_training_after_steps,
            )
            return env
        else:
            return BaseEnvWrapper(
                env_name,
                max_steps,
                do_render,
                start_training_after_steps=start_training_after_steps,
            )
