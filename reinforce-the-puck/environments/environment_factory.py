from agents.base_agent import BaseAgent
from agents.basic_hokey_oponent import BasicHokeyOpponentWrapper
from environments.base_wrapper import BaseEnvWrapper
from environments.hokey_wrapper import HokeyEnvWrapper


class EnvironmentFactory:
    @staticmethod
    def create_environment(env_name, max_steps: int, do_render: bool = False):
        if env_name == "Hockey-v0":
            env = HokeyEnvWrapper(max_steps, do_render)
            # todo: make opponent agent configurable
            env.opponent_agent = BasicHokeyOpponentWrapper(weak=False)
            return env
        else:
            return BaseEnvWrapper(env_name, max_steps, do_render)
