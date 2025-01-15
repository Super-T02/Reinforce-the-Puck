from agents.base_agent import BaseAgent
from environments.base_wrapper import BaseEnvWrapper


class HokeyEnvWrapper(BaseEnvWrapper):
    def __init__(
        self,
        env_name: str,
        agent: BaseAgent,
        max_steps: int,
        do_render: bool = False,
        checkpoint: str = None,
    ):
        pass
