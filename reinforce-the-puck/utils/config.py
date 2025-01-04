import os

import torch
import yaml
from utils import model_dir

global global_config

dtype_map = {
    "float32": torch.float32,
    "float": torch.float,  # alias for float32
    "float64": torch.float64,
    "double": torch.double,  # alias for float64
    "float16": torch.float16,
    "half": torch.half,  # alias for float16
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int16": torch.int16,
    "short": torch.short,  # alias for int16
    "int32": torch.int32,
    "int": torch.int,  # alias for int32
    "int64": torch.int64,
    "long": torch.long,  # alias for int64
    "bool": torch.bool,
}


####################################################################################################
# Base Classes
####################################################################################################
class ConfigGroup:
    """
    Base class for configuration groups.
    Only attributes defined in the class will be populated from the YAML file.
    """

    def update_from_dict(self, config_dict: dict):
        for key, value in config_dict.items():
            # just update the attribute if it exists
            if hasattr(self, key):
                current_attr = getattr(self, key)

                # if the current attribute is a ConfigGroup and the value is a dictionary, then update recursively
                if isinstance(current_attr, ConfigGroup) and isinstance(value, dict):
                    current_attr.update_from_dict(value)
                else:
                    # otherwise, just update the attribute
                    setattr(self, key, value)

    def to_dict(self) -> dict:
        """Convert the configuration group to a dictionary.

        Returns:
            dict: The configuration dictionary.
        """
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("__") and k not in ["update_from_dict", "to_dict"]
        }


####################################################################################################
# Configuration Groups
####################################################################################################


class BaseConfig(ConfigGroup):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float32
        self.num_episodes = 1
        self.max_memory_size = 100000

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value: str | torch.dtype):
        if isinstance(value, torch.dtype):
            self._dtype = value
            return
        self._dtype = dtype_map.get(value.lower(), torch.float32)


class AgentConfig(ConfigGroup):
    def __init__(self):
        self.type = "none"
        self.name = "BasicOpponent"
        self.version = 1
        self.epochs = 10
        self.env_id = 0
        self.memory_size = 10000
        self.trainer_config: TrainerConfig = TrainerConfig()
        self.specialized_config: BaseConfig = (
            BaseConfig()
        )  # todo: ggf extra config erstellen


class SACAgentConfig(AgentConfig):
    def __init__(self):
        super().__init__()
        self.type = "sac"
        self.eps = 0.2  # Noise level
        self.tau = 0.005  # Target network update rate (Soft update)
        self.memory_size = 100000
        self.discount = 0.95
        self.alpha = 0.2  # Entropy regularization
        self.trainer_config.batch_size = 128
        self.trainer_config.learning_rate_actor = 0.00001
        self.trainer_config.learning_rate_critic = 0.0001
        self.actor_hidden_sizes = [128, 128]
        self.critic_hidden_sizes = [128, 128, 64]
        self.update_target_every = 100
        self.use_target_net = True


class DDPGAgentConfig(AgentConfig):
    def __init__(self):
        super().__init__()
        self.type = "ddpg"
        self.eps = 0.2  # Noise level
        self.memory_size = 100000
        self.discount = 0.95
        self.trainer_config.batch_size = 128
        self.trainer_config.learning_rate_actor = 0.00001
        self.trainer_config.learning_rate_critic = 0.0001
        self.actor_hidden_sizes = [128, 128]
        self.critic_hidden_sizes = [128, 128, 64]
        self.update_target_every = 100
        self.use_target_net = True


class EnvironmentConfig(ConfigGroup):
    def __init__(self):
        self.max_steps = 1000
        self.env_name = "unnamed"
        self.id = -1


class TrainerConfig(ConfigGroup):
    def __init__(self):
        self.checkpoint_dir = os.path.join(model_dir, "checkpoints")
        self.learning_rate = 0.001
        self.batch_size = 32
        self.log_freq = 10
        self.save_checkpoint_freq = 100
        self.max_checkpoints = 5
        self.epochs = 100
        self.log_name = "unnamed"
        self.id = -1


####################################################################################################
# Configuration Class
####################################################################################################


class Config:
    """
    Main configuration class to load and manage YAML configurations.
    """

    TYPE2AGENT = {
        "ddpg": DDPGAgentConfig,
        "sac": SACAgentConfig,
    }

    def __init__(self):
        self.base_config = BaseConfig()

    def get_agents(self) -> list[AgentConfig]:
        agents = [attr for attr in dir(self) if attr.startswith("agent")]
        return [
            getattr(self, agent)
            for agent in agents
            if isinstance(getattr(self, agent), AgentConfig)
        ]

    def get_environments(self) -> list[EnvironmentConfig]:
        envs = [attr for attr in dir(self) if attr.startswith("env")]
        return [
            getattr(self, env)
            for env in envs
            if isinstance(getattr(self, env), EnvironmentConfig)
        ]

    def from_yaml(self, yaml_path: str):
        """
        Load configuration from a YAML file and populate the defined groups.

        Args:
            yaml_path (str): Path to the YAML configuration file.
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)

        for group_name, group_config in data.items():
            if group_name.startswith("agent"):
                agent_config = self.TYPE2AGENT[group_config["type"]]
                setattr(self, group_name, agent_config())
            if group_name.startswith("env"):
                env_config = EnvironmentConfig()
                setattr(self, group_name, env_config)

            if hasattr(self, group_name):
                group = getattr(self, group_name)
                if isinstance(group, ConfigGroup):
                    group.update_from_dict(group_config)

    def to_dict(self) -> dict:
        """
        Convert the configuration to a dictionary for inspection or debugging.

        Returns:
            dict: The configuration dictionary.
        """
        return {
            k: v.to_dict()
            for k, v in self.__dict__.items()
            if not k.startswith("__") and k not in ["from_yaml", "to_dict"]
        }


global_config: Config = Config()
