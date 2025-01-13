import os
import random

import numpy as np
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

    def to_yaml(self, yaml_path: str):
        """
        Save the configuration to a YAML file.

        Args:
            yaml_path (str): The path to the YAML file.
        """
        with open(yaml_path, "w") as file:
            yaml.dump(self.to_dict(), file)


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
        if value is None:
            self._dtype = None
            return

        if isinstance(value, torch.dtype):
            self._dtype = value
            return
        self._dtype = dtype_map.get(value.lower(), torch.float32)

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["_dtype"] = str(self.dtype)
        return dict_


class MutationConfig(ConfigGroup):
    def __init__(self):
        self.enabled = True
        self.means = [0, 0, 0]
        self.vars = [0.1, 0.1, 0.1]
        self.mins = [0.000001, 16, 1000]
        self.maxs = [0.1, 256, 100000]
        self.prob = 0.3
        self.parameters = ["learning_rate", "batch_size", "memory_size"]

    def get_min(self, i):
        val = self.mins[i]
        if val == "inf":
            return np.inf
        elif val == "-inf":
            return -np.inf
        return val

    def get_max(self, i):
        val = self.maxs[i]
        if val == "inf":
            return float("inf")
        elif val == "-inf":
            return -float("inf")
        return val

    def clip(self, value, i):
        return min(self.get_max(i), max(self.get_min(i), value))


class AgentConfig(ConfigGroup):
    def __init__(self):
        self.type = "none"
        self.name = "BasicOpponent"
        self.checkpoint = None
        self.version = 1
        self.epochs = 10
        self.eval_freq = 100
        self.eval_episodes = 10
        self.env_id = 0
        self.memory_size = 10000
        self.num_runs = 1
        self.mutation_config: MutationConfig = MutationConfig()
        self.trainer_config: TrainerConfig = TrainerConfig()

        self.specialized_config: BaseConfig = (
            BaseConfig()
        )  # Set all public attributes to None to enable inheritance from BaseConfig (See from_yaml)
        for attr in dir(self.specialized_config):
            if not attr.startswith("_") and not callable(
                getattr(self.specialized_config, attr)
            ):
                setattr(self.specialized_config, attr, None)

    def to_dict(self):
        dict_ = super().to_dict()
        dict_["trainer_config"] = self.trainer_config.to_dict()
        dict_["specialized_config"] = self.specialized_config.to_dict()
        dict_["mutation_config"] = self.mutation_config.to_dict()
        return dict_

    def mutate(self):
        """Mutate the agent configuration."""
        num_mutations = 0
        runs = 0
        while num_mutations < 1:
            for i, param in enumerate(self.mutation_config.parameters):
                param_value = getattr(
                    self, param, getattr(self.trainer_config, param, None)
                )
                if param_value is None:
                    continue
                if random.random() < self.mutation_config.prob:
                    mutation_rate = self.mutation_config.means[i]
                    mutationstds = self.mutation_config.vars[i]
                    self._mutate_param(
                        param, param_value, mutation_rate, mutationstds, i
                    )
                    num_mutations += 1
            runs += 1
            if runs >= 100:
                raise ValueError(
                    "Mutation failed to happen, check probabilities (or you are very unlucky)"
                )

    def _mutate_param(self, param, param_value, mutation_rate, mutationstds, i):
        """Mutate a parameter value.

        Args:
            param (str): The parameter name.
            param_value (int | float): The parameter value.
            mutation_rate (float | list): The mutation rate.
            mutationstds (float | list): The mutation standard deviation

        Raises:
            ValueError: If the parameter type is not supported.
        """
        if param not in self.mutation_config.parameters:
            return

        mutation_rate = (
            random.choice(self.mutation_config.means)
            if isinstance(mutation_rate, list)
            else mutation_rate
        )
        mutationstds = (
            random.choice(self.mutation_config.vars)
            if isinstance(mutationstds, list)
            else mutationstds
        )
        value = None
        if isinstance(param_value, int):
            value = param_value + int(np.random.normal(mutation_rate, mutationstds))
        elif isinstance(param_value, float):
            value = param_value * (1 + np.random.normal(mutation_rate, mutationstds))
        else:
            raise ValueError(f"Unsupported type for mutation: {type(param_value)}")
        goal = self
        if not param in self.to_dict().keys():
            goal = self.trainer_config
        value = self.mutation_config.clip(value, i)
        setattr(goal, param, value)


class SACAgentConfig(AgentConfig):
    def __init__(self):
        super().__init__()
        self.type = "sac"
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
        self.log_std_min = -20
        self.log_std_max = 2
        self.alpha_lr = 0.0003
        self.alpha_tuning = True


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


class TD3AgentConfig(DDPGAgentConfig):
    def __init__(self):
        super().__init__()
        self.type = "td3"
        self.eps = 1.0
        self.noise_sigma = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 3
        self.tao = 0.005


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

    TYPE2AGENT = {"ddpg": DDPGAgentConfig, "td3": TD3AgentConfig, "sac": SACAgentConfig}

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

        # impl inheritance for specialized_config from base_config
        agent_configs = self.get_agents()
        for agent_config in agent_configs:
            for attr in dir(agent_config.specialized_config):
                if not attr.startswith("_") and not callable(
                    getattr(agent_config.specialized_config, attr)
                ):
                    if getattr(agent_config.specialized_config, attr) is None:
                        setattr(
                            agent_config.specialized_config,
                            attr,
                            getattr(self.base_config, attr),
                        )

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

    def to_yaml(self, yaml_path: str):
        """
        Save the configuration to a YAML file.

        Args:
            yaml_path (str): The path to the YAML file.
        """
        with open(yaml_path, "w") as file:
            yaml.dump(self.to_dict(), file)


global_config: Config = Config()
