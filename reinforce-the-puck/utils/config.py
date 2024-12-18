import os

import torch
import yaml

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
        """Update the configuration group from a dictionary.

        Args:
            config_dict (dict): Dictionary with the new configuration values.
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
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
        self.name = "BasicOpponent"
        self.version = 1
        self.memory_size = 10000


####################################################################################################
# Configuration Class
####################################################################################################


class Config:
    """
    Main configuration class to load and manage YAML configurations.
    """

    def __init__(self):
        self.base_config = BaseConfig()
        self.agent1 = AgentConfig()
        self.agent2 = AgentConfig()

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


global_config = Config()
