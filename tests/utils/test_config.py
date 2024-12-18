import os
import sys

import torch

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "..", "reinforce-the-puck")
)

from utils import config
from utils.config import Config


def test_load_config():
    config = Config()
    config.from_yaml("tests/utils/test_config.yaml")
    assert config.base_config.device == "cuda"
    assert config.base_config.dtype == torch.float16
    assert config.base_config.max_memory_size == 100000
    assert config.agent1.name == "BasicOpponent"
    assert config.agent1.version == 1
    assert config.agent1.memory_size == 10000
    assert config.agent2.name == "BasicNotOpponent"
    assert config.agent2.version == 19
    assert config.agent2.memory_size == 1
    assert "agent3" not in config.__dict__


def test_global_config():
    assert config.global_config is not None
    assert config.global_config.base_config.device == "cuda"
    assert config.global_config.base_config.dtype == torch.float32
    assert config.global_config.base_config.max_memory_size == 100000
    assert config.global_config.agent1.name == "BasicOpponent"
    assert config.global_config.agent1.version == 1
    assert config.global_config.agent1.memory_size == 10000
    assert config.global_config.agent2.name == "BasicOpponent"
    assert config.global_config.agent2.version == 1
    assert config.global_config.agent2.memory_size == 10000
