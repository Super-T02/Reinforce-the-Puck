import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "..", "reinforce-the-puck")
)

from utils.checkpoint import Checkpoint, CheckpointManager


# Stubs for missing classes
class BaseAgent:
    def __init__(self, name, agent_type):
        self.name = name
        self.agent_type = agent_type

    def save(self, path):
        pass  # Stub method

    def get_name(self):
        return self.name

    def get_config(self):
        return AgentConfig(self.agent_type)


class AgentConfig:
    def __init__(self, agent_type):
        self.type = agent_type

    def to_yaml(self, path):
        pass  # Stub method


@pytest.fixture
def setup_checkpoint_manager():
    return CheckpointManager(environment_name="test_env")


@pytest.fixture
def setup_agent():
    return BaseAgent(name="test_agent", agent_type="test_type")


@pytest.fixture
def setup_checkpoint(setup_agent):
    return Checkpoint(agent=setup_agent, environment="test_env", avg_eval_reward=10.0)


def test_add_checkpoint(setup_checkpoint_manager, setup_checkpoint):
    setup_checkpoint_manager.add_checkpoint(setup_checkpoint)
    assert len(setup_checkpoint_manager._checkpoints) == 1
    assert setup_checkpoint_manager._checkpoints[0] == setup_checkpoint


def test_add_multiple_checkpoints(setup_checkpoint_manager, setup_checkpoint):
    for i in range(6):
        checkpoint = Checkpoint(
            agent=setup_checkpoint.agent, environment="test_env", avg_eval_reward=i
        )
        setup_checkpoint_manager.add_checkpoint(checkpoint)

    assert (
        len(setup_checkpoint_manager._checkpoints) == 5
    )  # Should only keep the last 5


def test_best_checkpoint_selection(setup_checkpoint_manager, setup_agent):
    checkpoint1 = Checkpoint(
        agent=setup_agent, environment="test_env", avg_eval_reward=5.0
    )
    checkpoint2 = Checkpoint(
        agent=setup_agent, environment="test_env", avg_eval_reward=10.0
    )

    setup_checkpoint_manager.add_checkpoint(checkpoint1)
    setup_checkpoint_manager.add_checkpoint(checkpoint2)

    assert (
        setup_checkpoint_manager._best_checkpoint[setup_agent.get_config().type]
        == checkpoint2
    )


def test_save_last_checkpoint(setup_checkpoint_manager, setup_checkpoint, monkeypatch):
    monkeypatch.setattr(setup_checkpoint, "save", MagicMock())
    setup_checkpoint_manager.add_checkpoint(setup_checkpoint)
    setup_checkpoint_manager.save_last_checkpoint()

    setup_checkpoint.save.assert_called_once_with(setup_checkpoint.get_path() + "_last")


def test_get_best_agent(setup_checkpoint_manager, setup_agent):
    checkpoint = Checkpoint(
        agent=setup_agent, environment="test_env", avg_eval_reward=10.0
    )
    setup_checkpoint_manager.add_checkpoint(checkpoint)

    best_agent = setup_checkpoint_manager.get_best_agent(setup_agent.get_config().type)
    assert best_agent == setup_agent


def test_get_best_agent_no_agent(setup_checkpoint_manager):
    best_agent = setup_checkpoint_manager.get_best_agent("non_existent_type")
    assert best_agent is None
