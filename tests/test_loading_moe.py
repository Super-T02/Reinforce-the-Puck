import os
import sys
import tempfile

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "reinforce-the-puck"))

from agents.agent_factory import AgentFactory
from agents.base_agent import BaseAgent
from agents.moe_agent import MOEAgent
from gymnasium import spaces
from utils.config import MoeAgentConfig

try:
    config = MoeAgentConfig()
    config.agent_a_path = (
        r"..\final_checkpoints\01-30-sac-foundation\checkpoint_best.pth"
    )
    config.agent_b_path = (
        r"..\final_checkpoints\01-30-sac-foundation\checkpoint_best.pth"
    )
    config.agent_a_type = "sac"
    config.agent_b_type = "sac"

    observation_space = spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32)

    num_actions = 2
    action_space = spaces.Box(-1, +1, (num_actions * 2,), dtype=np.float32)

    agent = AgentFactory.create_agent_from_config(
        config, observation_space, action_space
    )

    agent.save(os.path.join(tempfile.gettempdir(), "checkpoint"))

    agent.agent_a = None
    agent.agent_b = None

    agent.load(os.path.join(tempfile.gettempdir(), "checkpoint"))

    agent.act(np.zeros(18))
except Exception as e:
    print(e)
