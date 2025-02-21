# Reinforce the Puck [![Build LaTeX document](https://github.com/Super-T02/Reinforce-the-Puck/actions/workflows/build-pdf.yml/badge.svg?branch=main)](https://github.com/Super-T02/Reinforce-the-Puck/actions/workflows/build-pdf.yml) [![Python application](https://github.com/Super-T02/Reinforce-the-Puck/actions/workflows/python-app.yml/badge.svg)](https://github.com/Super-T02/Reinforce-the-Puck/actions/workflows/python-app.yml)

This repository implements agents for the [Hockey-Environment](https://github.com/martius-lab/hockey-env) of the Reinforcement Learning Lecture at the University of Tübingen in the Winter Term 24/25.

## Installation

The following section describes how to install all dependencies for the project.

### Prerequisites

- Python 3.12
- Poetry

To install Poetry, follow the instructions [here](https://python-poetry.org/docs/#installation).

### Install Dependencies

Ensure you have Python 3.12 and Poetry installed. Then, run the following commands to install the dependencies:

```bash
poetry install
```

Additionally, install the requirements from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## File Structure



## Training Configuration

The configuration for training and running the agents is written in a YAML file. The configuration file includes settings for agents, environments, and training parameters.

Example configuration:

```yaml
base_config:
  dtype: float32
  num_episodes: 1000

env1:
  env_name: Pendulum-v1
  id: 0
  max_steps: 1000 # Maximal number of steps

env2:
  env_name: Hockey-v0
  env_type: gym
  id: 2
  mode: 0 # 0: normal, 1: shooting, 2: defense
  max_steps: 1000
  start_training_after_steps: 200000 # 20k for each agent
  train_both: False # Train --> If False only one agent is trained per epoch
  train_all: False # Train all agents (also opponents)
  new_agents_after_eval: 2 # After 100 episodes
  do_render: False
  weights:
    winner_weight: 10.0 # Winner reward
    closeness_puck_weight: 0.1 # Closer = better
    touch_puck_weight: 0 # Touching puck = better
    puck_direction_weight: 0.1 # Puck direction = better
    no_touch_penalty: -1.0 # Penalty for not touching puck after 10% of episode
    timed_penalty_active: False # More time ==> less reward
    block_puck_weight: 0 # Blocking puck = better
    stay_in_goal_weight: 0.02 # Staying in goal = better if opponent is shooting

agent1:
  type: td3
  checkpoint: null
  opponent_names:
    ["opponent1", "opponent2"]
  env_id: 2
  eps: 1
  eval_freq: 50
  eval_episodes: 20
  discount: 0.98
  actor_hidden_sizes: [512, 256]
  critic_hidden_sizes: [512, 256, 64]
  policy_delay: 2
  update_target_every: -1
  noise_sigma: 0.1
  noise_clip: 0.5
  noise_beta: 1.0 # Pink noise
  memory_size: 1000000
  buffer_type: BPER # Buffer type: PER, BER, ER, BPER
  buffer_decay_steps: 1000000
  buffer_alpha: 0.6
  buffer_beta: 0.4
  num_runs: 1
  mutation_config:
    enabled: false
  trainer_config:
    learning_rate_actor: 0.00003
    learning_rate_critic: 0.00003
    log_name: buffer_eval_foundation/td3-ft-dist-goal-ball-foundation-bper
    batch_size: 256
  specialized_config:
    num_episodes: 60000
    do_render: False

agent2:
  env_id: 2
  type: sac
  opponents: ["opponent1", "opponent2"]
  tau: 0.005 # Target network update rate (Soft update)
  memory_size: 1000000
  discount: 0.98 # Discount factor
  alpha: 0.1 # Entropy regularization coefficient
  alpha_lr: 0.0003 # Learning rate for alpha
  log_std_min: -20 #lower bound for log_std
  log_std_max: 2 #upper bound for log_std
  actor_hidden_sizes: [512, 256]
  critic_hidden_sizes: [512, 256, 64]
  alpha_tuning: True
  trainer_config:
    learning_rate_actor: 0.0003
    learning_rate_critic: 0.0003
    log_name: pendulum_eval/sac
    batch_size: 512
  specialized_config:
    num_episodes: 1000
    do_render: False
    start_training_after_steps: 200000 # 200k

opponent1:
  type: "basic_opponent_strong"

opponent2:
  checkpoint: "../final_checkpoints/07-02_moe_foundation/checkpoint_best.pth"
  type: moe
  agent_a_path: "final_checkpoints/08-02-sac-ft/checkpoint_best.pth"
  agent_a_type: "sac"
  agent_b_path: "final_checkpoints/08-02-td3-ft-catch_bonus/checkpoint_last.pth"
  agent_b_type: "td3"
  buffer_type: ER
  gamma: 0.99
  hidden_size: [512, 512]
  memory_size: 100000
  mutation_config:
    enabled: false
  trainer_config:
    batch_size: 512
    beta1: 0.9
    beta2: 0.999
    learning_rate: 0.003
    log_freq: 10
    log_name: "DELETE-moe-ft-win-loose-dist"
```

## Run training

The framework provides
