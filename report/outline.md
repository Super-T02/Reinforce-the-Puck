# Reinforcement Learning Final Project Report

## 1. Introduction
- **Problem Statement:** Development of a Reinforcement Learning (RL) agent for a hockey game.
- **Motivation:** Analysis and comparison of different RL algorithms to optimize game performance.
- **Overview:** Description of the game and the RL methods implemented to solve the problem.

## 2. Methods
### 2.1 Algorithmic Foundations
- **Soft Actor Critic (SAC)**
  - Implementation details
  - Adaptations for the hockey environment
- **Twin Delayed Deep Deterministic Policy Gradient (TD3)**
  - Introduction to TD3
  - Implementation details
  - Adaptations for the hockey environment
- **Hybrid Approach (Combination of SAC and TD3 with DQN)**
  - Motivation for the combination
  - Architecture of the hybrid model
  - Implementation details

### 2.2 Training Framework (Jona)
- **Custom RL Training Framework**
  - Overview of features
  - Architecture of the framework
  - Implementation of self-play
  - Leaderboard (Ranking)

### 2.3 Enhancements through Prioritized Experience Replay
- **Description of the Prioritized Buffer method**
- **Expected benefits and impact on convergence**

### 2.4 Reward Function (Tom)

## 3. Experimental Evaluation
### 3.1 Training on Simple Environments
- **Results on Pendulum-v0, LunarLander-v2, or HalfCheetah**
- **AVG Reward**

### 3.2 Training in the Hockey Environment
- **Training against the basic strong opponent**
  - Sac/TD3/BER/BPER - Foundation Model (eval vs training)
- **Fine-tuning using self-play with rotating checkpoints, as well as both weak and strong opponents**
  - Leaderboard Scores


## 4. Discussion (Jona & Tom)
- **Analysis of training results**
- **Comparison of SAC, TD3, and the hybrid solution**
- **Impact of the framework and Prioritized Replay Buffer**
- **Limitations and potential improvements**

## 5. Conclusion (Jona & Tom)
- **Summary of key findings**
- **Future development possibilities**

## 6. References
- **Citations of relevant papers, including Rainbow, TD-MPC, Soft Actor-Critic, etc.**
