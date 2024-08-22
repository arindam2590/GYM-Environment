# **Reinforcement Learning Implementation**

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment to maximize cumulative rewards. Unlike supervised learning, RL doesn’t rely on labeled data; instead, the agent learns from the consequences of its actions.

Key components include the agent (learner), environment (external system), state (current situation), action (choices available), and reward (feedback). The agent balances exploring new actions with exploiting known ones to maximize rewards over time. RL is used in areas like robotics, gaming, and autonomous systems to handle complex decision-making tasks.

## **Table of Contents**
- [Overview](#overview)
- [Environments](#environments)
- [Algorithms Implemented](#algorithms-implemented)
- [Usage](#usage)
- [Results](#results)

## **Overview**

This repository focuses on implementing and experimenting with RL algorithms in the ToyText environments provided by Gymnasium. The aim is to provide a clear and concise implementation of popular RL algorithms, making it easier for researchers and enthusiasts to understand the core concepts of RL.

## **Environments**

The following ToyText environments from Gymnasium are included:

- `FrozenLake`
- `Taxi`
- `Blackjack`
- `CliffWalking`

These environments are simple yet effective for understanding the basics of reinforcement learning algorithms.

## **Algorithms Implemented**

The repository includes implementations of the following RL algorithms in *TensorFlow* and *PyTorch*:

- *Q-Learning*: A model-free RL algorithm that learns the value of actions in states.
- *SARSA (State-Action-Reward-State-Action)*: An on-policy RL algorithm that updates its Q-values based on the action taken.
- *Monte Carlo Methods*: A class of algorithms that learn from complete episodes by averaging sample returns.
- *Value Based*: A family of reinforcement learning algorithms that works on *Q* value of the policy.
- *Policy Gradient*: A family of algorithms that optimize the policy directly by gradient ascent.

## Usage

To run the RL algorithms on a specific ToyText environment, execute * *_main.py* by using the following command:

```bash
python *_main.py
```

For example, to run Q-Learning on the FrozenLake environment:

```bash
python frozenlake_ql_main.py
```

Results
The results of the experiments, including plots and performance metrics, will be saved in the figures/ directory. This includes learning curves, rewards per episode, and other relevant statistics.
