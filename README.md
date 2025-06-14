# Reinforcement Learning in Games

<!-- TOC -->
* [Reinforcement Learning in Games](#reinforcement-learning-in-games)
  * [Introduction](#introduction)
  * [Discrete Environments](#discrete-environments)
  * [Continuous Environments](#continuous-environments)
  * [Multi-Agent Environments](#multi-agent-environments)
  * [Creating a Custom Environment](#creating-a-custom-environment)
  * [Further Resources](#further-resources)
<!-- TOC -->

## Introduction

Reinforcement learning, although far less known and less explored than
supervised or unsupervised learning, might be the closest to how humans
actually learn - by interacting with the environment. Whether we are
learning to drive a car or to hold a conversation, we are always aware of
how our environment responds to what we do, and we seek to influence what
happens through our behaviour.

Reinforcement learning can be understood as learning through trial and
error by taking an action an observing how it affects the environment - the
change is then evaluated with a numerical reward or punishment. What sets 
reinforcement learning apart from other machine learning approaches is its 
inherently closed-loop nature, the absence of explicit instructions on which 
actions to take, and the fact that the effects of actions - including 
rewards - often unfold over longer time periods.

The main characters of RL are the **agent** and the **environment**. The 
image below shows the agent-environment interaction loop - at every step the 
agent makes an action based on an observation and the environment responds 
with its new state and a **reward** signal. The goal of the agent is to 
maximize its cumulative reward, called **return**.

<p align="center">
  <img src="/images/RL_loop.png" alt="RL learning loop"/>
</p>

The goal of this project is to introduce reinforcement learning concept and 
help implement game environments using libraries such as 
[Gymnasium](https://gymnasium.farama.org/), 
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/), 
and [PettingZoo](https://pettingzoo.farama.org/).

We'll start by discussing simple [**discrete environments**](#discrete-environments) using 
**Q-learning**, then move on to more complex [**continuous environments**](#continuous-environments) 
that require advanced algorithms like **PPO** or **DDPG**. Next, we'll 
explore [**multi-agent reinforcement learning**](#multi-agent-environments) (MARL), where multiple agents 
learn and interact within the same environment. Finally, we'll demonstrate 
how to [**build a custom environment**](#creating-a-custom-environment).

We hope this project will serve as a clear and practical introduction to the 
core concepts of reinforcement learning.

## Discrete Environments

Discrete environments are a great starting point for learning reinforcement 
learning. In these environments, both the set of possible states and the set 
of actions are finite and countable.

In this section, we'll explore how an agent can learn to make decisions in 
such settings using **Q-learning**, a foundational algorithm in RL.

[Open Notebook](01_discrete_environments.ipynb)

## Continuous Environments

In this section, we explore reinforcement learning in environments with 
**continuous state and/or action spaces**, where traditional table-based 
approaches like Q-learning are no longer practical. In such settings, the 
number of possible states or actions is infinite, making it impossible to 
store values for every state-action pair in a lookup table.

To overcome this, we turn to **function approximation techniques**, 
particularly neural networks, which can estimate value functions or policies 
across continuous domains.

We introduce the
[`stable-baselines3`](https://stable-baselines3.readthedocs.io/) library, 
which provides implementations of RL algorithms.

[Open Notebook](02_continous_environments.ipynb)

## Multi-Agent Environments

In this section, we explore **multi-agent reinforcement learning (MARL)** 
using the [PettingZoo](https://pettingzoo.farama.org/) library and train 
agents with **Maskable PPO** from [Stable-Baselines3]
(https://stable-baselines3.readthedocs.io/). We'll demonstrate how agents 
can interact, compete, or cooperate within shared environments designed for 
multiple participants.

[Open Notebook](03_multiagent_environments.ipynb)

In this bonus notebook, we explore a simple self-play setup where a **single 
agent is trained against a previous version of itself**. This approach can 
be useful for iterative training and evaluating learning progress over time. 
We'll use PettingZoo environments and Maskable PPO to demonstrate this technique.

[Open Notebook](03_bonus_single_agent_against_previous_agent.ipynb)

## Creating a Custom Environment

In this section, we walk through the process of **creating a custom 
reinforcement learning environment** compatible with the Gymnasium API. This 
allows for full control over the environment's dynamics, observation/action 
spaces, and reward structure, enabling experimentation and research.

[Open Notebook](04_custom_environment.ipynb)

## Further Resources

