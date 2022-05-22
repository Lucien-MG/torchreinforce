import gym
import collections

import torch
import torchreinforce
from torchreinforce.agents import Random
from torchreinforce.agents.random import RandomChoice
from torchreinforce.agents.monte_carlo import MonteCarloFirstVisitControl

SEED = 42

def test_blackjack():
    env = gym.make("Blackjack-v1")
    done = False

    agent = RandomChoice(2)
    observation, info = env.reset(seed=42, return_info=True)

    while not done:
        action = torch.argmax(agent.forward(observation)).item()
        observation, reward, done, info = env.step(action)

        if done:
            observation, info = env.reset(return_info=True)

    env.close()

    assert True

def test_blackjack_montecarlo():
    env = gym.make("Blackjack-v1")
    done = False

    mean_reward = collections.deque(maxlen=1000)

    agent = MonteCarloFirstVisitControl(num_actions=2, epsilon=0.5, gamma=0.3, alpha=0.1)
    observation, info = env.reset(seed=SEED, return_info=True)

    for i in range(5):
        while not done:
            action = agent.forward(observation).item()
            observation, reward, done, info = env.step(action)

            agent.push(observation, action, reward, done)

            if done:
                observation, info = env.reset(return_info=True)
                env.reset()
                done = False
                mean_reward.append(reward)
                if i % 10000 == 0:
                    print(sum(mean_reward) / len(mean_reward))
                break

    env.close()

    assert False
