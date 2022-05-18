import gym

import torch
import torchreinforce
from torchreinforce.agents import Random
from torchreinforce.agents.random import RandomChoice

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
