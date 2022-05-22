import gym

import torch
import torchreinforce
from torchreinforce.agents.temporal_difference import Sarsa
from torchreinforce.environments.train import train
from torchreinforce.environments.render import render

def test_frozenlake_sarsa():
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    agent = Sarsa(num_actions=4, epsilon=0.9, gamma=0.9, alpha=0.2)
    
    train(env, agent, num_episodes=1000)
    render(env, agent)

test_frozenlake_sarsa()
