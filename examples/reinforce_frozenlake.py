import gym

import torch
from torch import nn
import torchreinforce
from torchreinforce.agents.reinforce import Reinforce
from torchreinforce.environments.train import train
from torchreinforce.environments.render import render
from torchreinforce.environments.miscs.frozenlake.maps import frozenlake_maps_4x4, frozenlake_maps_8x8

model = nn.Sequential(
          nn.Linear(16,32),
          nn.Sigmoid(),
          nn.Linear(32,16),
          nn.Sigmoid(),
          nn.Linear(16,4),
        )

def frozenlake_simple_reinforce():
    env = gym.make('FrozenLake-v1', desc=frozenlake_maps_4x4["4x4_simple"], is_slippery=False)
    agent = Reinforce(model=model, epsilon=0.99, gamma=0.8, alpha=0.4)
    
    train(env, agent, num_episodes=300)
    render(env, agent, sleep_time=1)

frozenlake_simple_reinforce()
