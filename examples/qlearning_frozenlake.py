import gym

import torch
import torchreinforce

from torchreinforce.agents.temporal_difference import Qlearning
from torchreinforce.environments.train import train
from torchreinforce.environments.render import render
from torchreinforce.environments.miscs.frozenlake.maps import frozenlake_maps_4x4, frozenlake_maps_8x8

def frozenlake_simple_qlearning():
    env = gym.make('FrozenLake-v1', desc=frozenlake_maps_4x4["4x4_two_way_out"], is_slippery=False)
    agent = Qlearning(num_actions=4, epsilon=0.9, gamma=0.8, alpha=0.1)
    
    train(env, agent, num_episodes=10000)
    render(env, agent)

def frozenlake_8x8_qlearning():
    env = gym.make('FrozenLake-v1', desc=frozenlake_maps_8x8["8x8"], is_slippery=False)
    agent = Qlearning(num_actions=4, epsilon=0.99, gamma=0.8, alpha=0.1)
    
    train(env, agent, num_episodes=100000)
    render(env, agent)

def frozenlake_slippery_qlearning():
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    agent = Qlearning(num_actions=4, epsilon=0.99, gamma=0.8, alpha=0.4)
    
    train(env, agent, num_episodes=50000)
    render(env, agent)

frozenlake_8x8_qlearning()

#frozenlake_slippery_qlearning()
