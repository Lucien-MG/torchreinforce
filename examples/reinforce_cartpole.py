import gym

import torch
from torch import nn

import torchreinforce
from torchreinforce.agents.reinforce import Reinforce
from torchreinforce.environments.train import train
from torchreinforce.environments.render import render

torch.manual_seed(1)

model = nn.Sequential(
          nn.Linear(4,128),
          nn.ReLU(),
          nn.Linear(128,2),
        )

def cartpole_simple_reinforce():
    max_reward = (0, 0)
    max_agent = None
    for i in range(0, 10):
        torch.manual_seed(i)
        env = gym.make('CartPole-v1')
        agent = Reinforce(model=model, gamma=0.99, alpha=0.03)
    
        best_agent, best_reward = train(env, agent, num_episodes=30000)
        if best_reward > max_reward[0]:
          print("Save")
          max_reward = (best_reward, i)
          max_agent = best_agent
    print(max_reward)
    render(env, max_agent, sleep_time=0.2)

cartpole_simple_reinforce()
