from math import gamma
from os import stat
import numpy as np
import warnings
from collections import namedtuple, defaultdict
from typing import Optional, Tuple, List, Callable, Any

import torch
from torch import Tensor
from torch import optim
from torch import distributions
from torch import nn

'''class Reinforce(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.4,
        gamma: float = 0.2,
        alpha: float = 0.4
    ) -> None:
        """
        Reinforce main class
        Args:
            features : Module specifying the input layer to use
            num_actions (int): Number of classes
            dropout (float): The droupout probability
        """
        super().__init__()
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=alpha)

        self.num_actions = self.model._modules[str(len(self.model._modules) - 1)].out_features

        self.epsilon = epsilon
        self.gamma = gamma

    def _forward(self, x: Tensor) -> Tensor:
        state_action_values = self.model(x)
        policy = torch.where(state_action_values == torch.max(state_action_values), 1 - self.epsilon + (self.epsilon / self.num_actions), self.epsilon / self.num_actions)
        return policy

    def forward(self, x: Tensor) -> Tensor:
        x = torch.tensor(x).float()
        policy = self._forward(x)

        if self.training:
            policy = distributions.binomial.Binomial(10, probs=policy).sample()

        outputs = torch.argmax(policy)
        
        return outputs
    
    def push(self, state, new_state, action, reward, done):
        state = torch.tensor([state]).float()
        new_state = torch.tensor([new_state]).float()

        self.memory.append((state, new_state, action, reward, done))

        if done:
            discounted_reward = 0
            self.memory.reverse()

            for experience in self.memory:
                state, new_state, action, reward, done = experience
                discounted_reward = self.gamma * discounted_reward + reward

                self.state_values[observation][action] += (1 / self.alpha) * (discounted_reward - self.state_values[observation][action])

                optimal_action = torch.argmax(self.state_values[observation])

                discounted_reward = reward + self.gamma * new_state_value[torch.argmax(new_state_value)]
                error = (discounted_reward - state_value[action])
                loss = torch.pow(error, 2)

                # print(loss)
                loss.backward()
                self.optimizer.step()
            
            self.memory = []'''

class Reinforce(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        gamma: float = 0.2,
        alpha: float = 0.4,
        batch: int = 200
    ) -> None:
        """
        Reinforce main class
        Args:
            num_actions (int): Number of classes
            gamma (float): Reward propagation factor
        """
        super().__init__()
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=alpha)

        self.gamma = torch.tensor(gamma)
        self.batch = batch

        self.batch_states = list()
        self.batch_actions = list()
        self.batch_qvals = list()

        self.memory = list()
        self.counter = 0

        self._init_all(self.model, torch.nn.init.uniform_, a=-1., b=1) 
    
    def _init_all(self, model, init_func, *params, **kwargs):
        for p in model.parameters():
            init_func(p, *params, **kwargs)
    
    def _forward(self, x: Tensor) -> Tensor:
        #print(x)
        state_action_values = self.model(x)
        policy = torch.softmax(state_action_values, dim=-1)
        return policy

    @torch.no_grad()
    def forward(self, state: Tensor) -> Tensor:
        state = torch.tensor(state)
        policy = self._forward(state)
        #print(policy)
        output = torch.multinomial(policy, 1)
        #print(output)
        return output.item()
    
    def push(self, state, new_state, action, reward, done):
        self.memory.append([state, new_state, action, reward, done])

        if done:
            self.counter += 1
            discounted_reward = 0
            loss = torch.tensor([0.0])

            self.memory.reverse()
            for step in self.memory:
                state, new_state, action, reward, done = step
                discounted_reward = reward + self.gamma * discounted_reward
                
                self.batch_states.append(state)
                self.batch_actions.append(action)
                self.batch_qvals.append(discounted_reward)
            
            self.memory = list()
        
        if self.counter >= self.batch:
            batch_states = torch.tensor(self.batch_states)
            batch_actions = torch.tensor(self.batch_actions)
            batch_qvals = torch.tensor(self.batch_qvals)

            batch_qvals = (batch_qvals - batch_qvals.mean()) / batch_qvals.std()

            batch_state_action_value = self._forward(batch_states)
            batch_selected_state_action_value = batch_state_action_value.gather(-1, batch_actions.unsqueeze(1))
            loss = batch_qvals * torch.log(batch_selected_state_action_value) # torch.pow(discounted_reward - selected_state_action_value, 2)
            loss = - loss.mean()

            #print(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.batch_states = list()
            self.batch_actions = list()
            self.batch_qvals = list()

            self.counter = 0
