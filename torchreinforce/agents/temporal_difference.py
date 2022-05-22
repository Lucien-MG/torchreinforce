import warnings
from collections import namedtuple, defaultdict
from functools import partial
from typing import Optional, Tuple, List, Callable, Any

import torch
from torch import Tensor
from torch import distributions
from torch import nn

class TemporalDifference(nn.Module):
    def __init__(
        self,
        num_actions: int = 4,
        gamma: float = 0.9,
    ) -> None:
        """
        MonteCarlo main class
        Args:
            features : Module specifying the input layer to use
            num_actions (int): Number of classes
        """
        super().__init__()
    
    def _forward(self, x: Tensor) -> List[Tensor]:
        return x

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return outputs

class Sarsa(nn.Module):
    def __init__(
        self,
        num_actions: int,
        epsilon: float = 0.4,
        gamma: float = 0.2,
        alpha: float = 0.4
    ) -> None:
        """
        Sarsa main class
        Args:
            num_actions (int): Number of classes
            gamma (float): Reward propagation factor
        """
        super().__init__()
        self.num_actions = num_actions

        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        self.state_values = defaultdict(lambda: torch.zeros(num_actions))
    
    def _forward(self, x: Tensor) -> Tensor:
        state_action_values = self.state_values[x]
        policy = torch.where(state_action_values == torch.max(state_action_values), 1 - self.epsilon + (self.epsilon / self.num_actions), self.epsilon / self.num_actions)
        return policy

    def forward(self, x: Tensor) -> Tensor:
        policy = self._forward(x)

        if self.training:
            policy = distributions.binomial.Binomial(10, probs=policy).sample()

        outputs = torch.argmax(policy)
        
        return outputs
    
    def push(self, state, new_state, action, reward, done):
        discounted_reward = reward + self.gamma * self.state_values[new_state][self(new_state)]
        loss = (discounted_reward - self.state_values[state][action])
        self.state_values[state][action] += self.alpha * loss

class Qlearning(nn.Module):
    def __init__(
        self,
        num_actions: int,
        epsilon: float = 0.4,
        gamma: float = 0.2,
        alpha: float = 0.4
    ) -> None:
        """
        Sarsa main class
        Args:
            num_actions (int): Number of classes
            gamma (float): Reward propagation factor
        """
        super().__init__()
        self.num_actions = num_actions

        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        self.state_values = defaultdict(lambda: torch.zeros(num_actions))
    
    def _forward(self, x: Tensor) -> Tensor:
        state_action_values = self.state_values[x]
        policy = torch.where(state_action_values == torch.max(state_action_values), 1 - self.epsilon + (self.epsilon / self.num_actions), self.epsilon / self.num_actions)
        return policy

    def forward(self, x: Tensor) -> Tensor:
        policy = self._forward(x)

        if self.training:
            policy = distributions.binomial.Binomial(10, probs=policy).sample()

        outputs = torch.argmax(policy)
        
        return outputs
    
    def push(self, state, new_state, action, reward, done):
        best_state_value = torch.argmax(self.state_values[new_state])
        discounted_reward = reward + self.gamma * self.state_values[new_state][best_state_value]
        loss = (discounted_reward - self.state_values[state][action])
        self.state_values[state][action] += self.alpha * loss
