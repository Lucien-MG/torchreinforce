import warnings
from collections import namedtuple, defaultdict
from functools import partial
from typing import Optional, Tuple, List, Callable, Any

import torch
from torch import Tensor
from torch import distributions
from torch import nn

class MonteCarlo(nn.Module):
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

class MonteCarloFirstVisit(MonteCarlo):
    def __init__(
        self,
        features: nn.Module,
        num_actions: int = 4,
        gamma: float = 0.9
    ) -> None:
        """
        MonteCarloFirstVisit main class
        Args:
            features : Module specifying the input layer to use
            num_actions (int): Number of classes
            gamma (float): Reward propagation factor
        """
        super().__init__()
    
    def _forward(self, x: Tensor) -> List[Tensor]:

        return x

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return outputs

class MonteCarloEveryVisitControl(MonteCarlo):
    def __init__(
        self,
        features: nn.Module,
        num_actions: int = 4,
        epsilon: float = 0.1,
        gamma: float = 0.9
    ) -> None:
        """
        MonteCarloFirstVisitControl main class
        Args:
            features : Module specifying the input layer to use
            num_actions (int): Number of classes
            gamma (float): Reward propagation factor
        """
        super().__init__()

        self.gamma = gamma

        self.policy = defaultdict(lambda: -1)
        self.returns = defaultdict(lambda: list())
        self.memory = list()
    
    def _forward(self, x: Tensor) -> Tensor:
        state_action_values = self.policy(x)
        return state_action_values

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return outputs
    
    def push(self, observation, reward, done):
        self.memory.append((observation, reward, done))

        if done:
            discounted_reward = 0
            for experience in done.reverse():
                observation, reward, done = experience
                discounted_reward = self.gamma * discounted_reward + reward
                self.returns[observation].append()

class MonteCarloFirstVisitControl(MonteCarlo):
    def __init__(
        self,
        num_actions: int,
        epsilon: float = 0.4,
        gamma: float = 0.2,
        alpha: float = 0.4
    ) -> None:
        """
        MonteCarloFirstVisitControl main class
        Args:
            num_actions (int): Number of classes
            gamma (float): Reward propagation factor
        """
        super().__init__()

        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        self.policy = defaultdict(lambda: torch.ones(num_actions) / num_actions)
        self.state_values = defaultdict(lambda: torch.zeros(num_actions))

        self.memory = list()
    
    def _forward(self, x: Tensor) -> Tensor:
        state_action_values = self.policy[x]
        return state_action_values

    def forward(self, x: Tensor) -> Tensor:
        state_action_values = self._forward(x)
        action_probs = distributions.binomial.Binomial(10, probs=state_action_values).sample()
        outputs = torch.argmax(action_probs)
        
        return outputs
    
    def push(self, observation, action, reward, done):
        self.memory.append((observation, action, reward, done))

        if done:
            discounted_reward = 0
            self.memory.reverse()

            for experience in self.memory:
                observation, action, reward, done = experience
                discounted_reward = self.gamma * discounted_reward + reward

                self.state_values[observation][action] += (1 / self.alpha) * (discounted_reward - self.state_values[observation][action])

                optimal_action = torch.argmax(self.state_values[observation])

                self.policy[observation][:] = self.epsilon / len(self.policy[observation])
                self.policy[observation][optimal_action] = 1 - self.epsilon + (self.epsilon / len(self.policy[observation]))
            
            self.memory = []



