import warnings
from collections import namedtuple, defaultdict
from functools import partial
from typing import Optional, Tuple, List, Callable, Any

import torch
from torch import Tensor
from torch import nn

class MonteCarlo(nn.Module):
    def __init__(
        self,
        features: nn.Module,
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

class MonteCarloFirstVisitControl(MonteCarlo):
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



