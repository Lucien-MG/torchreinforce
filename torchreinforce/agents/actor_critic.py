import warnings
from typing import Optional, Tuple, List, Callable, Any

import torch
from torch import Tensor
from torch import distributions
from torch import nn


class ActorCritic(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        actor: nn.Module,
        critic: nn.Module,
    ) -> None:
        """
        ActorCritic main class
        Args:
            model (int): Number of classes
            actor (float): Reward propagation factor
            critic (float): Reward propagation factor
        """
        super().__init__()

        self.model = model
        self.actor = actor
        self.critic = critic
    
    def _forward(self, state: Tensor):
        state_encoding = self.model(state)
        state_action_value = self.actor(state_encoding)
        state_value = self.critic(state_encoding)
        return state_action_value, state_value

    def forward(self, state: Tensor):
        state_action_value, state_value = self._forward(state)
        return state_action_value, state_value
