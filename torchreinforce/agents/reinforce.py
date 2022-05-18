import warnings
from collections import namedtuple
from functools import partial
from typing import Optional, Tuple, List, Callable, Any

import torch
from torch import Tensor
from torch import nn

class Reinforce(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_actions: int = 4,
        dropout: float = 0.2,
    ) -> None:
        """
        Reinforce main class
        Args:
            features : Module specifying the input layer to use
            num_actions (int): Number of classes
            dropout (float): The droupout probability
        """
        super().__init__()
        self.features = features
        self.conv1 = nn.Conv1d(1, 8, 3)
        self.conv2 = nn.Conv1d(1, 4, 3)
        self.fc = nn.linear(10, num_actions)
    
    def _forward(self, x: Tensor) -> List[Tensor]:
        x = self.features(x)
        x = self.conv1(x)
        x = self.conv1(x)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return outputs
