import warnings
from collections import namedtuple
from functools import partial
from typing import Optional, Tuple, List, Callable, Any

import torch
from torch import Tensor
from torch import nn

class RandomChoice(nn.Module):
    def __init__(
        self,
        num_actions: int = 4,
    ) -> None:
        """
        RandomChoice main class
        Args:
            num_actions (int): Number of classes
        """
        super().__init__()
        self.num_actions = num_actions
    
    def _forward(self, x: Tensor) -> Tensor:
        return torch.rand(self.num_actions)

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return outputs
