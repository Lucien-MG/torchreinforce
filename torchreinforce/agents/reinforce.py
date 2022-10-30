from collections import OrderedDict
from typing import Any, Callable, Optional

from torch import nn

class reinforce(nn.Module):

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy
    
    def forward(self, state):
        return self.policy(state)
