import collections
from collections import deque

import warnings 
from typing import Optional, Tuple, List, Callable, Any

import torch
from torch import Tensor

class ExperienceBuffer():
    def __init__(
        self,
        size: int = 1024,
        gamma: float = 0.99
    ) -> None:
        """
        ExperienceBuffer main class
        Args:
            episodic (bool): The experience are episodic
            size (int): The number of max experiences to store
        """
        super().__init__()

        self.size = size
        self.gamma = gamma

        self.state_buffer = deque(maxlen=size)
        self.new_state_buffer = deque(maxlen=size)
        self.action_probs_buffer = deque(maxlen=size)
        self.value_buffer = deque(maxlen=size)
        self.reward_buffer = deque(maxlen=size)
        self.done_buffer = deque(maxlen=size)
    
    def __len__(self):
        return len(self.reward_buffer)
    
    def _bellman_backward(self):
        for step_n in range(len(self) - 2, -1, -1):
            if self.done_buffer[step_n] == True:
                break

            self.reward_buffer[step_n] = self.gamma * self.reward_buffer[step_n + 1] + self.reward_buffer[step_n]

    def push(self, state: Tensor, new_state: Tensor, action_probs: Tensor, value: float, reward: float, done: bool) -> Tensor:
        self.state_buffer.append(state)
        self.new_state_buffer.append(new_state)
        self.action_probs_buffer.append(action_probs)
        self.value_buffer.append(value)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)

        if done == True:
            self._bellman_backward()

    def batch(self) -> Tensor:
        action_probs_tensor = [action_probs.unsqueeze(dim=0) for action_probs in self.action_probs_buffer]
        action_probs_tensor = torch.cat(action_probs_tensor)

        value_tensor = [value for value in self.value_buffer]
        value_tensor = torch.tensor(value_tensor).unsqueeze(dim=1)

        reward_tensor = [reward for reward in self.reward_buffer]
        reward_tensor = torch.tensor(reward_tensor).unsqueeze(dim=1)

        self.action_probs_buffer.clear()
        self.value_buffer.clear()
        self.reward_buffer.clear()
        self.done_buffer.clear()

        return (
            action_probs_tensor,
            value_tensor,
            reward_tensor
        )
