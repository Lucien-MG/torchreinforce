import torch
from torch import Tensor
from torch.nn import functional

def actor_critic_loss(action_probs: Tensor, values: Tensor, returns: Tensor):
    returns = (returns - returns.mean()) / (returns.std() + 1e-7)
    advantage = returns - values

    action_log_probs = action_probs.log()
    actor_loss = - torch.mean(action_log_probs[action_log_probs.argmax(dim=1)] * advantage)

    critic_loss = functional.huber_loss(values, returns, reduction='mean')

    return actor_loss + critic_loss
