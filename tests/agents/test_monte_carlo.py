import gym
import collections

import torch
import torchreinforce
from torchreinforce.agents.monte_carlo import MonteCarloFirstVisitControl

class TestMonteCarloMCFVC:
    def test_agent_init(self):
        mcfvc = MonteCarloFirstVisitControl(num_actions=3)
        assert isinstance(mcfvc, MonteCarloFirstVisitControl)

    def test_agent_push(self):
        mcfvc = MonteCarloFirstVisitControl(num_actions=3)

        mcfvc.forward("state_1")

        mcfvc.push("state_1", 0, 1, True)

        print(mcfvc.policy)
        assert True
