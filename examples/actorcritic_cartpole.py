import gym
import collections

import torch
from torch import nn, sigmoid

import torchreinforce
from torchreinforce.agents.actor_critic import ActorCritic
from torchreinforce.buffers.experience_buffer import ExperienceBuffer
from torchreinforce.ops.loss import actor_critic_loss
from torchreinforce.environments.train import train
from torchreinforce.environments.render import render

NUM_EPISODE = 15000
GAMMA = 0.99
ALPHA = 0.01

def _init_all(self, model, init_func, *params, **kwargs):
    for p in model.parameters():
         init_func(p, *params, **kwargs)

model = nn.Sequential(
    nn.Linear(4,64),
    nn.ReLU(),
)

actor = nn.Sequential(
    nn.Linear(64, 2),
    nn.Softmax(),
)

critic = nn.Sequential(
    nn.Linear(64, 1),
    nn.Sigmoid()
)

def actorcritic_cartpole():
    env = gym.make('CartPole-v1')

    agent = ActorCritic(model=model, actor=actor, critic=critic)
    buffer = ExperienceBuffer(gamma=GAMMA)

    optimizer = torch.optim.SGD(agent.parameters(), lr=ALPHA)

    # Train agent:
    agent.train()

    # Variables:
    done = False
    episode_reward = 0

    mean_reward = collections.deque(maxlen=1000)
    best_mean_reward = float("-inf")

    # Train loop:
    observation, info = env.reset(return_info=True)

    for i in range(1, NUM_EPISODE):
        optimizer.zero_grad()

        while not done:
            observation = torch.tensor(observation)
            action_probs, state_value = agent.forward(observation)

            action = torch.multinomial(action_probs, 1)

            new_observation, reward, done, info = env.step(action.item())
            if done:
                reward = -10
            buffer.push(observation, new_observation, action_probs[action], state_value, reward, done)

            episode_reward += reward

            if done:
                observation, info = env.reset(return_info=True)
                done = False

                mean_reward.append(episode_reward)
                episode_reward = 0

                if (i % 100) == 0:
                    current_mean_reward = sum(mean_reward) / len(mean_reward)

                    if best_mean_reward <= current_mean_reward:
                        best_mean_reward = current_mean_reward

                    print("reward", current_mean_reward)
                break

            observation = new_observation
        
        #if i % 70 == 0:
        batch = buffer.batch()
        loss = actor_critic_loss(batch[0], batch[1], batch[2])

        # loss.backward(retain_graph=True)
        #print(loss)
        loss.backward()
        optimizer.step()


    env.close()

actorcritic_cartpole()
