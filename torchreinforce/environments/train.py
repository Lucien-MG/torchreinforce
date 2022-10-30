import collections
#from torch.utils.tensorboard import SummaryWriter

def train(env, agent, num_episodes=10):
    #writer = SummaryWriter('runs/')
    agent.train()

    done = False
    episode_reward = 0

    mean_reward = collections.deque(maxlen=1000)
    best_mean_reward = float("-inf")

    observation, info = env.reset(return_info=True)

    for i in range(num_episodes):
        while not done:
            action = agent.forward(observation)
            new_observation, reward, done, info = env.step(action)
            agent.push(observation, new_observation, action, reward, done)

            episode_reward += reward

            if done:
                observation, info = env.reset(return_info=True)
                env.reset()
                done = False
                mean_reward.append(episode_reward)
                episode_reward = 0
                if (i % 100) == 0:
                    current_mean_reward = sum(mean_reward) / len(mean_reward)
                    if best_mean_reward <= current_mean_reward:
                        best_mean_reward = current_mean_reward
                    print("reward", current_mean_reward)
                    #writer.log("reward", sum(mean_reward) / len(mean_reward))
                break

            observation = new_observation

    env.close()

    return best_mean_reward
       