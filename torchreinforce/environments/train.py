import collections

def train(env, agent, num_episodes=10):
    done = False
    mean_reward = collections.deque(maxlen=10000)

    observation, info = env.reset(return_info=True)

    for i in range(num_episodes):
        while not done:
            action = agent.forward(observation).item()
            new_observation, reward, done, info = env.step(action)
            agent.push(observation, new_observation, action, reward, done)

            if done:
                observation, info = env.reset(return_info=True)
                env.reset()
                done = False
                mean_reward.append(reward)
                if (i + 1) % (num_episodes // 10) == 0 and i != 0:
                    agent.epsilon = max(0.001, agent.epsilon - 0.05)
                    print("Score", sum(mean_reward) / len(mean_reward), "epsilon:", agent.epsilon)
                break

            observation = new_observation

    env.close()
       