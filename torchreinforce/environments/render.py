import collections
import time

def render(env, agent, sleep_time=0):
    agent.eval()

    done = False
    observation, info = env.reset(return_info=True)

    env.render()

    while not done:
        action = agent.forward(observation).item()
        new_observation, reward, done, info = env.step(action)

        env.render()
        observation = new_observation

        time.sleep(sleep_time)

    env.close()
