import gym
from non_actor_critic.dqn import Agent
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=0, batch_size=64, n_actions=env.action_space.n, eps_end=0,
                  input_dims=env.observation_space.shape, lr=0, max_mem_size=50000)
    agent.load_models()
    scores = []
    n_games = 10

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            observation = observation_
            env.render()
        scores.append(score)




    data = {'episode': np.arange(len(scores)), 'avg_reward': scores}
    df = pd.DataFrame(data)
    df.to_csv('./testing_log/dqn/lunar_lander.csv', index=False)
    