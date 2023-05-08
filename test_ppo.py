import gym
import numpy as np
from actor_critic.ppo_torch import Agent
from matplotlib import pyplot as plt
import pandas as pd

if __name__ == '__main__':
    # lunar lander, cart pole
    env = gym.make('LunarLander-v2')
    N = 20
    batch_size = 256
    n_epochs = 4
    alpha = 0
    # tweak to discover test configuration
    agent = Agent(n_actions=env.action_space.n, 
                    input_dims=env.observation_space.shape,
                    gamma=0.99, alpha=0, gae_lambda=1,
                    policy_clip=1, batch_size=256, n_epochs=10)
    
    agent.load_models()

    n_games = 10

    scores = []
    

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            observation = observation_
            env.render()
        scores.append(score)


    data = {'episode': np.arange(len(scores)), 'avg_reward': scores}
    df = pd.DataFrame(data)
    df.to_csv('./testing_log/ppo/lunar_lander.csv', index=False)

