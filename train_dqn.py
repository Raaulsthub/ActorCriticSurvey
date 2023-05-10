import gym
from non_actor_critic.dqn import Agent
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=env.action_space.n, eps_end=0.01,
                  input_dims=env.observation_space.shape, lr=0.001, max_mem_size=50000)
    scores, eps_history = [], []
    n_games = 1000
    best_score = env.reward_range[0]
    moving_avgs = []
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, 
                                    observation_, done)
            agent.learn()
            observation = observation_
            env.render()
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-20:])
        moving_avg = np.mean(scores[-5:])
        moving_avgs.append(moving_avg)
        plt.plot(np.arange(len(moving_avgs)), moving_avgs)
        plt.savefig('./plots/dqn_LunarLanderV2.pdf')

        if avg_score >= best_score:
            best_score = avg_score
            agent.save_models()


        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)

    # data = {'episode': np.arange(len(scores)), 'avg_reward': scores}
    # df = pd.DataFrame(data)
    # df.to_csv('./training_log/dqn/lunar_lander.csv', index=False)
