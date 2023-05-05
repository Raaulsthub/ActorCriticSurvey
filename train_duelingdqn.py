import gym
import numpy as np
from non_actor_critic.dueling_ddqn_torch import Agent
from matplotlib import pyplot as plt


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    num_games = 700
    load_checkpoint = False

    agent = Agent(gamma=0.99, epsilon=1.0, lr=5e-4,
                  input_dims=[8], n_actions=4, mem_size=50000, eps_min=0.01,
                  batch_size=64, eps_dec=1e-3, replace=100)

    if load_checkpoint:
        agent.load_models()

    filename = 'LunarLander-Dueling-DDQN-512-Adam-lr0005-replace100.png'
    scores = []
    eps_history = []
    moving_avgs = []
    n_steps = 0

    for i in range(num_games):
        done = False
        observation = env.reset()
        score = 0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action,
                                    reward, observation_, int(done))
            agent.learn()

            observation = observation_
            env.render()

        scores.append(score)
        avg_score = np.mean(scores[-100:])
        moving_avg = np.mean(scores[-5:])
        moving_avgs.append(moving_avg)
        plt.plot(np.arange(len(moving_avgs)), moving_avgs)
        plt.savefig('./plots/duelingdqn_lunar.pdf')
        print('episode: ', i,'score %.1f ' % score,
             ' average score %.1f' % avg_score,
            'epsilon %.2f' % agent.epsilon)
        if i > 0 and i % 10 == 0:
            agent.save_models()

        eps_history.append(agent.epsilon)
