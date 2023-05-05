import gym
import numpy as np
from actor_critic.ppo_torch import Agent
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # lunar lander, cart pole
    env = gym.make('Pendulum-v1')
    N = 20
    batch_size = 128
    n_epochs = 4
    alpha = 0.001
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)
    n_games = 500

    best_score = env.reward_range[0]
    score_history = []
    moving_avgs = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
            env.render()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        moving_avg = np.mean(score_history[-5:])
        moving_avgs.append(moving_avg)
        plt.plot(np.arange(len(moving_avgs)), moving_avgs)
        plt.savefig('./plots/ppo_Pendulum.pdf')

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)


