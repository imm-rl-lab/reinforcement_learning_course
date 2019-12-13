import numpy as np
import matplotlib.pyplot as plt

def get_epsilon_greedy_action(Q_state, epsilon, action_n):
    prob = np.ones(action_n) * epsilon / action_n
    argmax_action = np.argmax(Q_state)
    prob[argmax_action] += 1 - epsilon
    action = np.random.choice(np.arange(action_n), p=prob)
    return action

def Monte_Carlo(env, episode_n=500, t_max=500, gamma=0.99):
    state_n = env.observation_space.n
    action_n = env.action_space.n
    Q = np.zeros((state_n, action_n))
    N = np.zeros((state_n, action_n))
    total_rewards = np.zeros(episode_n)

    for episode in range(episode_n):
        epsilon = 1 - episode / episode_n
        states = np.zeros(t_max, dtype=int)
        actions = np.zeros(t_max, dtype=int)
        rewards = np.zeros(t_max)
        s = env.reset()
        T = 0
        for t in range(t_max):
            states[t] = s
            a = get_epsilon_greedy_action(Q[s], epsilon, action_n)
            actions[t] = a
            next_s, r, done, _ = env.step(a)
            rewards[t] = r
            s = next_s
            if done:
                break
        total_rewards[episode] = sum(rewards)
        returns = np.zeros(t_max+1)
        for t in range(t_max - 1, -1, -1):
            returns[t] = rewards[t] + gamma * returns[t + 1]
            Q[states[t]][actions[t]] += (returns[t] - Q[states[t]][actions[t]]) / (1 + N[states[t]][actions[t]])
            N[states[t]][actions[t]] += 1

    return total_rewards

def SARSA(env, episode_n=100, t_max=500, alpha=0.5, gamma=0.99):
    state_n = env.observation_space.n
    action_n = env.action_space.n
    Q = np.zeros((state_n, action_n))
    total_rewards = np.zeros(episode_n)
    for episode in range(episode_n):
        epsilon = 1 / (1 + episode)
        s = env.reset()
        a = get_epsilon_greedy_action(Q[s], epsilon, action_n)
        for t in range(t_max):
            next_s, r, done, _ = env.step(a)
            total_rewards[episode] += r
            next_a = get_epsilon_greedy_action(Q[next_s], epsilon, action_n)
            Q[s][a] = Q[s][a] + alpha * (r + (1 - done) * gamma * Q[next_s][next_a] - Q[s][a])
            s, a = next_s, next_a
            if done:
                break
    return total_rewards

def Q_Learning(env, episode_n=100, t_max=500, alpha=0.5, gamma=0.99):
    state_n = env.observation_space.n
    action_n = env.action_space.n
    Q = np.zeros((state_n, action_n))
    total_rewards = np.zeros(episode_n)
    for episode in range(episode_n):
        epsilon = 1 / (1 + episode)
        s = env.reset()
        for t in range(t_max):
            a = get_epsilon_greedy_action(Q[s], epsilon, action_n)
            next_s, r, done, _ = env.step(a)
            total_rewards[episode] += r
            Q[s][a] = Q[s][a] + alpha * (r + (1 - done) * gamma * max(Q[next_s]) - Q[s][a])
            s = next_s
            if done:
                break
    return total_rewards


import gym
env = gym.make("Taxi-v2")

MC_rewards = Monte_Carlo(env, episode_n=500)
SARSA_rewards = SARSA(env, episode_n=500)
Q_Learning_rewards = Q_Learning(env, episode_n=500)

plt.plot(MC_rewards)
plt.plot(SARSA_rewards)
plt.plot(Q_Learning_rewards)
plt.show()