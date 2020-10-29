import numpy as np
from Frozen_Lake import FrozenLakeEnv


def init_policy(env):
    policy = {}
    for state in env.get_all_states():
        policy[state] = {}
        for action in env.get_possible_actions(state):
            policy[state][action] = 1 / len(env.get_possible_actions(state))
    return policy


def init_values(env):
    values = {}
    for state in env.get_all_states():
        values[state] = 0
    return values


def init_q_values():
    q_values = {}
    for state in env.get_all_states():
        q_values[state] = {}
        for action in env.get_possible_actions(state):
            q_values[state][action] = 0
    return q_values


def get_q_values(env, gamma, values):
    q_values = init_q_values()
    for state in env.get_all_states():
        for action in env.get_possible_actions(state):
            for next_state in env.get_next_states(state, action):
                q_values[state][action] += env.get_reward(
                    state, action, next_state) + gamma * env.get_transition_prob(
                    state, action, next_state) * values[next_state]
    return q_values


def update_values(env, gamma, values, policy):
    new_values = init_values(env)
    for state in env.get_all_states():
        for action in env.get_possible_actions(state):
            q_values = get_q_values(env, gamma, values)
            new_values[state] += policy[state][action] * q_values[state][action]
    return new_values


def policy_evaluation(env, gamma, policy, M):
    values = init_values(env)
    for _ in range(M):
        values = update_values(env, gamma, values, policy)
    return values


def policy_improvement(env, gamma, values):
    q_values = get_q_values(env, gamma, values)
    policy = init_policy(env)
    for state in env.get_all_states():
        if len(env.get_possible_actions(state)) > 0:
            max_q_value = max([q_values[state][action] for action in env.get_possible_actions(state)])
            there_was_max = False
            for action in env.get_possible_actions(state):
                if q_values[state][action] == max_q_value and not there_was_max:
                    policy[state][action] = 1
                    there_was_max = True
                else:
                    policy[state][action] = 0
    return policy


def policy_iteration(env, gamma, N=20, M=20):
    policy = init_policy(env)
    for _ in range(N):
        values = policy_evaluation(env, gamma, policy, M)
        policy = policy_improvement(env, gamma, values)
    return policy


def get_total_reward(env, policy, session_len):
    total_reward = 0
    state = env.reset()
    for _ in range(session_len):
        prob = [policy[state][action] for action in env.get_possible_actions(state)]
        action = np.random.choice(env.get_possible_actions(state), p=prob)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def policy_test(env, policy, session_n, session_len=100):
    total_rewards = np.array([get_total_reward(env, policy, session_len) for _ in range(session_n)])
    return np.mean(total_rewards)


def value_iteration(env, gamma, N=20):
    values = init_values(env)
    for _ in range(N):
        q_values = get_q_values(env, gamma, values)
        for state in env.get_all_states():
            if len(env.get_possible_actions(state)) > 0:
                values[state] = max(q_values[state][action] for action in env.get_possible_actions(state))

    policy = policy_improvement(env, gamma, values)
    return policy


env = FrozenLakeEnv()
gamma = 0.99

policy = value_iteration(env, gamma, N=500)
print('value_iteration:', policy_test(env, policy, session_n=500))

policy = policy_iteration(env, gamma, N=20, M=20)
print('policy_iteration:', policy_test(env, policy, session_n=500))


