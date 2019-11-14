from Frozen_Lake import FrozenLakeEnv
env = FrozenLakeEnv()

# print(env.get_all_states())
# state = (3, 2)
# print(env.get_possible_actions(state))
# action = 'right'
# print(env.get_next_states(state, action))
# next_state
# = (3, 3)
# print(env.get_reward(state, action, next_state))
# print(env.get_transition_prob(state, action, next_state))
# print(env.is_terminal(next_state))

import numpy as np

GAMMA = 0.99

def test(pi, test_n=1000, step_n=100):
    rewards = np.zeros(test_n)
    for i in range(test_n):
        s = env.reset()
        for t in range(step_n):
            #env.render()
            prob = [pi[s][a] for a in env.get_possible_actions(s)]
            a = np.random.choice(env.get_possible_actions(s), p=prob)
            next_s, reward, done, _ = env.step(a)
            rewards[i] += reward# * GAMMA ** t
            s = next_s
            if done:
                break
    return np.mean(rewards)

def get_q(v):
    q = {}
    for s in env.get_all_states():
        q[s] = {}
        for a in env.get_possible_actions(s):
            q[s][a] = 0
            for next_s in env.get_next_states(s, a):
                q[s][a] += env.get_transition_prob(s, a, next_s) * \
                           (env.get_reward(s, a, next_s) + GAMMA * v[next_s])
    return q

def policy_evaluation(pi, iteration_n=100):
    v = {s : 0 for s in env.get_all_states()}
    for _ in range(iteration_n):
        q = get_q(v)
        new_v = {}
        for s in env.get_all_states():
            new_v[s] = 0
            for a in env.get_possible_actions(s):
                new_v[s] += pi[s][a] * q[s][a]
        v = new_v
    return v

def policy_improvment(v):
    q = get_q(v)
    pi = {}
    for s in env.get_all_states():
        pi[s] = {}
        if len(env.get_possible_actions(s)) != 0:
            q_max = max([q[s][a] for a in env.get_possible_actions(s)])
            there_was_max = False
            for a in env.get_possible_actions(s):
                if q[s][a] == q_max and not there_was_max:
                    pi[s][a] = 1
                    there_was_max = True
                else:
                    pi[s][a] = 0
    return pi

def policy_iteration(iteration_n=20):
    pi = {s : {a : 1/4 for a in env.get_possible_actions(s)} for s in env.get_all_states()}
    for i in range(iteration_n):
        v = policy_evaluation(pi)
        pi = policy_improvment(v)
    return pi

def value_iteration(iteration_n=20):
    v = {s: 0 for s in env.get_all_states()}
    for _ in range(iteration_n):
        q = get_q(v)
        for s in env.get_all_states():
            if len(env.get_possible_actions(s)) != 0:
                v[s] = max(q[s][a] for a in env.get_possible_actions(s))
    return policy_improvment(v)

pi = policy_iteration()
print(pi)
print(test(pi))

pi = value_iteration()
print(pi)
print(test(pi))

