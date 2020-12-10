import gym
import numpy as np
import torch
from torch import nn


class CrossEntropyAgent(nn.Module):
    def __init__(self, state_dim, action_n):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.ReLU(),
            nn.Linear(100, action_n)
        )
        self.softmax = nn.Softmax()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, input):
        return self.network(input)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.network(state)
        action_prob = self.softmax(logits).detach().numpy()
        action = np.random.choice(len(action_prob), p=action_prob)
        return action

    def update_policy(self, elite_sessions):
        elite_states, elite_actions = [], []
        for session in elite_sessions:
            elite_states.extend(session['states'])
            elite_actions.extend(session['actions'])

        elite_states = torch.FloatTensor(elite_states)
        elite_actions = torch.LongTensor(elite_actions)

        loss = self.loss(self.network(elite_states), elite_actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return None



def get_session(env, agent, session_len, visual=False):
    session = {}
    states, actions = [], []
    total_reward = 0

    state = env.reset()
    for _ in range(session_len):
        states.append(state)
        action = agent.get_action(state)
        actions.append(action)

        if visual:
            env.render()

        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    session['states'] = states
    session['actions'] = actions
    session['total_reward'] = total_reward
    return session


def get_elite_sessions(sessions, q_param):

    total_rewards = np.array([session['total_reward'] for session in sessions])
    quantile = np.quantile(total_rewards, q_param)

    elite_sessions = []
    for session in sessions:
        if session['total_reward'] > quantile:
            elite_sessions.append(session)

    return elite_sessions


env = gym.make("CartPole-v1")
agent = CrossEntropyAgent(4, 2)

episode_n = 100
session_n = 20
session_len = 500
q_param = 0.8

for episode in range(episode_n):
    sessions = [get_session(env, agent, session_len) for _ in range(session_n)]

    mean_total_reward = np.mean([session['total_reward'] for session in sessions])
    print('mean_total_reward = ', mean_total_reward)

    if mean_total_reward > 400:
        print('You win!')

    elite_sessions = get_elite_sessions(sessions, q_param)

    if len(elite_sessions) > 0:
        agent.update_policy(elite_sessions)

get_session(env, agent, session_len, visual=True)
