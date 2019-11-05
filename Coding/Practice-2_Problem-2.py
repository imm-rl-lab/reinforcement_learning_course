import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class CrossEntropyAgent(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        self.softmax = nn.Softmax()
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.parameters(), lr=0.01)

    def forward(self, x):
        return self.net(x)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.net(state)
        action_probs = self.softmax(logits).data.numpy()
        action = np.random.choice(len(action_probs), p=action_probs)
        return action

    def fit(self, states, actions):
        logits = self.net(states)
        loss = self.loss_function(logits, actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def update_policy(self, elite_sessions):
        elite_states, elite_actions = [], []
        for session in elite_sessions:
            states, actions, _ = session
            elite_states.extend(states)
            elite_actions.extend(actions)
        elite_states = torch.FloatTensor(elite_states)
        elite_actions = torch.LongTensor(elite_actions)
        #elite_actions = elite_actions.reshape(elite_actions.shape[0], 1)
        logits = self.net(elite_states)
        loss = self.loss_function(logits, elite_actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

def get_session(agent, env, t_max=1000, visual=False):
    session_states, session_actions = [], []
    session_reward = 0
    state = env.reset()

    for t in range(t_max):

        if visual:
            env.render()

        session_states.append(state)

        action = agent.get_action(state)
        session_actions.append(action)

        state, reward, done, _ = env.step(action)

        session_reward += reward

        if done:
            break

    session = [session_states, session_actions, session_reward]
    return session


def get_elite_sessions(sessions, percentile):

    session_rewards = np.array([sessions[i][2] for i in range(len(sessions))])

    reward_threshold = np.percentile(session_rewards, percentile)

    elite_sessions = []

    for session in sessions:
        _, _, session_reward = session
        if session_reward >= reward_threshold:
            elite_sessions.append(session)

    mean_rewards = np.mean(session_rewards)

    return elite_sessions, mean_rewards

EPISODE_N = 100
SESSION_N = 20
PERCENTILE = 80

env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]
action_n = env.action_space.n

agent = CrossEntropyAgent(state_dim, action_n)

for episode in range(EPISODE_N):
    sessions = [get_session(agent, env) for i in range(SESSION_N)]

    elite_sessions, mean_rewards = get_elite_sessions(sessions, PERCENTILE)

    if len(sessions) != len(elite_sessions):
        loss = agent.update_policy(elite_sessions)

        print("%d: loss=%.3f, mean_reward=%.1f" % (
            episode, loss.item(), mean_rewards))

        if mean_rewards > 400:
            print("You Win!")
            get_session(agent, env, visual=True)

