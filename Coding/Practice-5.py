import numpy as np
import torch
from torch import nn
import random
import gym

class Network(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, 32)
        self.linear_2 = nn.Linear(32, 32)
        self.linear_3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()

    def forward(self, input):
        hidden = self.linear_1(input)
        hidden = self.relu(hidden)
        hidden = self.linear_2(hidden)
        hidden = self.relu(hidden)
        output = self.linear_3(hidden)
        return output

class DQNAgent(nn.Module):

    def __init__(self, state_dim, action_n):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n

        self.gamma = 0.95
        self.epsilon = 1
        self.memory_size = 10000
        self.memory = []
        self.batch_size = 64
        self.learinig_rate = 1e-2

        self.q = Network(self.state_dim, self.action_n)
        self.optimazer = torch.optim.Adam(self.q.parameters(), lr=self.learinig_rate)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        argmax_action = torch.argmax(self.q(state))
        probs = np.ones(self.action_n) * self.epsilon / self.action_n
        probs[argmax_action] += 1 - self.epsilon
        actions = np.arange(self.action_n)
        action = np.random.choice(actions, p=probs)
        return action

    def fit(self, state, action, reward, done, next_state):

        self.memory.append([state, action, reward, done, next_state])
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)

            states, actions, rewards, dones, next_states = list(zip(*batch))
            states = torch.FloatTensor(states)
            q_values = self.q(states)
            next_states = torch.FloatTensor(next_states)
            next_q_values = self.q(next_states)
            targets = q_values.clone()
            for i in range(self.batch_size):
                targets[i][actions[i]] = rewards[i] + self.gamma * (1 - dones[i]) * max(next_q_values[i])

            loss = torch.mean((targets.detach() - q_values) ** 2)

            loss.backward()
            self.optimazer.step()
            self.optimazer.zero_grad()

            if self.epsilon > 0.01:
                self.epsilon *= 0.999

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_n = env.action_space.n
agent = DQNAgent(state_dim, action_n)

episode_n = 100
for episode in range(episode_n):
    state = env.reset()
    total_reward = 0
    for t in range(500):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.fit(state, action, reward, done, next_state)
        state = next_state
        total_reward += reward
        if done:
            break
    print(total_reward)




