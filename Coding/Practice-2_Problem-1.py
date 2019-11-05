import torch
from torch import nn

class Agent(nn.Module):

    def __init__(self):
        super().__init__()

        self.dense_1 = nn.Linear(1,100)
        self.dense_2 = nn.Linear(100,1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.1)

    def forward(self, input):
        hidden = self.dense_1(input)
        hidden = self.tanh(hidden)
        output = self.dense_2(hidden)
        return output

    def learn(self, input, target):
        for _ in range(1000):
            loss = torch.mean((self.forward(input) - target) ** 2)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

#Data
x = torch.tensor([0.01 * i for i in range(300)]).reshape(300,1)
noise = torch.tensor([torch.normal(torch.tensor(0.1), torch.tensor(0.1)) for i in range(300)]).reshape(300,1)
y = torch.sin(x) + noise

#Agent learn
agent = Agent()
agent.learn(x, y)

#Show
import matplotlib.pyplot as plt
plt.plot(x.numpy(), y.numpy())
plt.plot(x.numpy(), agent(x).data.numpy())
plt.show()