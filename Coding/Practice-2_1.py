import numpy as np
import torch
import matplotlib.pyplot as plt

x_data = torch.linspace(-5, 5, steps=300)
nu, sigma = torch.tensor(0.2), torch.tensor(0.5)
noise = torch.tensor([torch.normal(nu, sigma) for _ in range(300)])
y_data = x_data + noise

w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

learning_rate = 0.1
learning_step_n = 20
for _ in range(learning_step_n):
    loss = torch.mean((w * x_data + b - y_data) ** 2)
    print(loss)
    loss.backward()
    w.data = w.data - learning_rate * w.grad
    b.data = b.data - learning_rate * b.grad
    w.grad.zero_()
    b.grad.zero_()

plt.scatter(x_data.numpy(), y_data.numpy())
y = w * x_data + b
plt.plot(x_data.numpy(), y.detach().numpy(), 'r')
plt.show()



