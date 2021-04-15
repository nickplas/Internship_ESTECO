import torch.nn as nn
import torch.nn.functional as F
import torch


class Critic(nn.Module):
    def __init__(self, input_size, h1_size, h2_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size+1, h1_size)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.fc3 = nn.Linear(h2_size, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        return self.fc3(out)


class Actor(nn.Module):
    def __init__(self, input_size, h1_size, h2_size, out_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, h1_size)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.fc3 = nn.Linear(h2_size, out_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        return torch.tanh(self.fc3(out))
