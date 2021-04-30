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
        torch.nn.init.uniform_(self.fc1.weight, -0.0001, 0.0001)
        self.fc2 = nn.Linear(h1_size, h2_size)
        torch.nn.init.uniform_(self.fc2.weight, -0.0001, 0.0001)
        self.fc3 = nn.Linear(h2_size, out_size)
        torch.nn.init.uniform_(self.fc3.weight, -0.0001, 0.0001)
        # self.bn1 = nn.BatchNorm1d(h1_size)
        # self.bn2 = nn.BatchNorm1d(h2_size)

    def forward(self, x):
        # x = x.unsqueeze(0)
        # print('printing input', x)
        out = F.relu(self.fc1(x))  # x.view([50, 1])))
        # print('printing out after fc1', out.size())
        out = F.relu(self.fc2(out))
        # print('printing out after fc2', out.size())
        # print('last size', self.fc3(out).size())
        return torch.tanh(self.fc3(out))
