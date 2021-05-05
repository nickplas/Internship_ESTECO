import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import gym
import torch.optim as optim
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, layer1_size, layer2_size, action_size):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, layer1_size)
        self.linear2 = nn.Linear(layer1_size, layer2_size)
        self.linear3 = nn.Linear(layer2_size, action_size)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        return self.linear3(out)


class DQN(OffPolicyAlgorithm):
    def __init__(self, env, net, target, optim, loss, device):
        super(DQN, self).__init__(net, env, target, learning_rate=0.001)
        self.env = env
        self.memory = []
        self.net = net
        self.target = target
        self.target.load_state_dict(net.state_dict())
        self.target.eval()
        self.optimizer = optim
        self.loss = loss
        self.device = device

    def get_action(self, eps, s):
        state = torch.tensor(s).float().unsqueeze(0).to(self.device)
        self.net.eval()
        with torch.no_grad():
            action = self.net(state)
        self.net.train()
        if random.random() < eps:
            return torch.tensor([self.env.action_space.sample()])
        return torch.tensor([np.argmax(action.cpu().data.numpy())])

    def train(self, batch_size, gamma):
        if len(self.memory) < batch_size:
            return

        sample_memory = random.sample(self.memory, batch_size)
        state = torch.tensor([item[0] for item in sample_memory]).float()
        action = torch.tensor([item[1] for item in sample_memory]).view(-1, 1).long()
        reward = torch.tensor([item[2] for item in sample_memory]).float()
        next_state = torch.tensor([item[3] for item in sample_memory]).float()
        done = torch.tensor([int(item[4]) for item in sample_memory]).float()

        q_current = self.net(state).gather(1, action)

        with torch.no_grad():
            q_next = self.target(next_state).detach()
            max_q_next = q_next.max(1)[0].unsqueeze(1)

        q_values = torch.sum(torch.stack([reward.view([64, 1]), (gamma * max_q_next * ((1 - done).view([64, 1])))]), dim=0)

        score = self.loss(q_current, q_values)
        self.optimizer.zero_grad()
        score.backward()
        self.optimizer.step()

    def run(self, episodes, steps, eps, bs, gamma, target_update, capacity):
        for i in range(episodes):
            state = self.env.reset()
            for j in range(steps):
                self.env.render()
                action = self.get_action(eps, state)
                obs, reward, done, _ = self.env.step(action.item())
                if len(self.memory) >= capacity:
                    self.memory.pop(0)
                self.memory.append([state, action, reward, obs, done])
                self.train(bs, gamma)
                state = obs
                if done:
                    break
                if j % target_update == 0:
                    self.target.load_state_dict(self.net.state_dict())


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = NeuralNetwork(env.observation_space.shape[0], 32, 32, env.action_space.n).to(device)
    t_net = NeuralNetwork(env.observation_space.shape[0], 32, 32, env.action_space.n).to(device)
    optim = optim.Adam(net.parameters(), lr=0.001)

    agent = DQN(env, net, t_net, optim, nn.MSELoss())
    agent.run(100, 1000, 0.1, 64, 0.9, 5, 10000)




