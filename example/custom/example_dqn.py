from algorithm import DQN
from algorithm.dqn.dqn import NeuralNetwork
import gym
import torch.optim as optim
import torch.nn as nn
import torch


env = gym.make("CartPole-v0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = NeuralNetwork(env.observation_space.shape[0], 24, 24, env.action_space.n).to(device)
t = NeuralNetwork(env.observation_space.shape[0], 24, 24, env.action_space.n).to(device)
t.load_state_dict(net.state_dict())

opt = optim.Adam(net.parameters())
l = nn.MSELoss()

model = DQN(env, net, t, opt, l, device)
model.run(100, 1000, 0.1, 64, 0.9, 5, 10000)