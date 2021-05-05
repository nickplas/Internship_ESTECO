from algorithm import DDPG, Actor, Critic

import gym
import torch.optim as optim
import torch.nn as nn

env = gym.make("MountainCarContinuous-v0")

actor = Actor(env.observation_space.shape[0], 24, 24, 1)
t_actor = Actor(env.observation_space.shape[0], 24, 24, 1)
optimA = optim.Adam(actor.parameters(), lr=0.00001)
critic = Critic(env.observation_space.shape[0], 24, 24)
t_critic = Critic(env.observation_space.shape[0], 24, 24)
optimC = optim.Adam(critic.parameters(), lr=0.00005)
loss = nn.MSELoss()
agent = DDPG(env, actor, t_actor, optimA, critic, t_critic, optimC, loss, 10000)
agent.run(50, 10, 0.5, 64, 0.99, 0.001)
