import numpy as np
import gym
# import torch.optim as optim
# import torch.nn as nn
# import torch

# from code import ddpg
# from code import actor_critic


def f(x):
    return x*(x-1)*(x-3)*(x-7)# np.maximum((x-5)**2, x**2)  # x**2 -5# -x**4 * np.cos(0.9*x + 0.1)  # np.cos(3*np.pi*x)/x  # x**2

# My DDPG test

# from stable_baselines3.common.env_checker import check_env

# check_env(env)

# actor = actor_critic.Actor(env.observation_space.shape[0], 24, 24, 1)
# t_actor = actor_critic.Actor(env.observation_space.shape[0], 24, 24, 1)
# optimA = optim.Adam(actor.parameters(), lr=0.00001)
# critic = actor_critic.Critic(env.observation_space.shape[0], 24, 24)
# t_critic = actor_critic.Critic(env.observation_space.shape[0], 24, 24)
# optimC = optim.Adam(critic.parameters(), lr=0.00005)
# loss = nn.MSELoss()
# agent = ddpg.DDPG(env, actor, t_actor, optimA, critic, t_critic, optimC, loss, 10000)
# agent.run(50, 10, 0.5, 64, 0.99, 0.001)

env = gym.make('gym_Continuous:Continuous-v0', func1=f, min_action=-0.1, max_action=0.1)

# from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
n_actions = env.action_space.shape[-1]
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(1.) * np.ones(n_actions))
action_noise = NormalActionNoise(np.zeros(n_actions), sigma=float(2)*np.ones(n_actions))

# SAC
from stable_baselines3 import SAC
# env.reset()
# model = SAC("MlpPolicy", env, ent_coef='auto_0.5', action_noise=action_noise)  # , learning_rate=0.001, verbose=0, ent_coef='auto', learning_starts=300)
# model.learn(total_timesteps=3000)
# model.save("SAC-x^2-secondo")

# TD3
from stable_baselines3 import TD3
# action_noise = NormalActionNoise(np.zeros(n_actions), sigma=float(0.5)*np.ones(n_actions))
# model = TD3("MlpPolicy", env, learning_starts=300, action_noise=action_noise, batch_size=32)
# model.learn(total_timesteps=1000, log_interval=10)
# model.save('TD3-x^2')


def f1(x):
    return x*(x-1)*(x-3)*(x-7)  # np.maximum((x+5)**2, x**2)

model = SAC.load("SAC-x^2-secondo")
# model = TD3.load("TD3-x^2")
env1 = gym.make('gym_Continuous:Continuous-v0', func1=f1, min_action=-100, max_action=100)
# testing
print('Testing Part')
obs = env1.reset()
while True:
    action, _state = model.predict(obs, deterministic=True)
    obs, r, done, _ = env1.step(action)
    env1.render()
    if done:
        env1.reset()
        print("finish")
        break


