import gym
from algorithm.common import sarsa
from algorithm.common import Q_learning
import numpy as np

# from gym-RL.gym_RL.envs import RLEnv
# env = gym.make('gym_RL:RL-v0', prob = 0.5, initial_state = 49)

# env.reset()
# done = False
# while not done:
#    state, reward, done, _ = env.step(1)
#    print(state, reward, done)

# Q = []
# Q.append(np.zeros(1))
# for i in range(1,100):
#    Q.append(np.ones(i+1))
# Q.append(np.zeros(101))

# Q = Q_learning(env, Q, 2, 100, 0.1, 0.99, 0.2)
# print(Q)

def f(x):
    return x**2

env = gym.make('gym_Continuous:Continuous-v0', func1 = f)
env.reset()

done = False
while not done:
    obs, reward, done, _ = env.step(-0.01)
    print(obs, reward, done)

