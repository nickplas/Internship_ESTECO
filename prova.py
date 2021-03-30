import gym
from algorithm.common import *

import numpy as np
#from gym-RL.gym_RL.envs import RLEnv
env = gym.make('gym_RL:RL-v0', prob = 0.5, initial_state = 49)

env.reset()
#done = False
#while not done:
#    state, reward, done, _ = env.step(1)
#    print(state, reward, done)

Q = []
Q.append(np.zeros(1))
for i in range(1,100):
    Q.append(np.ones(i+1))
Q.append(np.zeros(101))

Q = Q_learning(env, Q, 2, 100, 0.1, 0.99, 0.2)
#print(Q)


