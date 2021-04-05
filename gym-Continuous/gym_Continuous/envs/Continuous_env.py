import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math

class ContinuousEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, func1):
    super(ContinuousEnv, self).__init__()
    self.f = func1 # come funzione python
    #self.grad = func2 da calcolare all'
    self.x = 5 # x 
    ''' state is a tuple composed by: current iterate, objective values, past and current gradients'''
    self.initial_state = [0, [self.f(self.x)],[self.g(self.f, self.x)]] # iteration, obj value, gradient of last h values
    self.state = self.initial_state
    self.H = 25 # number of gradients to be stored
    self.min_action = -1 # min_action
    self.max_action = 1 # max_action
    self.lower_bound = -5 # min_state
    self.upper_bound = 5 # max_state
    self.eps = 0.01 #eps # o inizializzare ad un valore basso
    
  def step(self, action):
    reward = self.reward(action)
    done = (reward > 1 or self.x < self.lower_bound or self.x > self.upper_bound)
    obs = self.x
    info = {}
    return obs, reward, done, info  
    
  def reset(self):
    self.state = self.initial_state
    return self.state
    
  def render(self, mode='human'):
    return str(self.state)

  #def close(self):
  #  ...

  def sample(self): # NOTE: da modificare
    return np.random.uniform(self.min_action, self.max_action)

  def reward(self, action): # -1 se si allontana, +1 se si avvicina, +100 se raggiunge il punto
    self.update(action)
    if  (np.abs(self.state[2][-1]) < self.eps and action != 0):
      return 100
    elif self.state[1][-1] > 0 :
      return -1
    elif  self.state[1][-1] < 0: 
      return 1
    else:
      return 0

  def update(self, action):
    old = self.f(self.x)
    self.x += action
    obj = self.f(self.x)
    self.state[1].append(obj-old)
    gradient = self.g(self.f, self.x)
    self.state[2].append(gradient)
    self.state[0] +=1
    if (self.state[0] >= self.H):
      self.state[2].pop(0)
      self.state[1].pop(0)
    for i in range(len(self.state[1])-2, -1, -1):
      self.state[1][i] += self.state[1][i+1]

  def g(self, func, x): 
    h = 1e-10
    return (func(x+h)-func(x))/h




    