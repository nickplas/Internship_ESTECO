import gym
from gym import error, spaces, utils
from gym.utils import seeding
#import numpy as np
import random

class RLEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, prob = 0.4, initial_state = 20):
    super(RLEnv, self).__init__()
    self.prob = prob
    self.state = initial_state
    self.initial_state = initial_state

  def step(self, action):
    reward = self.reward(action)
    done = (reward == 1 or self.state <= 0)
    obs = self.state
    info = {}
    return obs, reward, done, info  
    
  def reset(self):
    self.state = self.initial_state
    return self.state
    
  def render(self, mode='human'):
    return str(self.state)

  #def close(self):
  #  ...

  def reward(self, action):
    if(random.random() < self.prob):
      self.state += action
      return int(self.state >= 100)
    else:
      self.state -= action
      return 0

  def sample(self):
    possible_actions = range(1, min(100 - self.state, self.state)+1)
    return random.choice(possible_actions)
    