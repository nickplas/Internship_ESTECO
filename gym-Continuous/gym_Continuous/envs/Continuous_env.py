import gym
from gym import error, spaces, utils
from gym.utils import seeding
#import numpy as np
import random

class ContinuousEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(ContinuousEnv, self).__init__()
    
  def step(self:
    reward = 
    done = 
    obs = 
    info = {}
    return obs, reward, done, info  
    
  def reset(self):
    self.state = self.initial_state
    return self.state
    
  def render(self, mode='human'):
    return str(self.state)

  #def close(self):
  #  ...

    