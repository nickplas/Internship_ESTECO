import gym
from gym import spaces
import random
import numpy as np
import matplotlib.pyplot as plt


class ContinuousEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, training_bool, min_action, max_action):
        super(ContinuousEnv, self).__init__()
        #self.f = func1  # come funzione python
        self.lower_bound = np.array([0, -99999], dtype=np.float32)  # min_state
        self.upper_bound = np.array([100, 99999], dtype=np.float32)  # max_state
        self.param = [random.uniform(self.lower_bound[0], self.upper_bound[0]) for _ in range(4)]
        self.x = random.uniform(self.lower_bound[0], self.upper_bound[0])  # x
        self.y = self.f(self.x)
        self.H = 1  # number of gradients to be stored
        ''' state is a tuple composed by: current iterate, objective values, past and current gradients'''
        self.iteration = 0
        self.h = 1e-4
        self.difference = [self.f(self.x)]  # need to be a list
        self.gradient = [self.g(self.f, self.x)]  # need to be a list
        self.initial_state = np.array([self.difference, self.gradient])  # need to be a np.array
        self.state = self.initial_state
        self.min_action = min_action  # min_action
        self.max_action = max_action  # max_action
        self.eps = 0.01  # eps # o inizializzare ad un valore basso
        self.episodes = 0
        self.training = training_bool
        self.min = self.compute_min()

        # Figures for rendering
        self.x_axis = np.linspace(self.lower_bound[0], self.upper_bound[0], 100000)
        self.y_axis = self.f(self.x_axis)
        plt.plot(self.x_axis, self.y_axis)
        plt.ion()
        plt.show()

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,),
            dtype=np.float32)

        self.observation_space = spaces.Box(  # real shape is (2,)
            low=self.lower_bound,
            high=self.upper_bound,
            dtype=np.float32)

    def step(self, action):
        reward = self.reward(action)
        done = int(self.x < self.lower_bound[0] or self.x > self.upper_bound[0] or
                   reward >= 999*abs(self.y).item())
        obs = np.array(self.flatten()).reshape(self.observation_space.shape[0])
        info = {}
        # self.render()
        return obs, reward, done, info

    def flatten(self):  # the state needs to be flatten + iteration removed
        return np.array(self.difference + self.gradient)

    def reset(self):
        self.episodes += 1
        if self.episodes % 150 == 0:
            self.param = [random.uniform(self.lower_bound[0], self.upper_bound[0]) for _ in range(4)]
            self.min = self.compute_min()
        self.x = random.uniform(self.lower_bound[0], self.upper_bound[0])
        self.difference = [self.f(self.x)]  # need to be a list
        self.gradient = [self.g(self.f, self.x)]  # need to be a list
        self.initial_state = np.array([self.difference, self.gradient])
        self.state = self.initial_state
        result = np.array(self.flatten().reshape(self.observation_space.shape[0]))
        return result  # self.state

    def render(self, mode='human'):  # plot funzione e punto x e y
        plt.plot(self.x, self.y, 'o')
        plt.show()
        plt.pause(1)
        return str(self.state)

    # def close(self):
    #  ...

    def sample(self):  # NOTE: da modificare
        return np.random.uniform(self.min_action, self.max_action)

    def reward(self, action):
        self.update(action)
        # reward = (-self.y*0.01).item()
        if self.x - self.lower_bound[0] < 0 or self.upper_bound[0] - self.x < 0:
            return -100000*abs(self.x)
        # if abs(self.gradient[0]) < self.eps:
        #     reward = 10
        # print('x', self.x)
        if self.training and abs(self.x - self.min) < self.eps:
            print('minimum has been reached', self.min)
            return 1000*abs(self.y).item()
        return (-self.y).item()

    def update(self, action):
        self.iteration += 1
        old = self.f(self.x)
        self.x += action
        self.y = self.f(self.x)
        diff = self.y - old
        self.difference.append(diff)
        gradient = self.g(self.f, self.x)
        # print('grad update', gradient)
        self.gradient.append(gradient)
        # print('self.grad update', self.gradient)
        if len(self.difference) > self.H:
            self.difference.pop(0)
            self.gradient.pop(0)
        for i in range(len(self.state[0]) - 2, -1, -1):
            self.difference[i] += self.difference[i + 1]
        self.state = np.array([self.difference, self.gradient])
        # print('after self.grad update', self.gradient)

    def g(self, func, x):
        # if self.iteration % 5 == 0:
        #     self.h -= self.h*.1
        result = (func(x + self.h) - func(x)) / self.h
        return result

    def remove_from_memory(self, memory):
        l = []
        for i in range(len(memory)):
            if len(memory[i][0]) != self.H * 2:
                l.append(i)
        l.reverse()
        for i in l:
            memory.pop(i)
        return memory

    def init_memory_and_state(self, bs):
        memory = []
        state = self.reset()
        for i in range(bs + self.H):
            action = self.sample()
            obs, reward, done, info = self.step(action)
            memory.append([state, action, reward, obs, done])
            state = obs
        memory = self.remove_from_memory(memory)
        return memory

    def f(self, x):
        return (x - self.param[0])*(x - self.param[1])*(x - self.param[2])*(x - self.param[3])

    def compute_min(self):  # return the x of the global minimum
        parameters = sorted(self.param)
        if (parameters[1] - parameters[0]) < (parameters[3] - parameters[2]):
            return parameters[2] + (parameters[3] - parameters[2])/2
        return parameters[0] + (parameters[1] - parameters[0])/2

    # TODO: parametrizzare la funzione (x-a)(x-b)(x-c)(x-d) all'interno dell'env
    # TODO: avere un attributo che memorizza il punto di minimo della funzione (magari usare un bool che
    #       ci permette di capiare se siamo nel training, quindi ci serve sto valore, o in testing)
    # TODO: allenare la funzione con diversi valori di a,b,c,d (cambiano ogni x episodi)
    # TODO: fare un training con molti steps in modo da permettere all'agente di esplorare



