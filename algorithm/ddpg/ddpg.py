# from code import actor_critic
import numpy as np
# import torch.optim as optim
import gym
import torch
# import torch.nn as nn
import random


class DDPG:
    def __init__(self, env, anet, t_anet, optimA, cnet, t_cnet, optimC, loss, capacity):
        self.env = env

        # actor
        self.actor = anet
        self.t_actor = t_anet
        self.optimA = optimA

        # critic
        self.critic = cnet
        self.t_critic = t_cnet
        self.optimC = optimC

        # target
        self.t_actor.load_state_dict(self.actor.state_dict())
        self.t_actor.eval()
        self.t_critic.load_state_dict(self.critic.state_dict())
        self.t_critic.eval()

        # loss
        self.critic_loss = loss

        # memory
        self.memory = []  # state, action, reward, next_action, done
        self.capacity = capacity

    def gaussian_noise(self, std):
        return np.random.normal(0, std)

    def get_action(self, s, std):  # Gli passo una lista e non un tensore, da cambiare se gli passo tensori
        # print('state inside action', s)
        tensor_state = torch.tensor(s).float()#.unsqueeze(0)#.to(device)
        # tensor_state = tensor_state.reshape(tensor_state.size()[0], 1)  # serve un vettore colonna
        # print('tensor state', tensor_state)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(tensor_state)
            # print('action, get action', action)
        self.actor.train()
        #sigma = abs(sigma.item())
        #action = np.random.normal(mu.item(), sigma)
        #print('action', action)
        noise = self.gaussian_noise(std)  # rimettere su un'unica riga
        result = action + noise
        #print('risultato', result)
        return result

    def push_in_memory(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)  # Controllare se rimuovere il primo o uno a caso
        self.memory.append([state, action, reward, next_state, done])

    def sample_from_memory(self, batch_size):
        if len(self.memory) < batch_size:
            return
        return random.sample(self.memory, batch_size)

    def train(self, batch_size, gamma, tau):
        sample_memory = self.sample_from_memory(batch_size)
        if sample_memory is None:
            return

        # Build tensors
        state = torch.tensor([item[0] for item in sample_memory]).float()
        action = torch.tensor([item[1] for item in sample_memory]).view(-1, 1).long()
        reward = torch.tensor([item[2] for item in sample_memory]).float()
        next_state = torch.tensor([item[3] for item in sample_memory]).float()
        # done = torch.tensor([int(item[4]) for item in sample_memory]).float()  # TODO: rimuovere se non usata

        with torch.no_grad():
            target_action = self.t_actor(next_state)
            target_q = self.t_critic(torch.cat([next_state, target_action], 1))

        y = torch.sum(torch.stack([reward.view([batch_size, 1]), (gamma*target_q)]), dim=0)

        q_values = self.critic(torch.cat([state, action], 1))
        # print('q_val', q_values)

        # Update critic network
        score = self.critic_loss(y, q_values)
        self.optimC.zero_grad()
        score.backward()
        self.optimC.step()

        # update actor network
        pred_action = self.actor(state)
        pred_q = self.critic(torch.cat([state, pred_action], 1))
        loss = -1*torch.mean(pred_q)
        self.optimA.zero_grad()
        loss.backward()
        self.optimA.step()

        # soft update of target
        for param_target, param_net in (zip(self.t_actor.parameters(), self.actor.parameters())):
            param_target = (tau*param_net + (1 - tau)*param_target).clone().detach

        for param_target, param_net in (zip(self.t_critic.parameters(), self.critic.parameters())):
            param_target = (tau*param_net + (1 - tau)*param_target).clone().detach

    def run(self, episodes, steps, std, bs, gamma, tau):
        if hasattr(self.env, 'init_memory_and_state') and callable(getattr(self.env, 'init_memory_and_state')):
            self.memory = self.env.init_memory_and_state(bs)
        for i in range(episodes):
            print("Episode: ", i+1 )
            state = self.env.reset()
            for j in range(steps):
                print("Step: ", j + 1)
                self.env.render()
                action = self.get_action(state, std)#.item()
                obs, reward, done, info = self.env.step(action)
                self.push_in_memory(state, action, reward, obs, done)
                self.train(bs, gamma, tau)
                state = obs
                if done:
                    break
        # self.env.close()


if __name__ == "__main__":

    # Gym Environment
    env = gym.make('MountainCarContinuous-v0')  # 'Pendulum-v0'  # 'MountainCarContinuous-v0'  # 'LunarLanderContinuous-v2'

    # Neural Nets
    # actor = actor_critic.Actor(env.observation_space.shape[0], 24, 24, env.action_space.shape[0])
    # t_actor = actor_critic.Actor(env.observation_space.shape[0], 24, 24, env.action_space.shape[0])
    # actor_optimizer = optim.Adam(actor.parameters(), lr=0.0005)
    # critic = actor_critic.Critic(env.observation_space.shape[0], 32, 32)
    # t_critic = actor_critic.Critic(env.observation_space.shape[0], 32, 32)
    # critic_optimizer = optim.Adam(critic.parameters(), lr=0.005)

    # # loss
    # loss = nn.MSELoss()
    #
    # # Device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # # Test code
    # agent = DDPG(env, actor, t_actor, actor_optimizer, critic, t_critic, critic_optimizer, loss, 1000)
    # agent.run(10, 10000, 1, 64, 0.99, 0.001)



