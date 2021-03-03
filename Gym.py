#Imports
import gym
import numpy as np

#Notes
#observation is made up by: cart position, cart velocity, pole angle and
#                           pole velocity at tip

#Definitions

# Policy based on the angle, no RL used
def policy_angle(observation):
    _, _, angle, _ = observation 
    if angle < 0:
        return 0
    else:
        return 1

def policy_eps_greed():
    # Approccio Q-learning, non per forza tutte le cose in questa funzione
    # fare una tabella di dimensione n.stati, n.azioni
    # usare l'approccio epsilon-greedy per decidere l'azione
    # provare con epsilon a 50% e molti episodi
    return

#Main
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    cumulate_reward = 0
    for t in range(100):
        env.render()
        print(observation)
        action = policy_angle(observation) #env.action_space.sample()
        observation, reward, done, info = env.step(action)
        cumulate_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

