#Imports
import gym
import numpy as np

#Notes
#observation is made up by: cart position, cart velocity, pole angle and
#                           pole velocity at tip

#actions : 0 pushes the cart to the left
#          1 pushes the cart to the right 

#Definitions

# Policy based on the angle, no RL used
def policy_angle(observation):
    _, _, angle, _ = observation 
    if angle < 0:
        return 0
    else:
        return 1

def Q_learning(env):
    # Approccio Q-learning, non per forza tutte le cose in questa funzione
    # fare una tabella di dimensione n.stati, n.azioni
    # usare l'approccio epsilon-greedy per decidere l'azione
    # provare con epsilon a 50% e molti episodi

    #initialize Q(state, action) arbitrarily except for Q(terminal, _) = 0
    Q = np.zeros((len(states), len(actions)))

    #loop for each episode
        #initialize S
        #loop for each step episode
            #Choose A form S using Q
            #Take action A, observe R and S
            #update rule 
            #update S
        #until S is terminal

    return

#Main
if __name__=="__main__":
    env = gym.make('CartPole-v0')
    obs = env.reset()
    print(obs)



    #for i_episode in range(20):
    #    observation = env.reset()
    #    cumulate_reward = 0
    #    for t in range(100):
    #        env.render()
    #        #print(observation)
    #        action = policy_angle(observation) #env.action_space.sample()
    #        observation, reward, done, info = env.step(action)
    #        cumulate_reward += reward
    #        if done:
    #            print("Episode finished after {} timesteps".format(t+1))
    #            break
    #env.close()

