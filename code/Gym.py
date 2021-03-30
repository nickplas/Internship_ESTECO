#Imports
import gym
import numpy as np

def Q_learning(env, Q, episodes, steps, alpha, gamma, delta):
    #loop for each episode
    for eps in range (episodes):
        #initialize S
        state = env.reset()        
        #loop for each step episode
        for i in range(steps):
            #To show the environment
            #env.render()
            #Choose A form S using Q (epsilon greedy policy)
            a = eps_greedy(state, delta, Q)
            #Take action A, observe R and S
            new_state, reward, done, _  = env.step(a)
            #update rule 
            Q[state, a] = Q[state, a] + alpha *(reward + gamma * np.max(Q[new_state,:]) - Q[state, a])
            #update state
            state = new_state
            #until S is terminal
            if done:
                break            
    return Q

def eps_greedy(s, delta, Q):
    prob = np.random.rand()
    action = 0
    if prob < delta:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[s,: ])
    return action 

def sarsa(env, Q, episodes, steps, alpha, gamma, delta):
    for eps in range (episodes):
        #initialize S
        state = env.reset()
        #Choose A form S using Q (epsilon greedy policy)
        a1 = eps_greedy(state, delta, Q)        
        #loop for each step episode
        for _ in range(steps):
            #To show the environment
            #env.render()
            #Take action A, observe R and S
            new_state, reward, done, _  = env.step(a1)
            #Choose A' form S' using Q (epsilon greedy policy)
            a2 = eps_greedy(new_state, delta, Q)
            #update rule 
            Q[state, a1] = Q[state, a1] + alpha *(reward + gamma * Q[new_state,a2] - Q[state, a1])
            #update state
            state = new_state
            #update action
            a1 = a2
            #until S is terminal
            if done:
                break            
    return Q

def evolution(Q, env):
    values = [np.argmax(row) for row in Q]
    keys = list(range(len(Q)))
    policy = dict(zip(keys,values))
    env.reset()
    p=0
    while(True):
        a = policy[p]
        step = env.step(a)
        p=step[0]
        if(step[2]):
            return step[1]
    
#Main
if __name__=="__main__":
    env = gym.make('FrozenLake-v0')
    #Q learning
    Q1 = np.zeros((env.observation_space.n, env.action_space.n))
    episodes = 10000
    steps = 100
    alpha = 0.1 # da tenere basso, Learning rate
    gamma = 0.99 # se basso risultati non superano lo 0.1
    eps = 0.6
    Q1 = Q_learning(env, Q1, episodes, steps, alpha, gamma, eps)
    #print(Q1)
    

    #Sarsa
    # ottiene risultati migliori con poca esplorazione, eps = 0.2
    Q2 = np.zeros((env.observation_space.n, env.action_space.n))
    Q2 = sarsa(env, Q2, episodes, steps, alpha, gamma, 0.2)

    N=50
    print("Q-Learning \n", sum([evolution(Q1,env) for i in range(N)])/N)
    print("\n")
    print("Sarsa \n", sum([evolution(Q2,env) for i in range(N)])/N)