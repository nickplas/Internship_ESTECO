import numpy as np
from scipy.spatial import Delaunay
from math import exp

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
            a = eps_greedy(env, state, delta, Q)
            #Take action A, observe R and S
            new_state, reward, done, _  = env.step(a)
            #until S is terminal
            if done:
                break 
            #update rule 
            Q[state][a] = Q[state][a] + alpha *(reward + gamma * np.max(Q[new_state]) - Q[state][a])
            #update state
            state = new_state
                       
    return Q

def eps_greedy(env, s, delta, Q):
    prob = np.random.rand()
    if prob < delta:
        action = env.sample()
    else:
        action = np.argmax(Q[s])
    return action 

def sarsa(env, Q, episodes, steps, alpha, gamma, delta):
    for eps in range (episodes):
        #initialize S
        state = env.reset()
        #Choose A form S using Q (epsilon greedy policy)
        a1 = eps_greedy(env, state, delta, Q)        
        #loop for each step episode
        for _ in range(steps):
            #To show the environment
            #env.render()
            #Take action A, observe R and S
            new_state, reward, done, _  = env.step(a1)
            if done:
                break 
            #Choose A' form S' using Q (epsilon greedy policy)
            a2 = eps_greedy(env, new_state, delta, Q)
            #update rule 
            Q[state][a1] = Q[state][a1] + alpha *(reward + gamma * Q[new_state][a2] - Q[state][a1])
            #update state
            state = new_state
            #update action
            a1 = a2
            #until S is terminal                       
    return Q

def create_training_data(episodes, steps, env):
    experience = []
    for _ in range(episodes):
        env.reset()
        for i in range(steps):
            current = [env.state]
            action = env.sample()
            current.append(action)
            obs, reward, done, info = env.step(action)
            current.append(obs)
            current.append(reward)
            experience.append(current)
    return np.array(experience)

def closest_points(point, points, k):
    K = []
    for i in range(len(points)):      
        point2 = points[i][:2]
        dist = np.linalg.norm(point - point2)
        if dist < k:
            k.append(point2)
    return K

def in_hull(p, hull):
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def LWR(weights):

    return 0

def hedger_prediction(state, action, h, k_threshold, e):
    # INPUT: (s, a), bandwidth h

    #concatenate s and a in q
    q = np.array(state + action)
    # find set of K points near q (use k threshold)
    K = closest_points(q, e, k_threshold)
    # if cardinality K < k_t
    if len(K) < k_threshold:
    #   return don't know
        return 0
    # else
    else:
    #   calculate IVH H
    #   if q in H:
        if in_hull(q, np.array(K)):
    #       calculate kernel weight: exp(-(q - k_i)^2 / h^2)
            weights = np.zeros((len(K), len(q)))
            i = 0
            for k in K:
                weights[i] = exp(-(q-k)^2/h^2)
                i += 1
    #       do regression on K using weights

    #       return fitted function in (s, a)
            return 
    #   else:
        else:
    #       return don't know
            return 0

def hedger_training(e, alpha, gamma, h, state, action, k_threshold):
    # INPUT: Experience (s, a, s', r)
    #        Learning rate
    #        Discount factor
    #        Bandwidth h
    # q <- Qpredict (use hedger training)
    q = hedger_prediction(state, action, h, k_threshold, e)
    # qnext <- max Qpredict (s', a')   [max in a']
    qnext = 
    # K <- set used in calculation of q
    K = closest_points(q, e, k_threshold)
    # compute weight exp(-(q - k_i)^2 / h^2) = ki_i
    weight = exp(-(q-k)^2/h^2)
    # qnew <- q + alpha(r + gamma*qnext -q)
    qnext = q + alpha(r + gamma*qnext - q)
    # Learn Q(s, a) = qnew
    
    # for each (s_i, a_i) in K do
    for ():
    #   Q(s_i, a_i) <- Q(s_i, a_i) + ki_i(qnext - Q(s_i, a_i))


    return 0