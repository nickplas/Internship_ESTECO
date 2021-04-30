import numpy as np


def eps_greedy(env, s, delta, Q):
    prob = np.random.rand()
    if prob < delta:
        action = env.sample()
    else:
        action = np.argmax(Q[s])
    return action


def sarsa(env, Q, episodes, steps, alpha, gamma, delta):
    for eps in range(episodes):
        # initialize S
        state = env.reset()
        # Choose A form S using Q (epsilon greedy policy)
        a1 = eps_greedy(env, state, delta, Q)
        # loop for each step episode
        for _ in range(steps):
            # To show the environment
            # env.render()
            # Take action A, observe R and S
            new_state, reward, done, _ = env.step(a1)
            if done:
                break
                # Choose A' form S' using Q (epsilon greedy policy)
            a2 = eps_greedy(env, new_state, delta, Q)
            # update rule
            Q[state][a1] = Q[state][a1] + alpha * (reward + gamma * Q[new_state][a2] - Q[state][a1])
            # update state
            state = new_state
            # update action
            a1 = a2
            # until S is terminal
    return Q