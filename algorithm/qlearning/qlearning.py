import numpy as np


def qlearning(env, Q, episodes, steps, alpha, gamma, delta):  # Q da generare dentro
    # loop for each episode
    for eps in range(episodes):
        # initialize S
        state = env.reset()
        # loop for each step episode
        for i in range(steps):
            # To show the environment
            # env.render()
            # Choose A form S using Q (epsilon greedy policy)
            a = eps_greedy(env, state, delta, Q)
            # Take action A, observe R and S
            new_state, reward, done, _ = env.step(a)
            # until S is terminal
            if done:
                break
                # update rule
            Q[state][a] = Q[state][a] + alpha * (reward + gamma * np.max(Q[new_state]) - Q[state][a])
            # update state
            state = new_state

    return Q


def eps_greedy(env, s, delta, Q):
    prob = np.random.rand()
    if prob < delta:
        action = env.sample()
    else:
        action = np.argmax(Q[s])
    return action