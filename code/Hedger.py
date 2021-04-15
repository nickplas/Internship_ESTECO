import numpy as np
from scipy.spatial import Delaunay
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from math import inf


def closest_points(point, points, k):
    K = []
    for i in range(len(points)):
        point2 = [points[i][0][0], points[i][1][0]]  # points[i][0] + points[i][1]
        dist = np.linalg.norm(np.array(point) - np.array(point2))
        if dist < k:
            K.append(point2)
    return K


def in_hull(p, hull):
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def LWR(X, Q, weights, point):
    y = np.zeros((len(X), len(point)))
    for i in range(len(X)-1):
        y[i] = Q[X[i][0]][X[i][1]]
    regr = LinearRegression()  # need to use another function
    regr.fit(X, y, sample_weight=weights)
    return regr.predict(point)


def find_max_q(env, alpha, gamma, h, k_threshold, Q, e,):
    qmax = [-inf, -inf]
    actions = np.linspace(env.min_action, env.max_action, 1000)
    for a in actions:
        qnew = hedger_prediction(env.state, a, h, k_threshold, env, alpha, gamma, Q, e)
        if np.all(qmax < qnew):
            qmax = qnew
    return qmax


def hedger_prediction(state, action, h, k_threshold, env, alpha, gamma, Q, e):
    q = np.array(state + action)
    K = closest_points(q, e, k_threshold)  # state + action every point
    if len(K) < k_threshold:
        a = env.action_space.sample()
        qnext = np.array(state + a)
        return q + alpha*(gamma * qnext - q)
    else:
        if in_hull(q, K):
            weights = np.zeros((len(K), len(q)))
            i = 0
            for k in K:
                weights[i] = np.exp(-(q-k)**2/h**2)
                i += 1
            return LWR(K, Q, weights, q)  # y = value of Q(s, a)
        else:
            a = env.action_space.sample()
            qnext = np.array(state + a)
            return q + alpha * (gamma * qnext - q)


def hedger_training(alpha, gamma, h, k_threshold, env, Q, e):
    #  state, action, new_state, reward = e
    q = hedger_prediction(e[-1][0], e[-1][1], h, k_threshold, env, alpha, gamma, Q, e)
    qnext = find_max_q(env, alpha, gamma, h, k_threshold, Q, e)
    k_set = closest_points(q, e, k_threshold)
    weight = np.zeros((len(k_set), len(q)))
    i = 0
    for k_i in k_set:
        weight[i] = np.exp(-(q-k_i)**2/h**2)
        i += 1
    qnew = q + alpha*(e[-1][3] + gamma*qnext - q)
    Q[e[-1][0][0]][e[-1][1][0]] = qnew
    j = 0
    for i in k_set:
        s, a = i
        Q[s][a] = Q[s][a] + weight[j]*(qnew - Q[s][a])
        j += 1


def hedger(env, episodes, steps, h, k_threshold, gamma, alpha):  # episodes = 2500, steps = 200, gamma = alpha = 0.8
    Q = defaultdict(dict)
    e = []
    for _ in range(episodes):
        env.reset()
        for i in range(steps):
            state = env.state
            a = env.action_space.sample()
            obs, r, done, info = env.step(a)
            e.append([state, a, r, obs])
            hedger_training(alpha, gamma, h, k_threshold, env, Q, e)
    return Q