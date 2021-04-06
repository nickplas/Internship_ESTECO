# Internship_ESTECO

* Policy evaluation: used to solve the prediction problem. It is used to find the state-value function $v_{\pi}$ given a specific policy π.

* Policy improvement: improve the policy by using the state-value function and policy improvement theorem.

* Policy iteration: iterative algorithm in which policy evaluation and policy improvement are used in order to return the best policy and state-value function.

* Value iteration: find the optimal state-value function and extract from it the oprimal policy.

* Sarsa: On-policy learning algorithm. The action-value function is updated using an error and a learning rate. It follows a specific policy.

* Q Lerning: Off-policy learning algorithm. It is able to change the policy.

* Hedger: First approach to a continuous state and action spaces in which function approximations are used to extend the Q-Learning algorithm. Here the approximation is obtained using a locally weighted regression.

* DQN: RL algorithm for continuous state space. Extension of Q-learning in which a neural network is used to approximate the Q action-value function. If the problem has a continuous action space other algorithms need to be used (like DDPG) 

