import gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('LunarLander-v2')

model = DQN('MlpPolicy', env, 
            learning_rate=0.0005,
            verbose=0, 
            exploration_initial_eps = 0.1,
            exploration_final_eps = 0.0001, 
            target_update_interval = 250,
            seed= 11)

model.learn(total_timesteps=int(2e5))

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# model.save("dqn_lunar")
# del model 
# model = DQN.load("dqn_lunar")

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()