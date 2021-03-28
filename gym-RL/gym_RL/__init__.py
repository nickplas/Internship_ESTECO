from gym.envs.registration import register

register(
    id='RL-v0',
    entry_point='gym_env.envs:RLEnv',
)
register(
    id='RL-extrahard-v0',
    entry_point='gym_env.envs:RLExtraHardEnv',
)