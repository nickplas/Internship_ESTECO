from gym.envs.registration import register

register(
    id='Continuous-v0',
    entry_point='gym_Continuous.envs:ContinuousEnv',
)
register(
    id='Continuous-extrahard-v0',
    entry_point='gym_Continuous.envs:ContinuousExtraHardEnv',
)