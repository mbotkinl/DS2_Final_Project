from gym.envs.registration import register

register(
    id='ess-v0',
    entry_point='gym_ess.envs:ESSEnv',
)
