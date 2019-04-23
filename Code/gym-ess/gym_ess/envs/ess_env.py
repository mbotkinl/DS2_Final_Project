import gym
from gym import error, spaces, utils
from gym.utils import seeding


class ESSEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, ene_cap, ene_init, eff_ch, eff_dis, power_ch, power_dis, dt):
        self.ene_cap = ene_cap
        self.eff_ch = eff_ch
        self.eff_dis = eff_dis
        self.power_ch = power_ch
        self.power_dis = power_dis
        self.dt = dt

        self.ene = ene_init
        self.time = 0
        self.reward = 0
        self.total_reward = 0

    def step(self, power, price):
        self.ene = self.ene + self.dt*power
        self.time = self.time+self.dt
        self.reward = -self.dt*price*(self.eff_dis*min(0, power)+1/self.eff_ch*max(power, 0))
        self.total_reward = self.total_reward + self.reward

        #return self.ene

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass