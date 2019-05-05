import gym

class ESSEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, ene_cap, ene_init, eff_ch, eff_dis, power_ch, power_dis,  self_disch, dt):
        self.ene_cap = ene_cap
        self.eff_ch = eff_ch
        self.eff_dis = eff_dis
        self.power_ch = power_ch
        self.power_dis = power_dis
        self.self_disch = self_disch
        self.dt = dt

        self.ene = ene_init
        self.time = 0
        self.reward = 0
        self.cost = 0
        self.total_reward = 0
        self.total_cost = 0

    def step(self, power, price, avg_price, reward_mode):
        self.ene = (self.ene + self.dt*power - self.self_disch/100*self.ene).__round__(4)
        self.time = self.time+self.dt
        self.cost = self.dt*price*(self.eff_dis*min(0, power)+1/self.eff_ch*max(power, 0))
        self.total_cost = self.total_cost + self.cost
        if reward_mode == 1:
            self.reward = -self.cost
        elif reward_mode == 2:
            self.reward = -self.dt*(price-avg_price)*self.eff_dis*min(0, power)+self.dt*(avg_price-price)*1/self.eff_ch*max(power, 0)
        self.total_reward = self.total_reward + self.reward

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass