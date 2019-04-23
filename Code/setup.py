# DS 2 Final Project
# Micah Botkin-Levy
# 4/11/19


import time
import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import cvxpy as cp
import gym
import gym_ess

# matplotlib style
font = {'size': 20}
matplotlib.rc('font', **font)
#plt.style.use('default')

# READ IN DATA
data_folder = Path('./Data/historical')
files = os.listdir(data_folder)

loc_ID = '363'

raw_data = pd.DataFrame()

for file in files:
    date = file.split('_')[2]
    df = pd.read_csv(data_folder / file, skiprows=5)
    df = df.loc[df.H == 'D']
    df = df.loc[df['Location ID'] == loc_ID]
    df['dttm'] = date + ' ' + df['Local Time']
    df['dttm'] = pd.to_datetime(df['dttm'], format='%Y%m%d %H:%M:%S')
    df = df.set_index('dttm')
    df = df['LMP']
    df = df.astype('float')
    raw_data = pd.concat([raw_data, df])

raw_data.columns = pd.Index(['LMP'])

summ = raw_data.describe()
summ.transpose().to_latex()

fig = plt.figure()
plt.plot(raw_data)
plt.xlabel('Date')
plt.ylabel('LMP ($/MWh)')
fig.autofmt_xdate()
plt.tight_layout()


raw_data.hist(bins=50)
plt.xlabel('LMP ($/MWh)')
plt.ylabel('Frequency')
plt.tight_layout()


raw_data.plot.density(ind=np.linspace(raw_data.LMP.min()-raw_data.LMP.std(), raw_data.LMP.max()+raw_data.LMP.std(), num=1000))
plt.xlabel('LMP ($/MWh)')
plt.ylabel('Probability')
plt.tight_layout()


raw_data.hist(log=True, bins=50)
plt.xlabel('LMP ($/MWh)')
plt.ylabel('Frequency')
plt.tight_layout()


#raw_data.plot.density(logy=True, logx=True)

K = raw_data.shape[0]
ene_cap = 100
ene_init = 0.5*ene_cap
eff_ch = 0.9
eff_dis = 0.9
power_ch = 10
power_dis = 10
dt = 5/60

# Central Solution
p_ch = cp.Variable(K, name='p_ch', nonneg=True)    # EV Charging power
p_dis = cp.Variable(K, name='p_dis', nonneg=True)    # EV Discharging power
ene = cp.Variable(K+1, name='ene', nonneg=True)  # ESS Energy
constraints = []
constraints += [ene <= np.tile(ene_cap, K+1)]  # energy upper limit
constraints += [ene >= np.tile(0, K+1)]  # energy lower limit
constraints += [p_ch <= np.tile(power_ch, K)]  # power upper limit
constraints += [p_ch >= np.tile(0, K)]  # power lower limit
constraints += [p_dis <= np.tile(power_dis, K)]  # power upper limit
constraints += [p_dis >= np.tile(0, K)]  # power lower limit
constraints += [ene[K] == ene_init]  # ending
for k in range(K):
    constraints += [ene[k + 1] == ene[k] + (p_ch-p_dis) * dt]

energy_cost = - cp.sum(cp.multiply(p_ch, raw_data.LMP[0:K]) * dt * 1/eff_ch) + cp.sum(cp.multiply(p_dis, raw_data.LMP[0:K]) * dt* eff_dis)
obj = cp.Minimize(energy_cost)

# Form and solve problem using Gurobi
prob = cp.Problem(obj, constraints)
start = time.time()
prob.solve(verbose=True, solver=cp.GUROBI)
end = time.time()
print(end-start)
print("status:", prob.status)
print("optimal value", prob.value)


# Random Solution
env_random = gym.make('ess-v0', ene_cap=ene_cap, ene_init=ene_init, eff_ch=eff_ch, eff_dis=eff_dis, power_ch=power_ch, power_dis=power_dis, dt=dt)
log_random = np.zeros((K, 2)) # energy, power
actions = [-power_dis, 0, power_ch]
num_actions = len(actions)

for k in range(K-1):
    print("Time Step:", k)
    log_random[k, 0] = env_random.ene

    a_ind = np.random.randint(0, num_actions)
    a = actions[a_ind]
    # cap charge or discharge to keep ene limits
    a = max(min(a*env_random.dt, env_random.dt*(env_random.ene_cap-env_random.ene)), -env_random.ene*env_random.dt)/env_random.dt

    # Get new state and reward from environment
    env_random.step(a, raw_data.LMP[k])
    log_random[k, 1] = a



# Q Learning Solution
env = gym.make('ess-v0', ene_cap=ene_cap, ene_init=ene_init, eff_ch=eff_ch, eff_dis=eff_dis, power_ch=power_ch, power_dis=power_dis, dt=dt)

log_q = np.zeros((K, 2)) # energy, power

actions = [-power_dis, 0, power_ch]
num_actions = len(actions)
num_ene = round(ene_cap/10)
num_price = 50

ene_bins = np.linspace(0, ene_cap, num_ene).round()

price_bins = np.linspace(min(raw_data.LMP), max(raw_data.LMP), num_price).round()
price_inds = np.digitize(raw_data.LMP, price_bins)
prices = price_bins[price_inds]

Q = np.zeros([num_price*num_ene, num_actions])  # price_0/ene_0, price_1/ene_0,...

# lr = .8
# y = .95
alpha = 0.5
gamma = 0.8
epsilon = 0.5

for k in range(K-1):
    print("Time Step:", k)
    log_q[k, 0] = env.ene

    price = prices[k]
    price_ind = price_inds[k]
    ene_ind = np.digitize(env.ene, ene_bins)
    s_ind = price_ind+(ene_ind*num_price)

    if np.random.random() < 1 - epsilon:
        a_ind = np.random.randint(0, num_actions)
    else:
        a_ind = np.argmax(Q[s_ind, :])

    a = actions[a_ind]

    # cap charge or discharge to keep ene limits
    a = max(min(a*env.dt, env.dt*(env.ene_cap-env.ene)), -env.ene*env.dt)/env.dt

    # Get new state and reward from environment
    env.step(a, price)
    ene_ind_new = np.digitize(env.ene, ene_bins)
    price_ind_new = price_inds[k+1]
    s_ind_new = price_ind_new+(ene_ind_new*num_price)

    # Update Q-Table with new knowledge
    # Q[s_ind, a_ind] = Q[s_ind, a_ind] + lr * (env.reward + y * np.max(Q[s_ind_new, :]) - Q[s_ind, a_ind])
    Q[s_ind, a_ind] = (1 - alpha) * Q[s_ind, a_ind] + alpha * (env.reward + gamma * np.max(Q[s_ind_new, :]))
    log_q[k, 1] = a







# plotting

energy_plot = plt.plot(range(K), log_random[:, 0], range(K), log_q[:, 0])
power_plot = plt.plot(range(K), log_random[:, 1], range(K), log_q[:, 1])


