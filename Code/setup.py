# DS 2 Final Project
# Micah Botkin-Levy
# 4/11/19
import numpy as np
import pandas as pd
import ess_functions as ess
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


# matplotlib style
font = {'size': 20}
matplotlib.rc('font', **font)
#plt.style.use('default')

# READ IN DATA
loc_ID = '363'
raw_data = ess.read_data(loc_ID)
#ess.run_data_description()


# problem parameters
K = raw_data.shape[0]
ene_cap = 10  # MWh
ene_init = 0.5*ene_cap
eff_ch = 0.9
eff_dis = 0.9
power_ch = 5  # MW
power_dis = 5  # MW
dt = 5/60

# Central Solution
log_central, cost_central = ess.run_central_solution(K, raw_data.LMP, ene_cap, ene_init, power_ch, power_dis, eff_ch, eff_dis, dt)

# Random Solution
log_random, cost_random = ess.run_random_solution(K, raw_data.LMP, ene_cap, ene_init, power_ch, power_dis, eff_ch, eff_dis, dt)

# Q Learning Solution
log_q, cost_q = ess.run_q_solution(K, raw_data.LMP, ene_cap, ene_init, power_ch, power_dis, eff_ch, eff_dis, dt)


#summary
print("Central Cost:", cost_central)
print("Random Cost:", cost_random)
print("Q Cost:", cost_q)

# plotting

plt.figure()
plt.plot(range(K), log_random[:, 0], 'r')
plt.plot(range(K), log_q[:, 0], 'g')
plt.plot(range(K), log_central[:, 0], 'b')

plt.figure()
plt.plot(range(K), log_random[:, 1], 'r')
plt.plot(range(K), log_q[:, 1], 'g')
plt.plot(range(K), log_central[:, 1], 'b')

plt.figure()
plt.hist(log_random[:, 1], bins=40, color='r')
plt.hist(log_q[:, 1], bins=40, color='g')
plt.hist(log_central[:, 1], bins=40, color='b')


