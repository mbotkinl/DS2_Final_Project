# DS 2 Final Project
# Micah Botkin-Levy
# 4/11/19

# import packages
import pandas as pd
import ess_functions as ess
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# matplotlib style
font = {'size': 35}
matplotlib.rc('font', **font)

# READ IN DATA
loc_ID = '363'
raw_data = ess.read_data(loc_ID)
# ess.run_data_description()
# raw_data.LMP = np.maximum(raw_data.LMP, 0)

# problem parameters
num_runs = 40  # number of simulations to average over
K = raw_data.shape[0]  # number of timesteps
ene_cap = 20  # MWh
ene_init = 0.5*ene_cap  # starting energy in ESS
eff_ch = 0.9  # charging efficiency
eff_dis = 0.9  # discharging efficiency
power_ch = 5  # maximum charging power (MW)
power_dis = 5  # maximum discharging power (MW)
dt = 5/60  # timestep length (hours)
self_disch = 0.1  # percent of energy lost to self-discharge per hour

# Central Solution
log_central, cost_central = ess.run_central_solution(K, raw_data.LMP, ene_cap, ene_init, power_ch, power_dis,
                                                     eff_ch, eff_dis, self_disch, dt)
print("Central Cost:", cost_central)

# Random Solution
cost_random_avg = 0
log_random = None
for n in range(num_runs):
    log_random, cost_random = ess.run_random_solution(K, raw_data.LMP, ene_cap, ene_init, power_ch, power_dis,
                                                      eff_ch, eff_dis, self_disch, dt)
    cost_random_avg += cost_random

cost_random_avg = cost_random_avg/num_runs
print("Random Cost:", cost_random_avg)

# Q Learning Solution 1
reward_mode = 1
alpha = 0.4
gamma = 0.2
epsilon = 0.8
eta = 0.2
cost_q1_avg = 0
avg_price_init = 40  # starting average energy price for Q learning
log_q1 = None
for n in range(num_runs):
    log_q1, cost_q1 = ess.run_q_solution(K, raw_data.LMP, ene_cap, ene_init, power_ch, power_dis, eff_ch, eff_dis,
                                         self_disch, dt, epsilon, alpha, gamma, eta, reward_mode, avg_price_init)
    cost_q1_avg += cost_q1

cost_q1_avg = cost_q1_avg/num_runs
print("Q1 Cost:", cost_q1_avg)


# Q Learning Solution 2
reward_mode = 2
cost_q2_avg = 0
log_q2 = None
for n in range(num_runs):
    log_q2, cost_q2 = ess.run_q_solution(K, raw_data.LMP, ene_cap, ene_init, power_ch, power_dis, eff_ch, eff_dis, self_disch,
                                         dt, epsilon, alpha, gamma, eta, reward_mode, avg_price_init)
    cost_q2_avg += cost_q2

cost_q2_avg = cost_q2_avg/num_runs
print("Q2 Cost:", cost_q2_avg)

# summary
print("Central Cost:", cost_central)
print("Random Cost:", cost_random_avg)
print("Q1 Cost:", cost_q1_avg)
print("Q2 Cost:", cost_q2_avg)

# plotting
ess.plot_results(raw_data.LMP, log_random, log_q1, log_q2, log_central)
