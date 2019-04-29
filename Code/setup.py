# DS 2 Final Project
# Micah Botkin-Levy
# 4/11/19
import pandas as pd
import ess_functions as ess
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


# matplotlib style
font = {'size': 30}
matplotlib.rc('font', **font)
#plt.style.use('default')

# READ IN DATA
loc_ID = '363'
raw_data = ess.read_data(loc_ID)
#ess.run_data_description()


# problem parameters
num_runs = 20
K = raw_data.shape[0]
ene_cap = 10  # MWh
ene_init = 0.5*ene_cap
eff_ch = 0.9
eff_dis = 0.9
power_ch = 5  # MW
power_dis = 5  # MW
dt = 5/60
self_disch = 0.1
avg_price_init = 40

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
plot_ind = range(K)
plot_ind = range(10000, 10288)
plot_dates = raw_data.index[plot_ind]


fig, ax1 = plt.subplots()
ax1.plot(log_random.loc[:, 'Energy'], 'r', label='Random')
ax1.plot(log_q1.loc[:, 'Energy'], 'g', label='Q-Learning 1')
ax1.plot(log_q2.loc[:, 'Energy'], 'm', label='Q-Learning 2')
ax1.plot(log_central.loc[:, 'Energy'], 'b', label='Central Opt.')
ax2 = ax1.twinx()
ax2.plot(raw_data.LMP, 'k', label='LMP')
fig.autofmt_xdate()
ax1.set_xlabel('Time Step (5min)')
ax1.set_ylabel('Energy (MWh)')
ax2.set_ylabel('Prices ($/MWh)')
ax1.legend()
plt.tight_layout()


fig, ax1 = plt.subplots()
ax1.plot(log_random.loc[:, 'Power'], 'r', label='Random')
ax1.plot(log_q1.loc[:, 'Power'], 'g', label='Q-Learning 1')
ax1.plot(log_q2.loc[:, 'Power'], 'm', label='Q-Learning 2')
ax1.plot(log_central.loc[:, 'Power'], 'b', label='Central Opt.')
ax2 = ax1.twinx()
ax2.plot(raw_data.LMP, 'k', label='LMP')
fig.autofmt_xdate()
ax1.set_xlabel('Time Step (5min)')
ax1.set_ylabel('Power (MW)')
ax2.set_ylabel('Prices ($/MWh)')
ax1.legend()
plt.tight_layout()


# cumulative profits
fig, ax1 = plt.subplots()
ax1.plot(log_random.loc[:, 'cumul_prof'], 'r', label='Random')
ax1.plot(log_q1.loc[:, 'cumul_prof'], 'g', label='Q-Learning 1')
ax1.plot(log_q2.loc[:, 'cumul_prof'], 'm', label='Q-Learning 2')
ax1.plot(log_central.loc[:, 'cumul_prof'], 'b', label='Central Opt.')
fig.autofmt_xdate()
ax1.set_xlabel('Time Step (5min)')
ax1.set_ylabel('Cumulative Profit ($)')
ax1.legend()
plt.grid(axis='y')
plt.tight_layout()


plot_power = pd.DataFrame({'Random': log_random[:, 1],
                           'Q1': log_q1[:, 1],
                           'Q2': log_q2[:, 1],
                           'Central Opt.': log_central[:, 1]})


plt.figure()
# plot_power.hist(subplots=False)
plt.hist([log_random[:, 1], log_q[:, 1], log_central[:, 1]], bins=40, label=['Random', 'Q-Learning', 'Central Opt.'])
plt.hist([log_random[:, 1], log_q[:, 1], log_central[:, 1]], bins=40, label=['Random', 'Q-Learning', 'Central Opt.'], log=True)
# plt.hist(log_random[:, 1], bins=40, alpha=0.5, color='r', label='Random')
# plt.hist(log_q[:, 1], bins=40, alpha=0.5, color='g', label='Q-Learning')
# plt.hist(log_central[:, 1], bins=40, alpha=0.5, color='b', label='Central Opt.')
plt.xlabel('Power (MW)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()

