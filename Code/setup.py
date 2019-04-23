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
alpha = 0.5
gamma = 0.5
epsilon = 0.8
eta = 0.2
log_q, cost_q = ess.run_q_solution(K, raw_data.LMP, ene_cap, ene_init, power_ch, power_dis, eff_ch, eff_dis, dt,
                                   epsilon, alpha, gamma, eta)


#summary
print("Central Cost:", cost_central)
print("Random Cost:", cost_random)
print("Q Cost:", cost_q)

# plotting
plt.figure()
plt.plot(range(K), log_random[:, 0], 'r', label='Random')
plt.plot(range(K), log_q[:, 0], 'g', label='Q-Learning')
plt.plot(range(K), log_central[:, 0], 'b', label='Central Opt.')
plt.xlabel('Time Step (5min)')
plt.ylabel('Energy (MWh)')
plt.legend()
plt.tight_layout()


plot_power = pd.DataFrame({'Random': log_random[:, 1],
                           'Q-Learning': log_q[:, 1],
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

