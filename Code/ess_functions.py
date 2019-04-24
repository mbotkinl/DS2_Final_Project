import cvxpy as cp
import numpy as np
import pandas as pd
from pathlib import Path
import os
import gym
import gym_ess
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def read_data_old(loc_ID):
    print("Reading in Data")
    data_folder = Path('./Data/historical')
    files = os.listdir(data_folder)

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

    raw_data.columns = pd.Index(['LMP'])  # $/MWh
    return raw_data


def read_data(loc_ID):
    print("Reading in Data")
    data_folder = Path('./Data/', loc_ID)
    files = os.listdir(data_folder)

    raw_data = pd.DataFrame()
    for file in files:
        df = pd.read_csv(data_folder / file, skiprows=3)
        df = df.loc[df.H == 'D']
        df = df.loc[df['Location ID'] == loc_ID]
        df['dttm'] = pd.to_datetime(df['Local Time'], format='%Y-%m-%d %H:%M:%S')
        df = df.set_index('dttm')
        df = df['LMP']
        df = df.astype('float')
        raw_data = pd.concat([raw_data, df])

    raw_data.columns = pd.Index(['LMP'])  # $/MWh
    return raw_data


def run_data_description(raw_data):
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

    # raw_data.plot.density(logy=True, logx=True)


def run_central_solution(K, data, ene_cap, ene_init, power_ch, power_dis, eff_ch, eff_dis, self_disch, dt):
    print("Running Central Opt Solution")

    prices_round = data.round(2)
    len_opt = K-1
    p_ch = cp.Variable(len_opt, name='p_ch', nonneg=True)  # EV Charging power
    p_dis = cp.Variable(len_opt, name='p_dis', nonneg=True)  # EV Discharging power
    ene = cp.Variable(len_opt + 1, name='ene', nonneg=True)  # ESS Energy
    constraints = []
    constraints += [ene <= np.tile(ene_cap, len_opt + 1)]  # energy upper limit
    constraints += [ene >= np.tile(0, len_opt + 1)]  # energy lower limit
    constraints += [p_ch <= np.tile(power_ch, len_opt)]  # power upper limit
    constraints += [p_ch >= np.tile(0, len_opt)]  # power lower limit
    constraints += [p_dis <= np.tile(power_dis, len_opt)]  # power upper limit
    constraints += [p_dis >= np.tile(0, len_opt)]  # power lower limit
    constraints += [ene[len_opt] == ene_init]  # ending
    constraints += [ene[0] == ene_init]  # ending
    for k in range(len_opt):
        constraints += [ene[k + 1] == ene[k] + (p_ch[k] - p_dis[k]) * dt - self_disch/100*ene[k]]

    energy_cost = cp.sum(cp.multiply(p_ch, prices_round[0:len_opt]) * dt * 1 / eff_ch) - cp.sum(
        cp.multiply(p_dis, prices_round[0:len_opt]) * dt * eff_dis)
    obj = cp.Minimize(energy_cost)

    # Form and solve problem using Gurobi
    prob = cp.Problem(obj, constraints)
    # start = time.time()
    prob.solve(verbose=False, solver=cp.GUROBI)
    # end = time.time()
    # print(end - start)
    #print("status:", prob.status)
    #print("optimal value", prob.value)

    # ensure ESS is not charging and discharging at the same time
    assert all(p_ch.value * p_dis.value == 0)
    assert prob.status == cp.OPTIMAL

    log_central = np.zeros((K, 2))
    log_central[:, 0] = ene.value
    log_central[0:K-1, 1] = p_ch.value - p_dis.value

    return log_central, prob.value


def run_random_solution(K, data, ene_cap, ene_init, power_ch, power_dis, eff_ch, eff_dis, self_disch, dt):
    print("Running Random Solution")

    env_random = gym.make('ess-v0', ene_cap=ene_cap, ene_init=ene_init, eff_ch=eff_ch, eff_dis=eff_dis,
                          power_ch=power_ch, power_dis=power_dis, self_disch=self_disch, dt=dt)
    log_random = np.zeros((K, 2))  # energy, power
    actions = [-power_dis, 0, power_ch]
    num_actions = len(actions)

    for k in range(K - 1):
        #print("Time Step:", k)
        log_random[k, 0] = env_random.ene

        a_ind = np.random.randint(0, num_actions)
        a = actions[a_ind]
        # cap charge or discharge to keep ene limits
        a = max(min(a * env_random.dt, env_random.dt * (env_random.ene_cap - env_random.ene)),
                -env_random.ene * env_random.dt) / env_random.dt

        # Get new state and reward from environment
        env_random.step(a, data[k], 0)
        log_random[k, 1] = a

    return log_random, env_random.total_cost


def run_q_solution(K, data, ene_cap, ene_init, power_ch, power_dis, eff_ch, eff_dis, self_disch, dt, epsilon, alpha, gamma, eta):
    print("Running Q Solution")
    env = gym.make('ess-v0', ene_cap=ene_cap, ene_init=ene_init, eff_ch=eff_ch, eff_dis=eff_dis, power_ch=power_ch,
                   power_dis=power_dis,  self_disch=self_disch, dt=dt)

    log_q = np.zeros((K, 2))  # energy, power

    actions = [-power_dis, 0, power_ch]
    num_actions = len(actions)
    num_ene = round(ene_cap/1)
    num_price = 50

    ene_bins = np.linspace(0, ene_cap, num_ene).round()


    price_bins = np.quantile(data,np.linspace(0, 1, num_price))
    # price_bins = np.linspace(min(data), max(data), num_price).round()
    price_inds = np.digitize(data, price_bins)
    prices = price_bins[price_inds-1]
    avg_price = prices[0]

    Q = np.random.rand(num_price*num_ene, num_actions)/100  # price_0/ene_0, price_1/ene_0,...

    for k in range(K-1):
        #print("Time Step:", k)
        log_q[k, 0] = env.ene

        price = prices[k]
        avg_price = (1-eta)*avg_price + eta*price
        price_ind = price_inds[k]
        ene_ind = np.digitize(env.ene, ene_bins)
        s_ind = price_ind+((ene_ind-1)*num_price)

        if np.random.random() < 1 - epsilon:
            a_ind = np.random.randint(0, num_actions)
        else:
            a_ind = np.argmax(Q[s_ind, :])

        a = actions[a_ind]

        # cap charge or discharge to keep ene limits
        a = max(min(a*env.dt, env.dt*(env.ene_cap-env.ene)), -env.ene*env.dt)/env.dt

        # Get new state and reward from environment
        env.step(a, price, avg_price)
        ene_ind_new = np.digitize(env.ene, ene_bins)
        price_ind_new = price_inds[k+1]
        s_ind_new = price_ind_new+((ene_ind_new-1)*num_price)

        # Update Q-Table with new knowledge
        # Q[s_ind, a_ind] = Q[s_ind, a_ind] + lr * (env.reward + y * np.max(Q[s_ind_new, :]) - Q[s_ind, a_ind])
        Q[s_ind, a_ind] = (1 - alpha) * Q[s_ind, a_ind] + alpha * (env.reward + gamma * np.max(Q[s_ind_new, :]))
        log_q[k, 1] = a
    return log_q, env.total_cost
