#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ESS Functions
Micah Botkin-Levy
4/22/19
"""

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
import matplotlib.dates as mdates

# from pandas.plotting import register_matplotlib_converters


def read_data_old(loc_ID):
    """
    Function to read in LMP data from files (no longer used)
    :param loc_ID: location id to extract data for (string
    :return: raw_data(DataFrame)
    """

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
    """
    Function to read in LMP data from files
    :param loc_ID: location id to extract data for (string)
    :return: raw_data(DataFrame)
    """
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
    """
    Function to run exploratory data analysis on LMP data
    :param raw_data: LMP data (DataFrame)
    :return: Multiple Plots
    """
    summ = raw_data.describe()
    summ.transpose().round(2).to_latex()

    fig, ax = plt.subplots()
    plt.plot(raw_data)
    plt.xlabel('Time')
    plt.ylabel('LMP ($/MWh)')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    plt.tight_layout()

    raw_data.hist(bins=50)
    plt.xlabel('LMP ($/MWh)')
    plt.ylabel('Frequency')
    plt.title('')
    plt.tight_layout()

    raw_data.plot.density(ind=np.linspace(raw_data.LMP.min()-raw_data.LMP.std(), raw_data.LMP.max()+raw_data.LMP.std(), num=1000), legend=False)
    plt.xlabel('LMP ($/MWh)')
    plt.ylabel('Probability')
    plt.tight_layout()

    raw_data.hist(log=True, bins=50)
    plt.xlabel('LMP ($/MWh)')
    plt.ylabel('Frequency')
    plt.title('')
    plt.tight_layout()

    # raw_data.plot.density(logy=True, logx=True)


def run_central_solution(K, data, ene_cap, ene_init, power_ch, power_dis, eff_ch, eff_dis, self_disch, dt):
    """
    Run Perfect Forecast Optimization Problem
    :param K: number of timesteps
    :param data: LMP data
    :param ene_cap: energy capacity (MWh)
    :param ene_init: starting energy in ESS
    :param power_ch: maximum charging power (MW)
    :param power_dis: maximum discharging power (MW)
    :param eff_ch: charging efficiency
    :param eff_dis: discharging efficiency
    :param self_disch: percent of energy lost to self-discharge per hour
    :param dt: timestep length (hours)
    :return log_central: log of energy, power, and cumulative profit
    :return prob.value: cost of solution
    """
    print("Running Central Opt Solution")

    prices_round = data.round(2)
    len_opt = K
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

    costVector = cp.multiply(p_ch, prices_round[0:len_opt]) * dt * 1 / eff_ch - cp.multiply(p_dis, prices_round[0:len_opt]) * dt * eff_dis
    obj = cp.Minimize(cp.sum(costVector))

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

    log_central = pd.DataFrame({'LMP': data.values, 'Energy': ene.value[1:K+1], 'Power': p_ch.value - p_dis.value,
                               'cumul_prof': -np.cumsum(costVector.value)}, index=data.index)

    return log_central, prob.value


def run_random_solution(K, data, ene_cap, ene_init, power_ch, power_dis, eff_ch, eff_dis, self_disch, dt):
    """
    Run Random Algorithm

    :param K: number of timesteps
    :param data: LMP data
    :param ene_cap: energy capacity (MWh)
    :param ene_init: starting energy in ESS
    :param power_ch: maximum charging power (MW)
    :param power_dis: maximum discharging power (MW)
    :param eff_ch: charging efficiency
    :param eff_dis: discharging efficiency
    :param self_disch: percent of energy lost to self-discharge per hour
    :param dt: timestep length (hours)
    :return log_random: log of energy, power, and cumulative profit
    :return env_random.total_cost: cost of solution
    """

    print("Running Random Solution")

    env_random = gym.make('ess-v0', ene_cap=ene_cap, ene_init=ene_init, eff_ch=eff_ch, eff_dis=eff_dis,
                          power_ch=power_ch, power_dis=power_dis, self_disch=self_disch, dt=dt)

    dates = data.index
    ene = np.zeros(K)
    p = np.zeros(K)
    cumul_prof = np.zeros(K)

    actions = [-power_dis, 0, power_ch]
    num_actions = len(actions)

    for k in range(K):
        #print("Time Step:", k)
        # log_random.loc[dates[k], 'Energy'] = env_random.ene
        ene[k] = env_random.ene

        a_ind = np.random.randint(0, num_actions)
        a = actions[a_ind]
        # cap charge or discharge to keep ene limits
        a = max(min(a, (env_random.ene_cap - env_random.ene)/env_random.dt), -env_random.ene / env_random.dt)

        # Get new state and reward from environment
        env_random.step(a, data[k], 0, 1)
        p[k] = a
        cumul_prof[k] = -env_random.total_cost
        # log_random.loc[dates[k], 'Power'] = a
        # log_random.loc[dates[k], 'cumul_prof'] = env_random.total_cost

    log_random = pd.DataFrame({'LMP': data.values, 'Energy': ene, 'Power': p, 'cumul_prof': cumul_prof},
                              index=dates)
    return log_random, env_random.total_cost


def run_q_solution(K, data, ene_cap, ene_init, power_ch, power_dis, eff_ch, eff_dis, self_disch, dt, epsilon, alpha,
                   gamma, eta, reward_mode, avg_price_init):
    """

    :param K: number of timesteps
    :param data: LMP data
    :param ene_cap: energy capacity (MWh)
    :param ene_init: starting energy in ESS
    :param power_ch: maximum charging power (MW)
    :param power_dis: maximum discharging power (MW)
    :param eff_ch: charging efficiency
    :param eff_dis: discharging efficiency
    :param self_disch: percent of energy lost to self-discharge per hour
    :param dt: timestep length (hours)
    :param epsilon: epsilon greedy parameter
    :param alpha: Q update weight
    :param gamma: Q update weight
    :param eta: Average Price Weight
    :param reward_mode: Reward Mode
    :param avg_price_init: Initial Avarage Price
    :return log_q: log of energy, power, and cumulative profit
    :return env.total_cost: cost of solution
    """


    print("Running Q Solution")
    env = gym.make('ess-v0', ene_cap=ene_cap, ene_init=ene_init, eff_ch=eff_ch, eff_dis=eff_dis, power_ch=power_ch,
                   power_dis=power_dis,  self_disch=self_disch, dt=dt)

    dates = data.index
    ene = np.zeros(K)
    p = np.zeros(K)
    cumul_prof = np.zeros(K)

    actions = [-power_dis, 0, power_ch]
    num_actions = len(actions)
    num_ene = round(ene_cap/1)
    num_price = 50

    ene_bins = np.linspace(0, ene_cap, num_ene).round()

    price_bins = np.quantile(data,np.linspace(0, 1, num_price))
    # price_bins = np.linspace(min(data), max(data), num_price).round()
    price_inds = np.digitize(data, price_bins)
    prices = price_bins[price_inds-1]

    avg_price = avg_price_init


    Q = np.random.rand(num_price*num_ene, num_actions)/100  # price_0/ene_0, price_1/ene_0,...

    for k in range(K):
        # print("Time Step:", k)
        ene[k] = env.ene

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
        a = max(min(a, (env.ene_cap-env.ene)/env.dt), -env.ene/env.dt)

        # Get new state and reward from environment
        env.step(a, price, avg_price, reward_mode)
        ene_ind_new = np.digitize(env.ene, ene_bins)
        price_ind_new = price_inds[min(k+1, K-1)]
        s_ind_new = price_ind_new+((ene_ind_new-1)*num_price)

        # Update Q-Table with new knowledge
        # Q[s_ind, a_ind] = Q[s_ind, a_ind] + lr * (env.reward + y * np.max(Q[s_ind_new, :]) - Q[s_ind, a_ind])
        Q[s_ind, a_ind] = (1 - alpha) * Q[s_ind, a_ind] + alpha * (env.reward + gamma * np.max(Q[s_ind_new, :]))
        p[k] = a
        cumul_prof[k] = -env.total_cost

    log_q = pd.DataFrame({'LMP': data.values, 'Energy': ene, 'Power': p, 'cumul_prof': cumul_prof}, index=dates)
    return log_q, env.total_cost


def plot_results(data, log_random, log_q1, log_q2, log_central):
    """
    Plot results of simulations
    :param data: LMP data
    :param log_random:  Random Log
    :param log_q1: Q1 Log
    :param log_q2: Q2 Log
    :param log_central: Central Log
    :return:
    """
    # total energy plot
    fig, ax1 = plt.subplots()
    ax1.plot(log_random.loc[:, 'Energy'], 'r', label='Random', drawstyle='steps')
    ax1.plot(log_q1.loc[:, 'Energy'], 'g', label='Q-Learning 1', drawstyle='steps')
    ax1.plot(log_q2.loc[:, 'Energy'], 'm', label='Q-Learning 2', drawstyle='steps')
    ax1.plot(log_central.loc[:, 'Energy'], 'b', label='Central Opt.', drawstyle='steps')
    ax2 = ax1.twinx()
    ax2.plot(data[:], 'k', label='LMP', drawstyle='steps')
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Energy (MWh)')
    ax2.set_ylabel('Prices ($/MWh)')
    ax1.legend()
    plt.tight_layout()

    # total power plot
    # fig, ax1 = plt.subplots()
    # ax1.plot(log_random.loc[:, 'Power'], 'r', label='Random')
    # ax1.plot(log_q1.loc[:, 'Power'], 'g', label='Q-Learning 1')
    # ax1.plot(log_q2.loc[:, 'Power'], 'm', label='Q-Learning 2')
    # ax1.plot(log_central.loc[:, 'Power'], 'b', label='Central Opt.')
    # ax2 = ax1.twinx()
    # ax2.plot(data[:], 'k', label='LMP')
    # ax1.xaxis.set_major_locator(mdates.MonthLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    # ax1.set_xlabel('Time')
    # ax1.set_ylabel('Power (MW)')
    # ax2.set_ylabel('Prices ($/MWh)')
    # ax1.legend()
    # plt.tight_layout()

    # combine
    font = {'size': 20}
    matplotlib.rc('font', **font)
    randind = np.random.randint(5000, len(data))
    date_ind = data.index[randind:randind+288]
    fig, axarr = plt.subplots(3, 1, sharex=True)
    axarr[0].plot(log_random.loc[date_ind, 'Energy'], 'r', label='Random', drawstyle='steps')
    axarr[0].plot(log_q1.loc[date_ind, 'Energy'], 'g', label='Q-Learning 1', drawstyle='steps')
    axarr[0].plot(log_q2.loc[date_ind, 'Energy'], 'm', label='Q-Learning 2', drawstyle='steps')
    axarr[0].plot(log_central.loc[date_ind, 'Energy'], 'b', label='Central Opt.', drawstyle='steps')
    axarr[0].set_ylabel('Energy (MWh)')
    axarr[1].plot(log_random.loc[date_ind, 'Power'], 'r', label='Random', drawstyle='steps')
    axarr[1].plot(log_q1.loc[date_ind, 'Power'], 'g', label='Q-Learning 1', drawstyle='steps')
    axarr[1].plot(log_q2.loc[date_ind, 'Power'], 'm', label='Q-Learning 2', drawstyle='steps')
    axarr[1].plot(log_central.loc[date_ind, 'Power'], 'b', label='Central Opt.', drawstyle='steps')
    axarr[1].set_ylabel('Power (MW)')
    axarr[1].legend(loc='right')
    axarr[2].plot(data[date_ind], 'k', label='LMP')
    axarr[2].set_ylabel('Prices ($/MWh)')
    axarr[2].set_xlabel('Time')
    axarr[2].legend()
    fig.autofmt_xdate()
    plt.tight_layout()

    # cumulative profits
    fig, ax1 = plt.subplots()
    ax1.plot(log_random.loc[:, 'cumul_prof'], 'r', label='Random')
    ax1.plot(log_q1.loc[:, 'cumul_prof'], 'g', label='Q-Learning 1')
    ax1.plot(log_q2.loc[:, 'cumul_prof'], 'm', label='Q-Learning 2')
    ax1.plot(log_central.loc[:, 'cumul_prof'], 'b', label='Central Opt.')
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cumulative Profit ($)')
    ax1.legend()
    plt.grid(axis='y')
    plt.tight_layout()


    plt.figure()
    # plot_power.hist(subplots=False)
    plt.hist([log_random.loc[:, 'Power'], log_q1.loc[:, 'Power'], log_q2.loc[:, 'Power'], log_central.loc[:, 'Power']], bins=40,
             label=['Random', 'Q-Learning 1', 'Q-Learning 2', 'Central Opt.'])
    plt.hist([log_random.loc[:, 'Power'], log_q1.loc[:, 'Power'], log_q2.loc[:, 'Power'], log_central.loc[:, 'Power']], bins=40,
             label=['Random', 'Q-Learning 1', 'Q-Learning 2', 'Central Opt.'], log=True)
    # plt.hist(log_random[:, 1], bins=40, alpha=0.5, color='r', label='Random')
    # plt.hist(log_q[:, 1], bins=40, alpha=0.5, color='g', label='Q-Learning')
    # plt.hist(log_central[:, 1], bins=40, alpha=0.5, color='b', label='Central Opt.')
    plt.xlabel('Power (MW)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()

