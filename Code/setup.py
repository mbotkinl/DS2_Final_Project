
import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


# matplotlib style
font = {'size': 20}
matplotlib.rc('font', **font)
#plt.style.use('default')

# READ IN DATA
data_folder = Path('../Data/historical')
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
