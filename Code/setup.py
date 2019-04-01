import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib
matplotlib.use('Qt5Agg')


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
    raw_data = pd.concat([raw_data, df])

raw_data.columns = pd.Index(['LMP'])




