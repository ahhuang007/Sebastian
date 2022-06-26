# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 23:01:43 2021

@author: ahhua
"""
#Simple plotting script to see some data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage.filters import uniform_filter1d

#Loading data, set N for moving average
df = pd.read_csv("./data/timestep_rewards/ppo_rewards_v21.csv")
#df = pd.read_csv("./../../../Downloads/ppo_rewards_v14.csv")
N = 10000

cols = ['f_reward', 'd_reward', 'p_reward', 'y_reward', 'c_reward', 'r_reward']
for col in cols:
    #Removing outliers
    avg = np.mean(df[col])
    st = np.std(df[col])
    tempdf = df[df[col] < avg + 4*st]
    tempdf = tempdf[tempdf[col] > avg - 4*st]
    tempdf.reset_index(drop = True)
    #Plotting
    plt.plot(tempdf.index, tempdf[col])
    plt.title(col + ' over timestep')
    plt.xlabel('timestep')
    plt.ylabel(col)
    #Moving average
    y = uniform_filter1d(tempdf[col], size=N)
    plt.plot(tempdf.index, y)
    plt.show()
