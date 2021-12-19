# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 23:01:43 2021

@author: ahhua
"""
#Simple plotting script to see some data
import matplotlib.pyplot as plt
import pandas as pd

#df = pd.read_csv("./data/timestep_rewards/ppo_rewards_v14.csv")
df = pd.read_csv("./../../../Downloads/ppo_rewards_v14.csv")

plt.plot(df.index, df['f_reward'])
plt.title('forward_reward over timestep')
plt.xlabel('timestep')
plt.ylabel('forward_reward')
plt.show()

plt.plot(df.index, df['d_reward'])
plt.title('deviation_reward over timestep')
plt.xlabel('timestep')
plt.ylabel('deviation_reward')
plt.show()

plt.plot(df.index, df['p_reward'])
plt.title('pitch_reward over timestep')
plt.xlabel('timestep')
plt.ylabel('pitch_reward')
plt.show()

plt.plot(df.index, df['y_reward'])
plt.title('yaw_reward over timestep')
plt.xlabel('timestep')
plt.ylabel('yaw_reward')
plt.show()

plt.plot(df.index, df['c_reward'])
plt.title('ctrl_cost over timestep')
plt.xlabel('timestep')
plt.ylabel('ctrl_cost')
plt.show()