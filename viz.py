# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 17:43:49 2021

@author: ahhua

Program to visualize my model
"""

import pybullet as p
import time
import pybullet_data
import numpy as np
import pandas as pd
import csv
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.env_util import make_vec_env
import gym
import gym_seb

env = gym.make('seb-v0', max_timesteps = 100000, use_gui = True)

from stable_baselines3.common.env_checker import check_env

check_env(env, warn=True)

mode = p.POSITION_CONTROL

model = PPO.load("real_model_ppo", env = env)
print("loaded")
done = False
obs = env.reset()
i = 0
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    time.sleep(1/60)
    #print(i)
    i += 1
    #print(obs)
p.disconnect()