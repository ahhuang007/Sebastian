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

env = gym.make('seb-v0', episode_timesteps = 10000, use_gui = True)

from stable_baselines3.common.env_checker import check_env

check_env(env, warn=True)

mode = p.POSITION_CONTROL

model = PPO.load("random_model_ppo_action", env = env)
print("loaded")
done = False
obs = env.reset()
i = 0
from stable_baselines3.common.evaluation import evaluate_policy

#performance with random model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)

print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
'''
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    time.sleep(1/60)
    #print(i)
    i += 1
    #print(obs)
'''
env.close()