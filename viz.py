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
from stable_baselines3.common.evaluation import evaluate_policy
import random

env = gym.make('seb-v0', episode_timesteps = 10000, use_gui = True)

from stable_baselines3.common.env_checker import check_env

check_env(env, warn=True)

mode = p.POSITION_CONTROL

model = PPO.load("models/real_model_ppo_v12", env = env)
env.seed(4)
model.set_random_seed(4)
env.action_space.seed(4)
env.observation_space.seed(4)

print("loaded")
done = False
obs = env.reset()

#performance
testing_rewards = []
for j in range(10):
    reward = 0
    while not done:
        action, _states = model.predict(obs)#, deterministic = True)
        obs, rewards, done, info = env.step(action)
        reward += rewards
        time.sleep(1/240)
        #print(obs)
        #i += 1
    testing_rewards.append(reward)
    obs = env.reset()
    done = False

mean_reward = np.mean(testing_rewards)
std_reward = np.std(testing_rewards)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

env.close()