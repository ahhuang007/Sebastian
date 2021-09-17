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
'''
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
p.resetDebugVisualizerCamera(cameraDistance = 1.5, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[0,0,0])
planeId = p.loadURDF("plane.urdf")
#sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
#sphere2 = p.createCollisionShape(p.GEOM_SPHERE, radius = 0.09)
cubeStartPos = [0,0,0.1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("sebastian_v2.urdf",cubeStartPos, cubeStartOrientation, flags = p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
#box2Id = p.createMultiBody(0, sphere, basePosition = [0, 0, 0])
#box3Id = p.createMultiBody(0, sphere, basePosition = [0.25, 0, 0])
#box4Id = p.createMultiBody(0, sphere2, basePosition = [0, 0.3, 0])
'''
env = gym.make('seb-v0', max_timesteps = 100000)

from stable_baselines3.common.env_checker import check_env

check_env(env, warn=True)

mode = p.POSITION_CONTROL

model = PPO.load("real_model_ppo", env = env)
print("loaded")
done = False
obs = env.reset()
while done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    print(obs)
p.disconnect()