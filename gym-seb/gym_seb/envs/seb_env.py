# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 00:21:56 2020

@author: MSI
"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as p
import time
import pybullet_data
import numpy as np
import pandas as pd

class SebEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self,
               episode_timesteps,
               use_gui,
               urdf_root=pybullet_data.getDataPath(),
               distance_limit=float("inf"),
               self_collision_enabled=True,
               motor_velocity_limit=np.inf,
               pd_control_enabled=False,
               leg_model_enabled=True,
               accurate_motor_model_enabled=False,
               remove_default_joint_damping=False,
               motor_kp=1.0,
               motor_kd=0.02,
               control_latency=0.0,
               pd_latency=0.0,
               torque_control_enabled=False,
               motor_overheat_protection=False,
               hard_reset=True,
               on_rack=False,
               render=False,
               num_steps_to_log=1000,
               action_repeat=1,
               control_time_step=None,
               env_randomizer=None,
               forward_reward_cap=float("inf"),
               reflection=True,
               log_path=None):
    super(SebEnv, self).__init__()
    if use_gui:
        self.physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    else:
        self.physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-10)
    p.resetDebugVisualizerCamera(cameraDistance = 2.5, cameraYaw=40, cameraPitch=-36, cameraTargetPosition=[0,0,0])
    planeId = p.loadURDF("plane.urdf")
    
    self.cubeStartPos = [0,0,0.1]
    
    self.cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
    
    self.boxId = p.loadURDF("sebastian_v3.urdf",self.cubeStartPos, self.cubeStartOrientation, 
                       flags = p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
    
    self.mode = p.POSITION_CONTROL
    self.joints = [1,2,3,5,6,7,9,10,11,13,14,15]

    cubePos, cubeOrn = p.getBasePositionAndOrientation(self.boxId)
    jointData = p.getJointStates(self.boxId, jointIndices=self.joints, physicsClientId=self.physicsClient)
    jointPos = [x[0] for x in jointData]
    
       
    self.action_space = spaces.Box(-1, +1, shape = (12,), dtype = 'float32')
    self.observation_space = spaces.Box(-100000, +100000, shape = (18,), dtype = 'float32')
    self.episode_number = 0
    self.episode_timesteps = episode_timesteps
    self.timestep_num = 0

  def step(self, action):
    self.episode_number += 1
    self.timestep_num += 1
    p.stepSimulation()
    op, oo = p.getBasePositionAndOrientation(self.boxId)
    
    pos = action*1.5708 #.numpy()
    
    p.setJointMotorControlArray(self.boxId, self.joints, controlMode=self.mode, targetPositions=pos)
    
    nep, no = p.getBasePositionAndOrientation(self.boxId)
    jointData = p.getJointStates(self.boxId, jointIndices=self.joints, physicsClientId=self.physicsClient)
    jointPos = tuple([x[0] for x in jointData])
    
    
    observation = nep
    orientation = p.getEulerFromQuaternion(no)
    
    forward_reward = (nep[0] - op[0])/(1/240)
    deviation_reward = (np.abs(nep[1]))/(1/240) 
    pitch_reward = 0.005*np.abs(orientation[0])
    yaw_reward = 0.005*np.abs(orientation[2])
    ctrl_cost = 0.005 * np.square(pos).sum()
    survive_reward = 0.5
    reward = forward_reward - ctrl_cost - deviation_reward + survive_reward - pitch_reward - yaw_reward
    #Experimental reward function below
    #reward = -np.abs(forward_reward - 0.021) - 0.001*np.abs(no[0]) - 0.01*(no[2]**2 + no[1]**2)
    info = {}
    total_obs = tuple(observation) + tuple(orientation) + jointPos
    done = False
    if orientation[0] > 0.7 or orientation[0] < -1.5708:
      done = True
      print("robot has flipped over at timestep " + str(self.episode_number))
    elif self.timestep_num > 10000:
      done = True
      print("maximum timestep reached for episode")
    if self.episode_number % 10000 == 0:
        print("at episode " + str(self.episode_number))
        print(reward)
    info['ep'] = self.timestep_num
      
    return np.array(total_obs, dtype = 'float32'), reward, done, info

  def reset(self):
    p.resetBasePositionAndOrientation(self.boxId, self.cubeStartPos, self.cubeStartOrientation)
    p.resetBaseVelocity(self.boxId, [0, 0, 0], [0, 0, 0])
    position, ori = p.getBasePositionAndOrientation(self.boxId)
    jointData = p.getJointStates(self.boxId, jointIndices=self.joints, physicsClientId=self.physicsClient)
    jointPos = tuple([x[0] for x in jointData])
    self.timestep_num = 0
    print("resetting environment")
    ori = p.getEulerFromQuaternion(ori)
    all_data = tuple(position) + tuple(ori) + jointPos
    return np.array(all_data, dtype = 'float32')
    
  def render(self, mode='human'):
    ...
  def close(self):
    p.disconnect()