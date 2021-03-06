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
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-10)
    p.resetDebugVisualizerCamera(cameraDistance = 1.5, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[0,0,0])
    planeId = p.loadURDF("plane.urdf")
    
    self.cubeStartPos = [0,0,0.1]
    
    self.cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
    
    self.boxId = p.loadURDF("../../../sebastian_v2.urdf",self.cubeStartPos, self.cubeStartOrientation, 
                       flags = p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
    
    self.mode = p.POSITION_CONTROL
    self.joints = [1,2,3,5,6,7,9,10,11,13,14,15]

    cubePos, cubeOrn = p.getBasePositionAndOrientation(self.boxId)
    
    self.action_space = spaces.Box(np.array([-1.5708]*12), np.array([+1.5708]*12), dtype = np.float32)
    self.observation_space = spaces.Box(np.array([-100000]), np.array([+100000]))
  def step(self, action):
    p.stepSimulation()
    op, oo = p.getBasePositionAndOrientation(self.boxId)
    
    pos = action
    
    p.setJointMotorControlArray(self.boxId, self.joints, controlMode=self.mode, targetPositions=pos)
    time.sleep(1./25.)
    nep, no = p.getBasePositionAndOrientation(self.boxId)
    observation = nep
    reward = 0
    if nep > op:
        reward = 1
    return observation, reward
  def reset(self):
    p.resetBasePositionAndOrientation(self.boxId, self.cubeStartPos, self.cubeStartOrientation)
    p.resetBaseVelocity(self.boxId, [0, 0, 0], [0, 0, 0])
    
  def render(self, mode='human'):
    ...
  def close(self):
    p.disconnect()