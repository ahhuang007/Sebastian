# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 18:21:51 2020

@author: MSI
"""

import pybullet as p
import time
import pybullet_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version

p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

p.setGravity(0,0,0)
p.resetDebugVisualizerCamera(cameraDistance = 1.5, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[0,0,0.5])
planeId = p.loadURDF("plane.urdf")

cubeStartPos = [0,0,0.5]

cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])

boxId = p.loadURDF("sebastian_v3.urdf",cubeStartPos, cubeStartOrientation)
#box2Id = p.createMultiBody(0, sphere, basePosition = [0, 0, 0])
#box3Id = p.createMultiBody(0, sphere, basePosition = [0.25, 0, 0])
#box4Id = p.createMultiBody(0, sphere2, basePosition = [0, 0.3, 0])


mode = p.POSITION_CONTROL
joints = [1,2,3,5,6,7,9,10,11,13,14,15]

cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)

