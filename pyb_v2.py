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
import csv


physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version

p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

p.setGravity(0,0,-10)
p.resetDebugVisualizerCamera(cameraDistance = 1.5, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[0,0,0])
planeId = p.loadURDF("plane.urdf")
sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
sphere2 = p.createCollisionShape(p.GEOM_SPHERE, radius = 0.09)
cubeStartPos = [0,0,0.09]

cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])

boxId = p.loadURDF("sebastian_v2.urdf",cubeStartPos, cubeStartOrientation, flags = p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
#box2Id = p.createMultiBody(0, sphere, basePosition = [0, 0, 0])
#box3Id = p.createMultiBody(0, sphere, basePosition = [0.25, 0, 0])
#box4Id = p.createMultiBody(0, sphere2, basePosition = [0, 0.3, 0])


mode = p.POSITION_CONTROL
joints = [1,2,3,5,6,7,9,10,11,13,14,15]

cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)


coefs = []
for i in range(36):
    coefs.append(np.random.uniform(-1.5708, 1.5708))
positions = []
posind = []
allpos = []
allind = np.linspace(0, 9999, num = 10000)
highest = 0
for j in range(10):
    print(j)
    num = np.random.randint(0, 23)
    temp = coefs[num]
    coefs[num] = np.random.uniform(-1.5708, 1.5708)
    for i in range (10000):
        p.stepSimulation()
        
        pos = [coefs[0] + coefs[1]*np.sin(i*0.01 + coefs[2]), coefs[3] + coefs[4]*np.sin(i*0.01 + coefs[5]), 
               coefs[6] + coefs[7]*np.sin(i*0.01 + coefs[8]), coefs[9] + coefs[10]*np.sin(i*0.01 + coefs[11]), 
               coefs[12] + coefs[13]*np.sin(i*0.01 + coefs[14]), coefs[15] + coefs[16]*np.sin(i*0.01 + coefs[17]),
               coefs[18] + coefs[19]*np.sin(i*0.01 + coefs[20]), coefs[21] + coefs[22]*np.sin(i*0.01 + coefs[23]),
               coefs[24] + coefs[25]*np.sin(i*0.01 + coefs[26]), coefs[27] + coefs[28]*np.sin(i*0.01 + coefs[29]),
               coefs[30] + coefs[31]*np.sin(i*0.01 + coefs[32]), coefs[33] + coefs[34]*np.sin(i*0.01 + coefs[35])]
        #jump1 = [0, 0.3, 0, -0.3, 0, 0.3, 0, -0.3] 
        #jump2 = [0, -1.5708, 0, 1.5708] * 2
        #pos2 = [0, -1*(a2+b2*np.sin(i*0.01+c2)), 0, a2+b2*np.sin(i*0.01+c2 + 3.1415)] * 2
        p.setJointMotorControlArray(boxId, joints, controlMode=mode, targetPositions=pos)
        
        #time.sleep(1./240.)
    cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    print(cubePos)
    reward = cubePos[0] - np.abs(cubePos[1])
    #reward = -1*cubePos[1] - np.abs(cubePos[0])
    if reward > highest:
        highest = reward
        positions.append(reward)
        posind.append(j)
    else:
        coefs[num] = temp
    print(reward)
    allpos.append(reward)
    p.resetBasePositionAndOrientation(boxId, cubeStartPos, cubeStartOrientation)
    p.resetBaseVelocity(boxId, [0, 0, 0], [0, 0, 0])

print(highest)
p.disconnect()

vals = pd.DataFrame(data={"iter": posind, "pos": positions})
allvals = pd.DataFrame(data={"iter": allind, "pos": allpos})
vals.to_csv('C://Users//ahhua//Documents//hc.csv')
allvals.to_csv('C://Users//ahhua//Documents//ahc.csv')
#%%
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
p.resetDebugVisualizerCamera(cameraDistance = 1.5, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[0,0,0])
planeId = p.loadURDF("plane.urdf")
#sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
#sphere2 = p.createCollisionShape(p.GEOM_SPHERE, radius = 0.09)
cubeStartPos = [0,0,0.5]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("sebastian_v2.urdf",cubeStartPos, cubeStartOrientation, flags = p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
#box2Id = p.createMultiBody(0, sphere, basePosition = [0, 0, 0])
#box3Id = p.createMultiBody(0, sphere, basePosition = [0.25, 0, 0])
#box4Id = p.createMultiBody(0, sphere2, basePosition = [0, 0.3, 0])


mode = p.POSITION_CONTROL

for i in range (10000):
    p.stepSimulation()
    
    pos = [coefs[0] + coefs[1]*np.sin(i*0.01 + coefs[2]), coefs[3] + coefs[4]*np.sin(i*0.01 + coefs[5]), 
           coefs[6] + coefs[7]*np.sin(i*0.01 + coefs[8]), coefs[9] + coefs[10]*np.sin(i*0.01 + coefs[11]), 
           coefs[12] + coefs[13]*np.sin(i*0.01 + coefs[14]), coefs[15] + coefs[16]*np.sin(i*0.01 + coefs[17]),
           coefs[18] + coefs[19]*np.sin(i*0.01 + coefs[20]), coefs[21] + coefs[22]*np.sin(i*0.01 + coefs[23]),
           coefs[24] + coefs[25]*np.sin(i*0.01 + coefs[26]), coefs[27] + coefs[28]*np.sin(i*0.01 + coefs[29]),
           coefs[30] + coefs[31]*np.sin(i*0.01 + coefs[32]), coefs[33] + coefs[34]*np.sin(i*0.01 + coefs[35])]
    p.setJointMotorControlArray(boxId, joints, controlMode=mode, targetPositions=pos)
    
    time.sleep(1./240.)

p.disconnect()

#%%
with open('C://Users//MSI//Documents//coefs.txt', 'w', newline='') as myfile:
     for item in coefs:
        myfile.write("%s\n" % item)