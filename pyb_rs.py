# -*- coding: utf-8 -*-
"""
Created on Sun May  3 01:27:07 2020

@author: MSI
"""

import pybullet as p

import time

import pybullet_data

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

for k in range(4):
    physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    
    p.setGravity(0,0,-10)
    p.resetDebugVisualizerCamera(cameraDistance = 1.5, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[0,0,0])
    planeId = p.loadURDF("plane.urdf")
    sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
    sphere2 = p.createCollisionShape(p.GEOM_SPHERE, radius = 0.09)
    cubeStartPos = [0,0,1]
    
    cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
    
    boxId = p.loadURDF("sebastian.urdf",cubeStartPos, cubeStartOrientation)
    #box2Id = p.createMultiBody(0, sphere, basePosition = [0, 0, 0])
    #box3Id = p.createMultiBody(0, sphere, basePosition = [0.25, 0, 0])
    #box4Id = p.createMultiBody(0, sphere2, basePosition = [0, 0.3, 0])
    
    
    mode = p.POSITION_CONTROL
    joints = [1,2,4,5,7,8,10,11]
    
    cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    print(cubePos,cubeOrn)
    coefs = [0]*24
    best = coefs
    positions = []
    posind = []
    allpos = []
    allind = np.linspace(0, 999, num = 1000)
    highest = 0
    for j in range(1000):
        print(j)
        num = np.random.randint(0, 23)
        for i in range(24):
            coefs[i] = np.random.uniform(-1, 1)
        for i in range (10000):
            p.stepSimulation()
            
            pos = [coefs[0] + coefs[1]*np.sin(i*0.01 + coefs[2]), coefs[3] + coefs[4]*np.sin(i*0.01 + coefs[5]), 
                   coefs[6] + coefs[7]*np.sin(i*0.01 + coefs[8]), coefs[9] + coefs[10]*np.sin(i*0.01 + coefs[11]), 
                   coefs[12] + coefs[13]*np.sin(i*0.01 + coefs[14]), coefs[15] + coefs[16]*np.sin(i*0.01 + coefs[17]),
                   coefs[18] + coefs[19]*np.sin(i*0.01 + coefs[20]), coefs[21] + coefs[22]*np.sin(i*0.01+coefs[23])]
            #jump1 = [0, 0.3, 0, -0.3, 0, 0.3, 0, -0.3] 
            #jump2 = [0, -1.5708, 0, 1.5708] * 2
            #pos2 = [0, -1*(a2+b2*np.sin(i*0.01+c2)), 0, a2+b2*np.sin(i*0.01+c2 + 3.1415)] * 2
            p.setJointMotorControlArray(boxId, joints, controlMode=mode, targetPositions=pos)
            '''
            if i < 1000:
                p.setJointMotorControlArray(boxId, joints, controlMode=mode, targetPositions=jump1)
            elif i > 1000 and i < 1020:
                p.setJointMotorControlArray(boxId, joints, controlMode=mode, targetPositions=jump2)
            else:
                p.setJointMotorControlArray(boxId, joints, controlMode=mode, targetPositions=jump1)
            '''
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
            best = coefs
        print(reward)
        allpos.append(reward)
        p.resetBasePositionAndOrientation(boxId, cubeStartPos, cubeStartOrientation)
        p.resetBaseVelocity(boxId, [0, 0, 0], [0, 0, 0])
    
    print(highest)
    p.disconnect()
    
    vals = pd.DataFrame(data={"iter": posind, "pos": positions})
    allvals = pd.DataFrame(data={"iter": allind, "pos": allpos})
    vals.to_csv('C://Users//MSI//Documents//rs_pyb_vals' + str(k+1) + '.csv')
    allvals.to_csv('C://Users//MSI//Documents//rs_all_pyb_vals' + str(k+1) + '.csv')
#%%
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
p.resetDebugVisualizerCamera(cameraDistance = 1.5, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[0,0,0])
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("sebastian.urdf",cubeStartPos, cubeStartOrientation)



mode = p.POSITION_CONTROL

for i in range (10000):
    p.stepSimulation()
    
    pos = [best[0] + best[1]*np.sin(i*0.01 + best[2]), best[3] + best[4]*np.sin(i*0.01 + best[5]), 
           best[6] + best[7]*np.sin(i*0.01 + best[8]), best[9] + best[10]*np.sin(i*0.01 + best[11]), 
           best[12] + best[13]*np.sin(i*0.01 + best[14]), best[15] + best[16]*np.sin(i*0.01 + best[17]),
           best[18] + best[19]*np.sin(i*0.01 + best[20]), best[21] + best[22]*np.sin(i*0.01+best[23])]
    
    p.setJointMotorControlArray(boxId, joints, controlMode=mode, targetPositions=pos)
    
    time.sleep(1./240.)
    
p.disconnect()
#%%
plt.figure(1, figsize=(16, 12))
plt.step(vals["iter"], vals["pos"], where = 'post', label = 'Random Search')
plt.xlabel("generation")
plt.ylabel("fitness")
plt.show()
plt.figure(2, figsize=(16, 12))
plt.step(allvals["iter"], allvals["pos"], where = 'post', label = 'Random Search (all iterations)')
plt.xlabel("generation")
plt.ylabel("fitness")
plt.show()