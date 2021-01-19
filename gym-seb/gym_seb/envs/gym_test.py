# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 19:19:40 2020

@author: ahhua
"""

import gym
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
     if 'seb' in env:
          print('Remove {} from registry'.format(env))
          del gym.envs.registration.registry.env_specs[env]
import gym_seb
from baselines import deepq
env = gym.make('seb-v0')
env.reset()
act = deepq.learn(env, network = 'mlp')
act.save('train.pkl')
'''
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        #env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward = env.step(action)
'''      
env.close()