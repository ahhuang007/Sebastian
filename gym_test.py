# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 19:19:40 2020

@author: ahhua
"""

#Original code from here:
#https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/DDPG/tensorflow2/pendulum

import gym
from agent import Agent
import numpy as np

env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
     if 'seb' in env:
          print('Removed {} from registry'.format(env))
          del gym.envs.registration.registry.env_specs[env]

import gym_seb

env = gym.make('seb-v0')
agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])

best_reward = 0
scores = []
load_checkpoint = False
epochs = 250

if load_checkpoint:
    n_steps = 0
    while n_steps <= agent.batch_size:
        observation = env.reset()
        action = env.action_space.sample()
        observation_, reward, info = env.step(action)
        agent.remember(observation, action, reward, observation_)
        n_steps += 1
    agent.learn()
    agent.load_models()
    evaluate = True
else:
    evaluate = False

for i in range(epochs):
    observation = env.reset()
    score = 0
    iterations = 2000
    for j in range(iterations):
        action = agent.choose_action(observation, evaluate)
        observation_, reward, info = env.step(action)
        score += reward
        agent.remember(observation, action, reward, observation_)
        if not load_checkpoint:
            agent.learn()
        observation = observation_

    scores.append(score)
    avg_score = np.mean(scores[-100:])
    #avg_score = np.mean(scores[0:])

    if avg_score > best_reward:
        best_reward = avg_score
        if not load_checkpoint:
            agent.save_models()

    print('episode ', i, 'score %.4f' % score, 'avg score %.4f' % avg_score)

import matplotlib.pyplot as plt

if not load_checkpoint:
    x = [i+1 for i in range(epochs)]
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    #plt.savefig(figure_file)
      
env.close()