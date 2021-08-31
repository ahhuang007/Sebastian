# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 19:19:40 2020

@author: ahhua
"""

#Original code from here:
#https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/DDPG/tensorflow2/pendulum

import gym
from agent import Agent

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
distances = []
load_checkpoint = False

if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            n_steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation, evaluate)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
for i_episode in range(2):
    observation = env.reset()
    for t in range(100):
        #env.render()
        print(observation)
        action = env.action_space.sample()
        print(action)
        observation, reward = env.step(action)
      
env.close()