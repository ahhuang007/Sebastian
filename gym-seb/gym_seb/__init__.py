# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 00:16:01 2020

@author: MSI
"""

from gym.envs.registration import register

register(
    id='seb-v0',
    entry_point='gym_seb.envs:SebEnv',
)
