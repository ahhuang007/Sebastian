# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 00:13:50 2020

@author: MSI
"""

from setuptools import setup

setup(name='gym_seb',
      version='0.0.3',
      description='gym for sebastian',
      #packages=setuptools.find_packages(include="gym_seb*"),
      install_requires=['gym>=0.20.0', 'pybullet', 'numpy', 'pandas']  # And any other dependencies foo needs
      
)