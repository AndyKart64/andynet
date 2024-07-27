#!/bin/python
import torch
import gym, gym_mupen64plus
import numpy as np
import os
from PIL import Image
import matplotlib.image

def rgb_to_gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

env = gym.make('Mario-Kart-Luigi-Raceway-v0')
env.reset()

print("NOOP waiting for green light")
for i in range(18):
    (obs, rew, end, info) = env.step([0, 0, 0, 0, 0]) # NOOP until green light

print("GO! ...drive straight as fast as possible...")

## Sanity Grayscale
(obs, rew, end, info) = env.step([0, 0, 1, 0, 0]) # Drive straight
obs_grayscale = rgb_to_gray(obs)
print(obs_grayscale)
print(obs_grayscale.shape)
matplotlib.image.imsave('/src/gym_mupen64plus/logs/sanity.jpeg', obs_grayscale, cmap='gray')

for i in range(50):
    (obs, rew, end, info) = env.step([0, 0, 1, 0, 0]) # Drive straight

print("Doughnuts!!")
for i in range(10000):
    if i % 100 == 0:
        print("Step " + str(i))
    (obs, rew, end, info) = env.step([-80, 0, 1, 0, 0]) # Hard-left doughnuts!
    (obs, rew, end, info) = env.step([-80, 0, 0, 0, 0]) # Hard-left doughnuts!

raw_input("Press <enter> to exit... ")

env.close()
