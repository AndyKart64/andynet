#!/bin/python
import gym, gym_mupen64plus
import torch
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Mario-Kart-Luigi-Raceway-v0')
env.reset()

print("NOOP waiting for green light")
for i in range(18):
    (obs, rew, end, info) = env.step([0, 0, 0, 0, 0]) # NOOP until green light

print("GO! ...drive straight as fast as possible...")
for i in range(50):
    (obs, rew, end, info) = env.step([0, 0, 1, 0, 0]) # Drive straight

print("Doughnuts!!")
for i in range(10000):
    if i % 100 == 0:
        print("Step " + str(i))
    (obs, rew, end, info) = env.step([-80, 0, 1, 0, 0]) # Hard-left doughnuts!
    (obs, rew, end, info) = env.step([-80, 0, 0, 0, 0]) # Hard-left doughnuts!
    # print(type(obs), obs.shape, obs)

    # convert observation to greyscale
    greyscale = np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140])

    if i == 0:
        plt.imsave('saved_greyscale_image.png', greyscale, cmap='gray')

raw_input("Press <enter> to exit... ")

env.close()
