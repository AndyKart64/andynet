#!/bin/python
import random
import os
import torch
import gym, gym_mupen64plus
import numpy as np
import matplotlib.image
from PIL import Image
from collections import deque
import wandb

def rgb_to_gray(rgb):
    """
    Converts the three channel RGB colour to grayscale
        - rgb : np.ndarray
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def save_grayscale_image(gray, file_name):
    """
    Saves the grayscale image to the logs file
        - gray: np.ndarray
        - file_name: str
    """
    matplotlib.image.imsave('/src/gym_mupen64plus/logs/' + "file_name", gray, cmap='gray')


## Hyperparameters
ERB_CAPACITY=1000
BATCH_SIZE=32

EPISODES=100
C=64 # learning rate
EPSILON=0.9 # for e-greedy

## Experience Replay Buffer
class ReplayBuffer:
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    # Add a new experience to the ERB, discarding experiences if at max capacity
    def add(self, state, action, reward, next_state):
        experience = (state, action, reward, next_state)
        self.buffer.append(experience)

    # Randomly sample N experiences
    # It is crucial that these are not sampled in order, to break temporal correlation
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
wandb.init(
    project="andy-kart"
)

replay_buffer = ReplayBuffer(capacity=ERB_CAPACITY)

env = gym.make('Mario-Kart-Luigi-Raceway-v0')
env.reset()

for episode in range(EPISODES):
    # episode doesn't stop until terminal
    while True:
        print("Episode " + str(episode) + " =========")
        print("NOOP waiting for green light")
       
        for i in range(18):
            (obs, rew, end, info) = env.step([0, 0, 0, 0, 0]) # NOOP until green light

        print("GO!")
        for i in range(50):

            # choose action to take via e-greedy approach

            
            # execute action in emulator
            (obs, rew, end, info) = env.step([0, 0, 1, 0, 0]) # Drive straight

            # preprocess image
            # convert observation to greyscale
            greyscale = rgb_to_gray(obs)

            if i == 0:
                save_grayscale_image(greyscale, 'saved_greyscale_image.jpg')

            # store observation in ERB

            # sample random minibatch from ERB

            # compute loss

            # backprop on CNN

            # reset target action-value function

raw_input("Press <enter> to exit... ")

env.close()
