#!/bin/python
import random
import os
import torch
import gym, gym_mupen64plus
import numpy as np
import matplotlib.image
import wandb
import pytz
import cv2
from datetime import datetime
from PIL import Image
from collections import deque

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
    matplotlib.image.imsave('/src/gym_mupen64plus/logs/' + file_name, gray, cmap='gray')

    return cv2.resize(gray, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)


## Hyperparameters
ERB_CAPACITY=1000
BATCH_SIZE=32

EPISODES=100
C=64 # learning rate
EPSILON=0.9 # for e-greedy

# Timezone for logging
timezone = pytz.timezone("Canada/Eastern")

wandb.init(
    # set the wandb project where this run will be logged
    project="AndyKart",
    name=datetime.now(timezone).strftime("%H:%M:%S"),
    # track hyperparameters and run metadata
    config={
    "erb capacity": ERB_CAPACITY,
    "batch size": BATCH_SIZE,
    "episodes": EPISODES,
    "c": C,
    "epsilon": EPSILON
    }
)

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
            wandb.log({ "reward": rew })
            # preprocess image
            # convert observation to greyscale
            greyscale = rgb_to_gray(obs)

            # Resize the image to 84 x 84
            rescaled = cv2.resize(greyscale, dsize=(84, 84), interpolation=cv2.INTER_LINEAR)

            if i == 0:
                save_grayscale_image(greyscale, 'greyscale_image.jpg')
                save_grayscale_image(rescaled, 'rescaled_image.jpg')

            # store observation in ERB

            # sample random minibatch from ERB

            # compute loss

            # backprop on CNN

            # reset target action-value function

raw_input("Press <enter> to exit... ")

env.close()
