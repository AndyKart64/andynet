#!/bin/python
import gym
from gym_mupen64plus.envs.MarioKart64.discrete_envs import DiscreteActions
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import matplotlib.pyplot as plt
from collections import deque
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

def downscale(img):
    return cv2.resize(img, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)

def preprocess(state):
    return downscale(rgb_to_gray(state))

def save_grayscale_image(gray, file_name):
    """
    Saves the grayscale image to the logs file
        - gray: np.ndarray
        - file_name: str
    """
    matplotlib.image.imsave('/src/gym_mupen64plus/logs/' + file_name, gray, cmap='gray')

    return downscale(gray)


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
class DQN(nn.Module):

    N_OBS=84*84
    N_ACTIONS=len(DiscreteActions.ACTION_MAP)
    HIDDEN_SIZE=128

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * DQN.N_OBS, DQN.HIDDEN_SIZE)
        self.fc2 = nn.Linear(DQN.HIDDEN_SIZE, DQN.N_ACTIONS)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

model = DQN()
print(model)

replay_buffer = ReplayBuffer(capacity=ERB_CAPACITY)

env = gym.make('Mario-Kart-Luigi-Raceway-v0')
state = env.reset()
print('state', state.shape, state)


for episode in range(EPISODES):

    print("Episode ", episode, " ========= ")

    print("NOOP waiting for green light")
    for i in range(18):
        (obs, rew, end, info) = env.step([0, 0, 0, 0, 0]) # NOOP until green light

    print("GO!")

    # episode doesn't stop until terminal
    while True:

        # choose action to take via e-greedy approach
        if random.random() < 1-EPSILON:
            # select random action
            action = random.choice(DiscreteActions.ACTION_MAP)
            #print('selected action', action)

        else:
            # select optimal action
            phi_state = preprocess(state)
            tensor_state = torch.tensor(state, dtype=torch.float32).unsqueeze(1)
            q_values = model(phi_state)
            action = DiscreteActions.ACTION_MAP[torch.argmax(q_values, dim=1)]
            #print('selected optimal action', action)

            
        # execute action in emulator
        print('executing action', action[0])
        (obs, rew, end, info) = env.step(action[1])
        wandb.log({ "reward": rew })

        # preprocess image

        # sample random minibatch from ERB

        # compute loss

        # backprop on CNN

        # reset target action-value function

raw_input("Press <enter> to exit... ")

env.close()
