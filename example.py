#!/bin/python
import gym
from gym_mupen64plus.envs.MarioKart64.discrete_envs import DiscreteActions
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image
# import wandb
# import pytz
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
BATCH_SIZE=4

EPISODES=100
C=64 # how often we update Q to Q_hat
LEARNING_RATE=1e-4
EPSILON=0.99 # for e-greedy
GAMMA=0.9 # for Q-learning

# Timezone for logging
# timezone = pytz.timezone("Canada/Eastern")

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="AndyKart",
#     name=datetime.now(timezone).strftime("%H:%M:%S"),
#     # track hyperparameters and run metadata
#     config={
#     "erb capacity": ERB_CAPACITY,
#     "batch size": BATCH_SIZE,
#     "episodes": EPISODES,
#     "c": C,
#     "epsilon": EPSILON
#     }
# )

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
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = map(np.array, zip(*batch))
        return states, actions, rewards, next_states

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
        x = F.relu(x)
        x = self.fc2(x)

        # TODO sigmoid here

        return x

model = DQN()
target_model = DQN()
target_model.load_state_dict(model.state_dict())

replay_buffer = ReplayBuffer(capacity=ERB_CAPACITY)
AndyW = optim.AdamW
optimizer = AndyW(model.parameters(), lr=LEARNING_RATE, amsgrad=True)

env = gym.make('Mario-Kart-Luigi-Raceway-v0')

for episode in range(EPISODES):

    print("Episode ", episode, " ========= ")

    state = env.reset()
    print('state', state.shape, state)

    print("NOOP waiting for green light")
    for i in range(18):
        (obs, rew, end, info) = env.step([0, 0, 0, 0, 0]) # NOOP until green light

    print("GO!")

    counter=0

    # episode doesn't stop until terminal
    while True:

        # choose action to take via e-greedy approach
        if random.random() < 1-EPSILON:
            # select random action
            action = random.randint(0, len(DiscreteActions.ACTION_MAP))
            #print('selected action', action)

        else:
            # select optimal action
            phi_state = preprocess(state)
            tensor_state = torch.tensor(phi_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            q_values = model(tensor_state)
            action = torch.argmax(q_values, dim=1)
            #print('selected optimal action', action)

            
        # execute action in emulator
        # print('executing action', action[0])
        (next_state, reward, end, info) = env.step(DiscreteActions.ACTION_MAP[action][1])
        # wandb.log({ "reward": rew })

        # save to ERB
        # TODO could technically reuse some of the reprocess calls
        replay_buffer.add(preprocess(state), action, reward, preprocess(next_state))

        # sample random minibatch from ERB
        if len(replay_buffer) >= BATCH_SIZE:
            state_batch, action_batch, reward_batch, next_state_batch = replay_buffer.sample(BATCH_SIZE)
            state_batch = torch.tensor(state_batch, dtype=torch.float32).unsqueeze(1)
            action_batch = torch.tensor(action_batch).unsqueeze(1)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
            next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32).unsqueeze(1)

            # compute Q(s_t, a)
            print('action_batch', action_batch.shape, action_batch)
            state_action_values = model(state_batch).gather(1, action_batch)
            # print('state_action_values', state_action_values.shape, state_action_values)
            # print('state_action_values', state_action_values.shape)

            # compute argmax Q_hat(s_t+1, a)
            with torch.no_grad():
                argmax_Q = target_model(next_state_batch).max(1)[0]

            print('argmax_q', argmax_Q.shape)
            print('rewards', reward_batch.shape)

            target_q_values = reward_batch + GAMMA * argmax_Q

            # compute loss
            # TODO terminate
            loss = nn.SmoothL1Loss()(target_q_values, state_action_values)

            # backprop on CNN
            optimizer.zero_grad()
            loss.backward()

        # reset target action-value function
        state = next_state

raw_input("Press <enter> to exit... ")

env.close()
