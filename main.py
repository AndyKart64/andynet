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
import math
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image
# import wandb
from time import gmtime, strftime
import cv2
from datetime import datetime
from PIL import Image
from collections import deque
from datetime import datetime

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

def save_weights(model, optimizer, filename="model_weights.pth"):
    """
    Save the model weights and optimizer state.
    """
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    output_dir = '/src/gym_mupen64plus/logs/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, filename)
    torch.save(state, filename)
    print("Model weights saved to", filename)

def load_weights(model, optimizer, filename="model_weights.pth"):
    """
    Load the model weights and optimizer state.
    """
    output_dir = '/src/gym_mupen64plus/logs/'
    filename = os.path.join(output_dir, filename)
    if os.path.isfile(filename):
        state = torch.load(filename)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        print("Model weights loaded from", filename)
    else:
        print("No checkpoint found at", filename)

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
ERB_CAPACITY=5000
BATCH_SIZE=32

EPISODES=100
C=64 # how often we update Q to Q_hat
LEARNING_RATE=1e-4
EPSILON_ORIGINAL=0.3 # for e-greedy
EPSILON=0
GAMMA=0.9 # for Q-learning

EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

EPISODE_TIME = 70

# Timezone for logging
# now = strftime("%Y-%m-%d %H:%M:%S", gmtime())

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="AndyKart",
#     name=now,
#     # track hyperparameters and run metadata
#     config={
#     "erb capacity": ERB_CAPACITY,
#     "batch size": BATCH_SIZE,
#     "episodes": EPISODES,
#     "c": C,
#     "epsilon": EPSILON
#     }
# )

def lane_detection(image):
    # Convert to grayscale
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    # Perform Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    # Define region of interest
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height * 0.8),
        (width, height * 0.8),
        (width, height),
        (0, height),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    # Perform Hough Transform to detect lines
    lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, 50, maxLineGap=50)
    # Create an image to draw lines on
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    # Combine the line image with the original image
    lanes = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return lanes

def is_on_pavement(image, car_position, threshold=50):
    """
    Check if the car is on the pavement by checking the color of the pixel at car_position.
    """
    pixel_color = image[car_position[1], car_position[0]]
    gray_value = np.dot(pixel_color[:3], [0.2989, 0.5870, 0.1140])
    return gray_value > threshold

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
AndyW = optim.AdamW

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    cond = error.abs() < delta
    squared_loss = 0.5 * error.pow(2)
    linear_loss = delta * (error.abs() - 0.5 * delta)
    return torch.where(cond, squared_loss, linear_loss).mean()

model = DQN()
optimizer = AndyW(model.parameters(), lr=LEARNING_RATE, amsgrad=True)

load_weights(model, optimizer)

# Second model for Double DQN learning
model_prime = DQN
model_prime.load_state_dict(model.state_dict())

target_model = DQN()
target_model.load_state_dict(model.state_dict())

replay_buffer = ReplayBuffer(capacity=ERB_CAPACITY)

# Load checkpoint if it exists
load_weights(model, optimizer)

env = gym.make('Mario-Kart-Moo-Moo-Farm-v0')

best_checkpoint = 0
cur_checkpoint = 0

# loss_values = []
reward_values = [] # used for graphing reward

for episode in range(EPISODES):

    print("Episode ", episode, " ========= ")

    state = env.reset()
    print('state', state.shape, state)

    print("NOOP waiting for green light")
    for i in range(18):
        (obs, rew, end, info) = env.step([0, 0, 0, 0, 0]) # NOOP until green light

    print("GO!")

    if episode % 10 == 0:
        EPSILON = EPSILON_ORIGINAL

    # episode doesn't stop until terminal
    max_frames = EPISODE_TIME
    frame = 0
    frames_since_checkpoint = 0
    total_reward = 0
    while frame < max_frames:
    

        # choose action to take via e-greedy approach
        if random.random() < EPSILON:
            # select random action
            action = random.randint(0, len(DiscreteActions.ACTION_MAP) - 1)
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

        # Penalize if not on the pavement
        car_position = (next_state.shape[1] // 2, next_state.shape[0] // 2)  # Center of the screen
        if not is_on_pavement(next_state, car_position):
            reward -= 10  # Penalize for going off-pavement
            print("Penalize for going off pavement")

        if reward > 0:
            cur_checkpoint += 1
            max_frames += 1 # get more time if we make progress

            reward = math.exp(-1/4*frames_since_checkpoint) + 0.5
            frames_since_checkpoint = 0
            # print('reached checkpoint, reward:', reward)

            # first time bonus reward
            '''
            if cur_checkpoint < best_checkpoint:
                best_checkpoint = cur_checkpoint
                reward += 1
            '''
        # wandb.log({ "reward": rew })

        total_reward += reward
        # save to ERB
        # TODO could technically reuse some of the reprocess calls
        replay_buffer.add(preprocess(state), action, reward, preprocess(next_state))

        # lane_image = lane_detection(next_state)
        # # Save lane detection result
        # output_dir = '/src/gym_mupen64plus/logs/'
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # lane_image_filename = os.path.join(output_dir, 'lane_image.jpeg')
        # cv2.imwrite(lane_image_filename, lane_image)
        
        # sample random minibatch from ERB
        if len(replay_buffer) >= BATCH_SIZE:
            state_batch, action_batch, reward_batch, next_state_batch = replay_buffer.sample(BATCH_SIZE)
            state_batch = torch.tensor(state_batch, dtype=torch.float32).unsqueeze(1)
            action_batch = torch.tensor(action_batch).unsqueeze(1)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
            next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32).unsqueeze(1)

            # compute Q(s_t, a)
            # print('action_batch', action_batch.shape, action_batch)
            state_action_values = model(state_batch).gather(1, action_batch)
            # print('state_action_values', state_action_values.shape, state_action_values)
            # print('state_action_values', state_action_values.shape)

            # compute argmax Q_hat(s_t+1, a)
            with torch.no_grad():
                argmax_Q = target_model(next_state_batch).max(1)[0]

            # print('argmax_q', argmax_Q.shape)
            # print('rewards', reward_batch.shape)

            target_q_values = reward_batch + GAMMA * argmax_Q

            # compute loss
            # TODO terminate
            loss = huber_loss(target_q_values.unsqueeze(1), state_action_values)

            # loss_values.append(loss.item())

            # backprop on CNN
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 100) # TODO what value to set
            optimizer.step()

        if frame % C == 0:
            target_model.load_state_dict(model.state_dict())
            print('frame', frame, '/', max_frames, '=======')
            print('epsilon', EPSILON)

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

        # reset target action-value function
        state = next_state
        frame += 1
        frames_since_checkpoint += 1

        # kill agent if taking too long to get checkpoint
        # if frames_since_checkpoint > 10:
        #     break

    reward_values.append(total_reward)

    # save plot of rewards
    x = np.arange(0, len(reward_values))
    plt.plot(x, reward_values)
    plt.savefig('/src/gym_mupen64plus/logs/' + 'rewards_' + timestamp)

    # if episode % 10 == 0:
    #     save_weights(model, optimizer)

    save_weights(model, optimizer)

raw_input("Press <enter> to exit... ")


env.close()
cv2.destroyAllWindows()
# plt.figure(figsize=(10, 6))
# plt.plot(range(len(loss_values)), loss_values, label='Loss')
# plt.xlabel('Episode')
# plt.ylabel('Loss')
# plt.title('Loss over Episodes')
# plt.legend()
# plt.grid(True)
# plt.savefig('loss_plot.png')
