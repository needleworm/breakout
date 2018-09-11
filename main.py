"""
    Pytorch Exercise
    Breakout with pytorch
    2018.09.11.
"""

import torch as tc
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import gym
import utils
import graph as G
from itertools import count

logs_dir = "logs/"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GAME = "Breakout-v0"
env = gym.make(GAME)

learning_rate = 0.001
batch_size = 200
num_episodes = 15000
NETWORK = G.DQN

xtrim = (33, 193)
ytrim = (8, -8)

GAMMA = 0.99
EPS_init = 0.9
EPS_final = 0.05
EPS_decay = 200
TARGET_UPDATE = 10


def train():
    # Graph Part
    print("Graph initialization...")
    policy_net = NETWORK(env.obsercation_space.shape,
                        env.action_space.size,
                        learning_rate,
                        batch_size).to(DEVICE)

    target_net = NETWORK(env.obsercation_space.shape,
                        env.action_space.size,
                        learning_rate,
                        batch_size).to(DEVICE)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Memory
    memory = utils.ReplayMemory(10000)

    # ETCs
    steps_done = 0
    episode_durations = []

    for episode in range(len(num_episodes)):
        previous_screenshot = env.reset()[xtrim[0]:xtrim[1], ytrim[0]:ytrin[1]]
        current_screenshot = previous_screenshot
        state = current_screenshot - previous_screenshot
        for t in count():
            action = utils.select_action(state, steps_done, policy_net)
            observation, reward, done, _ = env.step(action.item())
            previous_screenshot = current_screenshot
            current_screenshot = observation[xtrim[0]:xtrim[1], ytrim[0]:ytrim[1]]

            if not done:
                next_status = current_screenshot - previous_screenshot
            else:
                next_status = None

            memory.push(state, action, next_status, reward)
            state = next_status

            utils.optimize_model(policy_net, target_net)
            if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
        if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

train()
