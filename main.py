"""
    Pytorch Exercise
    Breakout with pytorch
    2018.09.11.
"""

import torch
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
batch_size = 100
num_episodes = 150000
NETWORK = G.DQN

xtrim = (33, 193)
ytrim = (8, 172)

TARGET_UPDATE = 10


def train():
    # Graph Part
    print("Graph initialization...")
    xdim = xtrim[1] - xtrim[0]
    ydim = ytrim[1] - ytrim[0]
    channel=3
    num_action = env.action_space.n
    policy_net = NETWORK(ydim=ydim, xdim=xdim, channel=channel,
                        num_action=num_action,
                        learning_rate=learning_rate,
                        batch_size=batch_size)

    target_net = NETWORK(ydim=ydim, xdim=xdim, channel=channel,
                        num_action=num_action,
                        learning_rate=learning_rate,
                        batch_size=batch_size)
    policy_net.to(DEVICE)
    target_net.to(DEVICE)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Memory
    memory = utils.ReplayMemory(10000)

    # ETCs
    steps_done = 0
    episode_durations = []

    policy_net.float()
    target_net.float()

    print("Training Start.....")
    for episode in range(num_episodes):
        REWARD = 0
        previous_screenshot = utils.dimension_manipulation(env.reset()[xtrim[0]:xtrim[1], ytrim[0]:ytrim[1]])
        current_screenshot = previous_screenshot
        state = torch.from_numpy(current_screenshot - previous_screenshot).float().to(DEVICE)
        for t in count():
            #env.render()
            action = utils.select_action(state, steps_done, policy_net)
            observation, reward, done, _ = env.step(action.item())
            previous_screenshot = current_screenshot
            current_screenshot = utils.dimension_manipulation(observation[xtrim[0]:xtrim[1], ytrim[0]:ytrim[1]])

            if not done:
                next_status = torch.from_numpy(current_screenshot - previous_screenshot).float().to(DEVICE)
                REWARD += reward
            else:
                next_status = None
            if True :
                memory.push(state,
                            action,
                            next_status,
                            torch.tensor(float(t+1)).to(DEVICE)[None])
            state = next_status
            utils.optimize_model(policy_net, target_net, memory, batch_size)

            if done:
                utils.optimize_model(policy_net, target_net, memory, batch_size)
                episode_durations.append(t + 1)
                utils.plot_durations(episode_durations)
                if REWARD != 0:
                    print("\n########  Episode " + str(episode))
                    print("Duration : " + str(t + 1))
                    print("REWARD : " + str(REWARD))
                    print("loss : " + str(policy_net.loss.item()))
                break
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

train()
