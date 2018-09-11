import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import random
import math

GAMMA = 0.99
EPS_init = 0.9
EPS_final = 0.05
EPS_decay = 200
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


Transition = namedtuple("Transition", ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def select_action(state, steps_done, policy_net):
    sample = random.random()
    eps_threshold = EPS_final + (EPS_init - EPS_final) * \
        math.exp(-1. * steps_done / EPS_decay)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=DEVICE, dtype=torch.long)


def dimension_manipulation(input):
    w, h, c = input.shape
    ret = np.zeros((c, w, h), dtype=np.double)
    ret[0, :, :] = input[:, :, 0]
    ret[1, :, :] = input[:, :, 1]
    ret[2, :, :] = input[:, :, 2]
    return ret[None, :, :, :]


def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model(policy_net, target_net, memory, BATCH_SIZE):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.uint8)
    non_final_next_states = []
    for s in batch.next_state:
        if s is not None:
            non_final_next_states.append(s)
    non_final_next_states = torch.cat(non_final_next_states)

#    torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    policy_net.optimize(state_action_values, expected_state_action_values.unsqueeze(1))
