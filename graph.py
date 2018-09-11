import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class DQN(nn.Moddule):
    def __init__(obs_shape, act_size, learning_rate, batch_size):
        self.ydim, self.xdim, self.channels = obs_shape
        self.layers(act_size)

        self.optimizer = optim.Adam(self.parameters())

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def layers(self, act_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.channels, 64, kernel_size=3, stride=1),
            nn.BatchNorm2dd(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2dd(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2dd(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.BatchNorm2dd(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        )
        self.fc1 = nn.Linear(9 * 8 * 512, 128)
        self.fc2 = nn.Linear(128, act_size)

    def optimize(self, state_action_value, expected_state_action_values):
        self.loss = F.smooth_l1_loss(state_action_value, expected_state_action_values)
        self.optimizer.zero_grad()
        self.loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
