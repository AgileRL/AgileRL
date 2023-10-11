import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
import yaml
import supersuit as ss
from tqdm import trange
import torch
import os
import sys
from pettingzoo.mpe import simple_speaker_listener_v4

sys.path.append('../')

from agilerl.wrappers.make_evolvable import MakeEvolvable

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.training.train_on_policy import train_on_policy
from agilerl.training.train_multi_agent import train_multi_agent
from agilerl.networks.custom_activation import GumbelSoftmax
from agilerl.training.train import train
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import initialPopulation, makeVectEnvs, printHyperparams


from pettingzoo.atari import pong_v3

from networks import ClipReward,  BasicNetCritic,  \
SimpleCNNCritic, MultiCNNActor, MultiCNNCritic

import torch.nn as nn

class BasicNetActor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(BasicNetActor, self).__init__()
        layers = []

        # Add input layer with LayerNorm
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.LayerNorm(hidden_sizes[0]))  # LayerNorm
        layers.append(nn.ReLU())  # Activation function

        # Add hidden layers with LayerNorm
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.LayerNorm(hidden_sizes[i + 1]))  # LayerNorm
            layers.append(nn.ReLU())  # Activation function

        # Add output layer with a sigmoid activation
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        #layers.append(nn.Softmax())  # Sigmoid activation

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
import torch.nn as nn

class SimpleCNNActor(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNNActor, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)  # W: 160, H: 210
        self.bn1 = nn.BatchNorm2d(16)  # Batch Normalization
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)  # W: ...
        self.bn2 = nn.BatchNorm2d(32)  # Batch Normalization
        self.relu2 = nn.ReLU()

        # Define the max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define fully connected layers
        self.fc1 = nn.Linear(128, 256)  # Assuming input images are 128x128
        self.fc2 = nn.Linear(256, num_classes)

        # Define activation function
        self.relu3 = nn.ReLU()

        # Define softmax for classification
        self.softmax = nn.Softmax(dim=1)

        self.tanh = nn.Tanh()

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)  # Batch Normalization
        x = self.relu1(x)
        x = self.pool(x)  # Apply max-pooling

        x = self.conv2(x)
        x = self.bn2(x)  # Batch Normalization
        x = self.relu2(x)
        x = self.pool(x)  # Apply max-pooling

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Forward pass through fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x



device = "cuda" if torch.cuda.is_available() else "cpu"

network_actor = BasicNetActor(6 ,[64], 4)
actor = MakeEvolvable(network_actor,
                      input_tensor=torch.ones(6),
                      device=device)

network_cnn_actor = SimpleCNNActor(4)
cnn_actor = MakeEvolvable(network_cnn_actor,
                          input_tensor=torch.ones(4, 84, 84).unsqueeze(0),
                          device=device)
critic = None

print(cnn_actor)
print(actor)
