import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class ClipReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward: float) -> float:
        # Clip the reward to the range (-1, 1)
        return np.sign(float(reward))


class BasicNetActorDQN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []

        # Add input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())  # Activation function

        # Add hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())  # Activation function

        # Add output layer with a sigmoid activation
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        # layers.append(nn.Tanh())  # Sigmoid activation

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class BasicNetActor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []

        # Add input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())  # Activation function

        # Add hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())  # Activation function

        # Add output layer with a sigmoid activation
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        layers.append(nn.Tanh())  # Sigmoid activation

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class SoftmaxActor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []

        # Add input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())  # Activation function

        # Add hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())  # Activation function

        # Add output layer with a sigmoid activation
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        layers.append(nn.Softmax())  # Output activation

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class BasicNetCritic(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []

        # Add input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())  # Activation function

        # Add hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())  # Activation function

        # Add output layer with a sigmoid activation
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        # layers.append(nn.Sigmoid())  # Sigmoid activation

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class SimpleCNNActor(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=16, kernel_size=8, stride=4
        )  # W: 160, H: 210
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=4, stride=2
        )  # W:

        # Define the max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define fully connected layers
        self.fc1 = nn.Linear(2592, 256)  # Assuming input images are 128x128
        self.fc2 = nn.Linear(256, num_classes)

        # Define activation function
        self.relu = nn.ReLU()

        # Define softmax for classification
        self.softmax = nn.Softmax(dim=1)

        self.tanh = nn.Tanh()

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Forward pass through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class SimpleCNNCritic(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=16, kernel_size=8, stride=4
        )  # W: 160, H: 210
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=4, stride=2
        )  # W:

        # Define the max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define fully connected layers
        self.fc1 = nn.Linear(2592, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Define activation function
        self.relu = nn.ReLU()

        # Define softmax for classification
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Forward pass through fully connected layers
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)

        # Apply softmax for classification
        x = self.softmax(x)

        return x


class MultiCNNActor(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv3d(
            in_channels=4, out_channels=16, kernel_size=(1, 3, 3), stride=4
        )  # W: 160, H: 210
        self.conv2 = nn.Conv3d(
            in_channels=16, out_channels=32, kernel_size=(1, 3, 3), stride=2
        )

        # Define the max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define fully connected layers
        self.fc1 = nn.Linear(3200, 256)  # Assuming input images are 128x128
        self.fc2 = nn.Linear(256, num_classes)

        # Define activation function
        self.relu = nn.ReLU()

        # Define softmax for classification
        self.softmax = nn.Softmax(dim=1)

        self.tanh = nn.Tanh()

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Forward pass through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class MultiCNNCritic(nn.Module):
    def __init__(self, num_classes, num_agents):
        super().__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv3d(
            in_channels=4, out_channels=16, kernel_size=(1, 3, 3), stride=4
        )  # W: 160, H: 210
        self.conv2 = nn.Conv3d(
            in_channels=16, out_channels=32, kernel_size=(1, 3, 3), stride=2
        )  # W:

        # Define the max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define fully connected layers
        self.fc1 = nn.Linear(3202, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Define activation function
        self.relu = nn.ReLU()

        # Define softmax for classification
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

        self.action_tensor = torch.ones(1, num_agents, device="cuda")

    def forward(self, x, xc):
        # Forward pass through convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        x = torch.cat([x, xc], dim=1)
        # Forward pass through fully connected layers
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)

        # Apply softmax for classification
        x = self.softmax(x)

        return x
