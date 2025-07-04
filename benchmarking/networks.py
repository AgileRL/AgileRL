from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from agilerl.modules.base import EvolvableModule, MutationType, mutation
from agilerl.typing import ArrayOrTensor
from agilerl.utils.evolvable_networks import create_mlp


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
            in_channels=4, out_channels=16, kernel_size=(2, 3, 3), stride=4
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


class SimpleCritic(EvolvableModule):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        hidden_size: List[int],
        activation: str = "ReLU",
        output_activation: str = None,
        min_hidden_layers: int = 1,
        max_hidden_layers: int = 3,
        min_mlp_nodes: int = 64,
        max_mlp_nodes: int = 500,
        layer_norm: bool = True,
        output_vanish: bool = True,
        init_layers: bool = True,
        noisy: bool = False,
        noise_std: float = 0.5,
        new_gelu: bool = False,
        device: str = "cpu",
        name: str = "mlp",
    ):
        super().__init__(device)

        assert (
            num_inputs > 0
        ), "'num_inputs' cannot be less than or equal to zero, please enter a valid integer."
        assert (
            num_outputs > 0
        ), "'num_outputs' cannot be less than or equal to zero, please enter a valid integer."
        for num in hidden_size:
            assert (
                num > 0
            ), "'hidden_size' cannot contain zero, please enter a valid integer."
        assert len(hidden_size) != 0, "MLP must contain at least one hidden layer."
        assert (
            min_hidden_layers < max_hidden_layers
        ), "'min_hidden_layers' must be less than 'max_hidden_layers."
        assert (
            min_mlp_nodes < max_mlp_nodes
        ), "'min_mlp_nodes' must be less than 'max_mlp_nodes."

        self.name = name
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self._activation = activation
        self.new_gelu = new_gelu
        self.output_activation = output_activation
        self.min_hidden_layers = min_hidden_layers
        self.max_hidden_layers = max_hidden_layers
        self.min_mlp_nodes = min_mlp_nodes
        self.max_mlp_nodes = max_mlp_nodes
        self.layer_norm = layer_norm
        self.output_vanish = output_vanish
        self.init_layers = init_layers
        self.hidden_size = hidden_size
        self.noisy = noisy
        self.noise_std = noise_std

        self.model = create_mlp(
            input_size=self.num_inputs,
            output_size=self.num_outputs,
            hidden_size=self.hidden_size,
            output_vanish=self.output_vanish,
            output_activation=self.output_activation,
            noisy=self.noisy,
            init_layers=self.init_layers,
            layer_norm=self.layer_norm,
            activation=self.activation,
            noise_std=self.noise_std,
            device=self.device,
            new_gelu=self.new_gelu,
            name=self.name,
        )

    @property
    def net_config(self) -> Dict[str, Any]:
        """Returns model configuration in dictionary."""
        net_config = self.init_dict.copy()
        for attr in ["num_inputs", "num_outputs", "device", "name"]:
            if attr in net_config:
                net_config.pop(attr)

        return net_config

    @property
    def activation(self) -> str:
        """Returns activation function."""
        return self._activation

    @activation.setter
    def activation(self, activation: str) -> None:
        """Set activation function."""
        self._activation = activation

    def init_weights_gaussian(
        self, std_coeff: float = 4, output_coeff: float = 4
    ) -> None:
        """Initialise weights of neural network using Gaussian distribution."""
        EvolvableModule.init_weights_gaussian(self.model, std_coeff=std_coeff)

        # Output layer is initialised with std_coeff=2
        output_layer = self.get_output_dense()
        EvolvableModule.init_weights_gaussian(output_layer, std_coeff=output_coeff)

    def forward(
        self, obs: Dict[str, ArrayOrTensor], actions: ArrayOrTensor
    ) -> torch.Tensor:
        """Returns output of neural network.

        :param x: Neural network input
        :type x: torch.Tensor()

        :return: Neural network output
        :rtype: torch.Tensor
        """
        if not isinstance(obs, dict):
            obs = {
                k: torch.tensor(v, dtype=torch.float32, device=self.device)
                for k, v in obs.items()
            }

        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)

        if len(next(iter(obs.values())).shape) == 1:
            obs = {k: v.unsqueeze(0) for k, v in obs.items()}

        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)

        x = torch.cat([torch.cat(list(obs.values()), dim=1), actions], dim=1)
        return self.model(x)

    def get_output_dense(self) -> torch.nn.Module:
        """Returns output layer of neural network."""
        return getattr(self.model, f"{self.name}_linear_layer_output")

    def change_activation(self, activation: str, output: bool = False) -> None:
        """Set the activation function for the network.

        :param activation: Activation function to use.
        :type activation: str
        :param output: Flag indicating whether to set the output activation function, defaults to False
        :type output: bool, optional
        """
        if output:
            self.output_activation = activation

        self.activation = activation
        self.recreate_network()

    @mutation(MutationType.LAYER)
    def add_layer(self) -> None:
        """Adds a hidden layer to neural network. Falls back on ``add_node()`` if
        max hidden layers reached."""
        # add layer to hyper params
        if len(self.hidden_size) < self.max_hidden_layers:  # HARD LIMIT
            self.hidden_size += [self.hidden_size[-1]]
        else:
            return self.add_node()

    @mutation(MutationType.LAYER)
    def remove_layer(self) -> None:
        """Removes a hidden layer from neural network. Falls back on ``add_node()`` if
        min hidden layers reached."""
        if len(self.hidden_size) > self.min_hidden_layers:  # HARD LIMIT
            self.hidden_size = self.hidden_size[:-1]
        else:
            return self.add_node()

    @mutation(MutationType.NODE)
    def add_node(
        self, hidden_layer: Optional[int] = None, numb_new_nodes: Optional[int] = None
    ) -> Dict[str, int]:
        """Adds nodes to hidden layer of neural network.

        :param hidden_layer: Depth of hidden layer to add nodes to, defaults to None
        :type hidden_layer: int, optional
        :param numb_new_nodes: Number of nodes to add to hidden layer, defaults to None
        :type numb_new_nodes: int, optional
        """
        if hidden_layer is None:
            hidden_layer = np.random.randint(0, len(self.hidden_size), 1)[0]
        else:
            hidden_layer = min(hidden_layer, len(self.hidden_size) - 1)

        if numb_new_nodes is None:
            numb_new_nodes = np.random.choice([16, 32, 64], 1)[0]

        if (
            self.hidden_size[hidden_layer] + numb_new_nodes <= self.max_mlp_nodes
        ):  # HARD LIMIT
            self.hidden_size[hidden_layer] += numb_new_nodes

        return {"hidden_layer": hidden_layer, "numb_new_nodes": numb_new_nodes}

    @mutation(MutationType.NODE)
    def remove_node(
        self, hidden_layer: Optional[int] = None, numb_new_nodes: Optional[int] = None
    ) -> Dict[str, int]:
        """Removes nodes from hidden layer of neural network.

        :param hidden_layer: Depth of hidden layer to remove nodes from, defaults to None
        :type hidden_layer: int, optional
        :param numb_new_nodes: Number of nodes to remove from hidden layer, defaults to None
        :type numb_new_nodes: int, optional
        """
        if hidden_layer is None:
            hidden_layer = np.random.randint(0, len(self.hidden_size), 1)[0]
        else:
            hidden_layer = min(hidden_layer, len(self.hidden_size) - 1)

        if numb_new_nodes is None:
            numb_new_nodes = np.random.choice([16, 32, 64], 1)[0]

        # HARD LIMIT
        if self.hidden_size[hidden_layer] - numb_new_nodes > self.min_mlp_nodes:
            self.hidden_size[hidden_layer] -= numb_new_nodes

        return {"hidden_layer": hidden_layer, "numb_new_nodes": numb_new_nodes}

    def recreate_network(self) -> None:
        """Recreates neural networks.

        :param shrink_params: Shrink parameters of neural networks, defaults to False
        :type shrink_params: bool, optional
        """
        model = create_mlp(
            input_size=self.num_inputs,
            output_size=self.num_outputs,
            hidden_size=self.hidden_size,
            output_vanish=self.output_vanish,
            output_activation=self.output_activation,
            noisy=self.noisy,
            init_layers=self.init_layers,
            layer_norm=self.layer_norm,
            activation=self.activation,
            noise_std=self.noise_std,
            new_gelu=self.new_gelu,
            device=self.device,
            name=self.name,
        )

        self.model = EvolvableModule.preserve_parameters(
            old_net=self.model, new_net=model
        )
