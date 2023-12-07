import copy
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agilerl.networks.custom_components import GumbelSoftmax, NoisyLinear


class EvolvableCNN(nn.Module):
    """The Evolvable Convolutional Neural Network class.

    :param input_shape: Input shape
    :type input_shape: list[int]
    :param channel_size: CNN channel size
    :type channel_size: list[int]
    :param kernel_size: Comvolution kernel size
    :type kernel_size: list[int]
    :param stride_size: Convolution stride size
    :type stride_size: list[int]
    :param hidden_size: Hidden layer(s) size
    :type hidden_size: list[int]
    :param num_actions: Action dimension
    :type num_actions: int
    :param num_atoms: Number of atoms for Rainbow DQN, defaults to 50
    :type num_atoms: int, optional
    :param mlp_output_activation: MLP output activation layer, defaults to 'softmax'
    :type mlp_output_activation: str, optional
    :param mlp_activation: MLP activation layer, defaults to 'relu'
    :type mlp_activation: str, optional
    :param cnn_activation: CNN activation layer, defaults to 'relu'
    :type cnn_activation: str, optional
    :param min_hidden_layers: Minimum number of hidden layers the fully connected layer will shrink down to, defaults to 1
    :type min_hidden_layers: int, optional
    :param max_hidden_layers: Maximum number of hidden layers the fully connected layer will expand to, defaults to 3
    :type max_hidden_layers: int, optional
    :param min_mlp_nodes: Minimum number of nodes a layer can have within the fully connected layer, defaults to 64
    :type min_mlp_nodes: int, optional
    :param max_mlp_nodes: Maximum number of nodes a layer can have within the fully connected layer, defaults to 1024
    :type max_mlp_nodes: int, optional
    :param min_cnn_hidden_layers: Minimum number of hidden layers the convolutional layer will shrink down to, defaults to 1
    :type min_cnn_hidden_layers: int, optional
    :param max_cnn_hidden_layers: Maximum number of hidden layers the convolutional layer will expand to, defaults to 6
    :type max_cnn_hidden_layers: int, optional
    :param min_channel_size: Minimum number of channels a convolutional layer can have, defaults to 32
    :type min_channel_size: int, optional
    :param max_channel_size: Maximum number of channels a convolutional layer can have, defaults to 256
    :type max_channel_size: int, optional
    :param n_agents: Number of agents, defaults to None
    :type n_agents: int, optional
    :param multi: Boolean flag to indicate if this is a multi-agent problem, defaults to False
    :type multi: bool, optional
    :param layer_norm: Normalization between layers, defaults to False
    :type layer_norm: bool, optional
    :param support: Atoms support tensor, defaults to None
    :type support: torch.Tensor(), optional
    :param rainbow: Using Rainbow DQN, defaults to False
    :type rainbow: bool, optional
    :param critic: CNN is a critic network, defaults to False
    :type critic: bool, optional
    :param normalize: Normalize CNN inputs, defaults to True
    :type normalize: bool, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    """

    def __init__(
        self,
        input_shape: List[int],
        channel_size: List[int],
        kernel_size: List[int],
        stride_size: List[int],
        hidden_size: List[int],
        num_actions: int,
        num_atoms=51,
        mlp_output_activation="Softmax",
        mlp_activation="ReLU",
        cnn_activation="ReLU",
        min_hidden_layers=1,
        max_hidden_layers=3,
        min_mlp_nodes=64,
        max_mlp_nodes=1024,
        min_cnn_hidden_layers=1,
        max_cnn_hidden_layers=6,
        min_channel_size=32,
        max_channel_size=256,
        n_agents=None,
        multi=False,
        layer_norm=False,
        support=None,
        rainbow=False,
        critic=False,
        normalize=True,
        device="cpu",
        accelerator=None,
    ):
        super().__init__()
        assert len(kernel_size) == len(
            channel_size
        ), "Length of kernel size list must be the same length as channel size list."
        assert len(stride_size) == len(
            channel_size
        ), "Length of stride size list must be the same length as channel size list."
        assert len(input_shape) >= 3, "Input shape must have at least 3 dimensions."
        assert (
            len(hidden_size) > 0
        ), "Fully connected layer must contain at least one hidden layer."
        assert (
            num_actions > 0
        ), "'num_actions' cannot be less than or equal to zero, please enter a valid integer."
        if multi:
            assert (
                n_agents is not None
            ), "'multi' set as True, specify the number of agents (n_agents) too."
        if n_agents is not None:
            assert (
                multi
            ), f"'n_agents' has been set to {n_agents} implying a multi-agent system, please also specify 'multi' as True."

        assert (
            min_hidden_layers < max_hidden_layers
        ), "'min_hidden_layers' must be less than 'max_hidden_layers."
        assert (
            min_mlp_nodes < max_mlp_nodes
        ), "'min_mlp_nodes' must be less than 'max_mlp_nodes."
        assert (
            min_cnn_hidden_layers < max_cnn_hidden_layers
        ), "'min_cnn_hidden_layers' must be less than 'max_cnn_hidden_layers."
        assert (
            min_channel_size < max_channel_size
        ), "'min_channel_size' must be less than 'max_channel_size'."

        self.input_shape = input_shape
        self.channel_size = channel_size
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.mlp_activation = mlp_activation
        self.mlp_output_activation = mlp_output_activation
        self.cnn_activation = cnn_activation
        self.min_hidden_layers = min_hidden_layers
        self.max_hidden_layers = max_hidden_layers
        self.min_mlp_nodes = min_mlp_nodes
        self.max_mlp_nodes = max_mlp_nodes
        self.min_cnn_hidden_layers = min_cnn_hidden_layers
        self.max_cnn_hidden_layers = max_cnn_hidden_layers
        self.min_channel_size = min_channel_size
        self.max_channel_size = max_channel_size
        self.layer_norm = layer_norm
        self.support = support
        self.rainbow = rainbow
        self.critic = critic
        self.normalize = normalize
        self.device = device
        self.accelerator = accelerator
        self.multi = multi
        self.n_agents = n_agents

        self.net = self.create_nets()
        self.feature_net, self.value_net, self.advantage_net = self.create_nets()

    def get_activation(self, activation_names):
        """Returns activation function for corresponding activation name.

        :param activation_names: Activation function name
        :type activation_names: str
        """
        activation_functions = {
            "Tanh": nn.Tanh,
            "Identity": nn.Identity,
            "GELU": nn.GELU,
            "ReLU": nn.ReLU,
            "ELU": nn.ELU,
            "Softsign": nn.Softsign,
            "Sigmoid": nn.Sigmoid,
            "Softmax": nn.Softmax,
            "GumbelSoftmax": GumbelSoftmax,
            "Softplus": nn.Softplus,
            "LeakyReLU": nn.LeakyReLU,
            "PReLU": nn.PReLU,
        }
        return (
            activation_functions[activation_names](dim=-1)
            if activation_names == "softmax"
            else activation_functions[activation_names]()
        )

    def create_mlp(
        self,
        input_size,
        output_size,
        hidden_size,
        name,
        output_activation,
        noisy=False,
    ):
        """Creates and returns multi-layer perceptron."""
        net_dict = OrderedDict()
        if noisy:
            net_dict[f"{name}_linear_layer_0"] = NoisyLinear(input_size, hidden_size[0])
        else:
            net_dict[f"{name}_linear_layer_0"] = nn.Linear(input_size, hidden_size[0])
        if self.layer_norm:
            net_dict[f"{name}_layer_norm_0"] = nn.LayerNorm(hidden_size[0])
        net_dict["activation_0"] = self.get_activation(self.mlp_activation)
        if len(hidden_size) > 1:
            for l_no in range(1, len(hidden_size)):
                if noisy:
                    net_dict[f"{name}_linear_layer_{str(l_no)}"] = NoisyLinear(
                        hidden_size[l_no - 1], hidden_size[l_no]
                    )
                else:
                    net_dict[f"{name}_linear_layer_{str(l_no)}"] = nn.Linear(
                        hidden_size[l_no - 1], hidden_size[l_no]
                    )
                if self.layer_norm:
                    net_dict[f"{name}_layer_norm_{str(l_no)}"] = nn.LayerNorm(
                        hidden_size[l_no]
                    )
                net_dict[f"{name}_activation_{str(l_no)}"] = self.get_activation(
                    self.mlp_activation
                )
        if noisy:
            output_layer = NoisyLinear(hidden_size[-1], output_size)
        else:
            output_layer = nn.Linear(hidden_size[-1], output_size)
        net_dict[f"{name}_linear_layer_output"] = output_layer
        if output_activation is not None:
            net_dict[f"{name}_activation_output"] = self.get_activation(
                output_activation
            )
        net = nn.Sequential(net_dict)
        return net

    def create_cnn(self, input_size, channel_size, kernel_size, stride_size, name):
        """Creates and returns convolutional neural network."""
        if self.multi:
            net_dict = OrderedDict()
            net_dict[f"{name}_conv_layer_0"] = nn.Conv3d(
                in_channels=input_size,
                out_channels=channel_size[0],
                kernel_size=kernel_size[0],
                stride=stride_size[
                    0
                ],  ## Maybe include the ability to have 3 dim kernel and stride
            )
            if self.layer_norm:
                net_dict[f"{name}_layer_norm_0"] = nn.BatchNorm3d(channel_size[0])
            net_dict[f"{name}_activation_0"] = self.get_activation(self.cnn_activation)

            if len(channel_size) > 1:
                for l_no in range(1, len(channel_size)):
                    net_dict[f"{name}_conv_layer_{str(l_no)}"] = nn.Conv3d(
                        in_channels=channel_size[l_no - 1],
                        out_channels=channel_size[l_no],
                        kernel_size=kernel_size[l_no],
                        stride=stride_size[l_no],
                    )
                    if self.layer_norm:
                        net_dict[f"{name}_layer_norm_{str(l_no)}"] = nn.BatchNorm3d(
                            channel_size[l_no]
                        )
                    net_dict[f"{name}_activation_{str(l_no)}"] = self.get_activation(
                        self.cnn_activation
                    )
        else:
            net_dict = OrderedDict()
            net_dict[f"{name}_conv_layer_0"] = nn.Conv2d(
                in_channels=input_size,
                out_channels=channel_size[0],
                kernel_size=kernel_size[0],
                stride=stride_size[0],
            )
            if self.layer_norm:
                net_dict[f"{name}_layer_norm_0"] = nn.BatchNorm2d(channel_size[0])
            net_dict[f"{name}_activation_0"] = self.get_activation(self.cnn_activation)

            if len(channel_size) > 1:
                for l_no in range(1, len(channel_size)):
                    net_dict[f"{name}_conv_layer_{str(l_no)}"] = nn.Conv2d(
                        in_channels=channel_size[l_no - 1],
                        out_channels=channel_size[l_no],
                        kernel_size=kernel_size[l_no],
                        stride=stride_size[l_no],
                    )
                    if self.layer_norm:
                        net_dict[f"{name}_layer_norm_{str(l_no)}"] = nn.BatchNorm2d(
                            channel_size[l_no]
                        )
                    net_dict[f"{name}_activation_{str(l_no)}"] = self.get_activation(
                        self.cnn_activation
                    )

        return nn.Sequential(net_dict)

    def create_nets(self):
        """Creates and returns neural networks."""

        feature_net = self.create_cnn(
            self.input_shape[0],
            self.channel_size,
            self.kernel_size,
            self.stride_size,
            name="feature",
        )

        with torch.no_grad():
            if self.multi:
                if self.critic:
                    critic_input = torch.stack(
                        (
                            torch.zeros(1, *self.input_shape),
                            torch.zeros(1, *self.input_shape),
                        ),
                        dim=2,
                    )
                    input_size = feature_net(critic_input).view(1, -1).size(1)
                else:
                    input_size = (
                        feature_net(torch.zeros(1, *self.input_shape).unsqueeze(2))
                        .view(1, -1)
                        .size(1)
                    )
            else:
                input_size = (
                    feature_net(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
                )

        if self.critic:
            input_size += self.num_actions

        if self.rainbow:
            value_net = self.create_mlp(
                input_size,
                output_size=self.num_atoms,
                hidden_size=self.hidden_size,
                name="value",
                output_activation=None,
                noisy=True,
            )
            advantage_net = self.create_mlp(
                input_size,
                output_size=self.num_atoms * self.num_actions,
                hidden_size=self.hidden_size,
                name="advantage",
                output_activation=None,
                noisy=True,
            )
            if self.accelerator is not None:
                feature_net, value_net, advantage_net = self.accelerator.prepare(
                    feature_net, value_net, advantage_net
                )
            else:
                feature_net, value_net, advantage_net = (
                    feature_net.to(self.device),
                    value_net.to(self.device),
                    advantage_net.to(self.device),
                )
        else:
            if self.critic:
                value_net = self.create_mlp(
                    input_size,
                    output_size=1,
                    hidden_size=self.hidden_size,
                    name="value",
                    output_activation=self.mlp_output_activation,
                )
            else:
                value_net = self.create_mlp(
                    input_size,
                    output_size=self.num_actions,
                    hidden_size=self.hidden_size,
                    name="value",
                    output_activation=self.mlp_output_activation,
                )
            advantage_net = None
            if self.accelerator is None:
                (
                    feature_net,
                    value_net,
                ) = feature_net.to(
                    self.device
                ), value_net.to(self.device)

        return feature_net, value_net, advantage_net

    def reset_noise(self):
        """Resets noise of value and advantage networks."""
        for layer in self.value_net:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        if self.rainbow:
            for layer in self.advantage_net:
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()

    def forward(self, x, xc=None, q=True):
        """Returns output of neural network.

        :param x: Neural network input
        :type x: torch.Tensor() or np.array
        :param xc: Actions to be evaluated by critic, defaults to None
        :type xc: torch.Tensor() or np.array, optional
        :param q: Return Q value if using rainbow, defaults to True
        :type q: bool, optional
        """
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
            x = x.to(self.device)

        if x.dtype != torch.float32:
            x = x.type(torch.float32)

        batch_size = x.size(0)

        if self.normalize:
            x = x / 255.0

        x = self.feature_net(x)
        x = x.reshape(batch_size, -1)

        if self.critic:
            x = torch.cat([x, xc], dim=1)

        value = self.value_net(x)

        if self.rainbow:
            advantage = self.advantage_net(x)

            value = value.view(batch_size, 1, self.num_atoms)
            advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)

            x = value + advantage - advantage.mean(1, keepdim=True)
            x = F.softmax(x.view(-1, self.num_atoms), dim=-1).view(
                -1, self.num_actions, self.num_atoms
            )
            x = x.clamp(min=1e-3)

            if q:
                x = torch.sum(x * self.support, dim=2)

        else:
            x = value

        return x

    @property
    def init_dict(self):
        """Returns model information in dictionary."""
        init_dict = {
            "input_shape": self.input_shape,
            "channel_size": self.channel_size,
            "kernel_size": self.kernel_size,
            "stride_size": self.stride_size,
            "hidden_size": self.hidden_size,
            "num_actions": self.num_actions,
            "n_agents": self.n_agents,
            "num_atoms": self.num_atoms,
            "support": self.support,
            "normalize": self.normalize,
            "mlp_activation": self.mlp_activation,
            "cnn_activation": self.cnn_activation,
            "mlp_output_activation": self.mlp_output_activation,
            "min_hidden_layers": self.min_hidden_layers,
            "max_hidden_layers": self.max_hidden_layers,
            "min_mlp_nodes": self.min_mlp_nodes,
            "max_mlp_nodes": self.max_mlp_nodes,
            "min_cnn_hidden_layers": self.min_cnn_hidden_layers,
            "max_cnn_hidden_layers": self.max_cnn_hidden_layers,
            "min_channel_size": self.min_channel_size,
            "max_channel_size": self.max_channel_size,
            "multi": self.multi,
            "layer_norm": self.layer_norm,
            "critic": self.critic,
            "rainbow": self.rainbow,
            "device": self.device,
            "accelerator": self.accelerator,
        }
        return init_dict

    def add_mlp_layer(self):
        """Adds a hidden layer to fully connected layer."""
        if len(self.hidden_size) < self.max_hidden_layers:  # HARD LIMIT
            self.hidden_size += [self.hidden_size[-1]]
            self.recreate_nets()
        else:
            self.add_mlp_node()

    def remove_mlp_layer(self):
        """Removes a hidden layer from fully connected layer."""
        if len(self.hidden_size) > self.min_hidden_layers:
            self.hidden_size = self.hidden_size[:-1]
            self.recreate_nets(shrink_params=True)
        else:
            self.add_mlp_node()

    def add_mlp_node(self, hidden_layer=None, numb_new_nodes=None):
        """Adds nodes to hidden layer of Multi-layer Perceptron.

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
            numb_new_nodes = np.random.choice([32, 64, 128], 1)[0]

        if (
            self.hidden_size[hidden_layer] + numb_new_nodes <= self.max_mlp_nodes
        ):  # HARD LIMIT
            self.hidden_size[hidden_layer] += numb_new_nodes

            self.recreate_nets()
        return {"hidden_layer": hidden_layer, "numb_new_nodes": numb_new_nodes}

    def remove_mlp_node(self, hidden_layer=None, numb_new_nodes=None):
        """Removes nodes from hidden layer of fully connected layer.
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

        if (
            self.hidden_size[hidden_layer] - numb_new_nodes > self.min_mlp_nodes
        ):  # HARD LIMIT
            self.hidden_size[hidden_layer] = (
                self.hidden_size[hidden_layer] - numb_new_nodes
            )
            self.recreate_nets(shrink_params=True)

        return {"hidden_layer": hidden_layer, "numb_new_nodes": numb_new_nodes}

    def add_cnn_layer(self):
        """Adds a hidden layer to convolutional neural network."""
        if len(self.channel_size) < self.max_cnn_hidden_layers:  # HARD LIMIT
            self.channel_size += [self.channel_size[-1]]
            if self.multi:
                self.kernel_size += [(1, 3, 3)]
            else:
                self.kernel_size += [(3, 3)]
            stride_size_list = [
                [4],
                [4, 2],
                [4, 2, 1],
                [2, 2, 2, 1],
                [2, 1, 2, 1, 2],
                [2, 1, 2, 1, 2, 1],
            ]
            self.stride_size = stride_size_list[len(self.channel_size) - 1]
            self.recreate_nets()
        else:
            self.add_cnn_channel()

    def remove_cnn_layer(self):
        """Removes a hidden layer from convolutional neural network."""
        if len(self.channel_size) > self.min_cnn_hidden_layers:
            self.channel_size = self.channel_size[:-1]
            self.kernel_size = self.kernel_size[:-1]
            stride_size_list = [
                [4],
                [4, 2],
                [4, 2, 1],
                [2, 2, 2, 1],
                [2, 1, 2, 1, 2],
                [2, 1, 2, 1, 2, 1],
            ]
            self.stride_size = stride_size_list[len(self.channel_size) - 1]
            self.recreate_nets(shrink_params=True)
        else:
            self.add_cnn_channel()

    def change_cnn_kernel(self):
        """Randomly alters convolution kernel of random CNN layer."""

        if self.multi:
            if len(self.channel_size) > 1:
                hidden_layer = np.random.randint(1, min(4, len(self.channel_size)), 1)[
                    0
                ]
                kernel_size_value = np.random.choice([3, 4, 5, 7])
                if self.critic:
                    self.kernel_size[hidden_layer] = tuple(
                        min(kernel_size_value, self.n_agents - 1)
                        if idx == 0
                        else kernel_size_value
                        for idx in range(3)
                    )
                else:
                    self.kernel_size[hidden_layer] = tuple(
                        1 if idx == 0 else kernel_size_value for idx in range(3)
                    )
                self.recreate_nets()
            else:
                self.add_cnn_layer()
        else:
            if len(self.channel_size) > 1:
                hidden_layer = np.random.randint(1, min(4, len(self.channel_size)), 1)[
                    0
                ]
                self.kernel_size[hidden_layer] = np.random.choice([3, 4, 5, 7])

                self.recreate_nets()
            else:
                self.add_cnn_layer()

    def add_cnn_channel(self, hidden_layer=None, numb_new_channels=None):
        """Adds channel to hidden layer of convolutional neural network.

        :param hidden_layer: Depth of hidden layer to add channel to, defaults to None
        :type hidden_layer: int, optional
        :param numb_new_channels: Number of channels to add to hidden layer, defaults to None
        :type numb_new_channels: int, optional
        """
        if hidden_layer is None:
            hidden_layer = np.random.randint(0, len(self.channel_size), 1)[0]
        else:
            hidden_layer = min(hidden_layer, len(self.channel_size) - 1)
        if numb_new_channels is None:
            numb_new_channels = np.random.choice([8, 16, 32], 1)[0]

        if (
            self.channel_size[hidden_layer] + numb_new_channels <= self.max_channel_size
        ):  # HARD LIMIT
            self.channel_size[hidden_layer] += numb_new_channels

            self.recreate_nets()

        return {"hidden_layer": hidden_layer, "numb_new_channels": numb_new_channels}

    def remove_cnn_channel(self, hidden_layer=None, numb_new_channels=None):
        """Remove channel from hidden layer of convolutional neural network.

        :param hidden_layer: Depth of hidden layer to add channel to, defaults to None
        :type hidden_layer: int, optional
        :param numb_new_channels: Number of channels to add to hidden layer, defaults to None
        :type numb_new_channels: int, optional
        """
        if hidden_layer is None:
            hidden_layer = np.random.randint(0, len(self.channel_size), 1)[0]
        else:
            hidden_layer = min(hidden_layer, len(self.channel_size) - 1)
        if numb_new_channels is None:
            numb_new_channels = np.random.choice([8, 16, 32], 1)[0]

        if (
            self.channel_size[hidden_layer] - numb_new_channels > self.min_channel_size
        ):  # HARD LIMIT
            self.channel_size[hidden_layer] -= numb_new_channels

            self.recreate_nets(shrink_params=True)

        return {"hidden_layer": hidden_layer, "numb_new_channels": numb_new_channels}

    def recreate_nets(self, shrink_params=False):
        """Recreates neural networks."""
        new_feature_net, new_value_net, new_advantage_net = self.create_nets()
        if shrink_params:
            new_feature_net = self.shrink_preserve_parameters(
                old_net=self.feature_net, new_net=new_feature_net
            )
            new_value_net = self.shrink_preserve_parameters(
                old_net=self.value_net, new_net=new_value_net
            )
            if self.rainbow:
                new_advantage_net = self.shrink_preserve_parameters(
                    old_net=self.advantage_net, new_net=new_advantage_net
                )
        else:
            new_feature_net = self.preserve_parameters(
                old_net=self.feature_net, new_net=new_feature_net
            )
            new_value_net = self.preserve_parameters(
                old_net=self.value_net, new_net=new_value_net
            )
            if self.rainbow:
                new_advantage_net = self.preserve_parameters(
                    old_net=self.advantage_net, new_net=new_advantage_net
                )
        self.feature_net, self.value_net, self.advantage_net = (
            new_feature_net,
            new_value_net,
            new_advantage_net,
        )

    def clone(self):
        """Returns clone of neural net with identical parameters."""
        clone = EvolvableCNN(**copy.deepcopy(self.init_dict))
        clone.load_state_dict(self.state_dict())
        clone.rainbow = self.rainbow
        clone.critic = self.critic
        return clone

    def preserve_parameters(self, old_net, new_net):
        """Returns new neural network with copied parameters from old network.

        :param old_net: Old neural network
        :type old_net: nn.Module()
        :param new_net: New neural network
        :type new_net: nn.Module()
        """
        old_net_dict = dict(old_net.named_parameters())

        for key, param in new_net.named_parameters():
            if key in old_net_dict.keys():
                if old_net_dict[key].data.size() == param.data.size():
                    param.data = old_net_dict[key].data
                else:
                    if "norm" not in key:
                        old_size = old_net_dict[key].data.size()
                        new_size = param.data.size()
                        if len(param.data.size()) == 1:
                            param.data[: min(old_size[0], new_size[0])] = old_net_dict[
                                key
                            ].data[: min(old_size[0], new_size[0])]
                        elif len(param.data.size()) == 2:
                            param.data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                            ] = old_net_dict[key].data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                            ]
                        elif len(param.data.size()) == 3:
                            param.data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                                : min(old_size[2], new_size[2]),
                            ] = old_net_dict[key].data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                                : min(old_size[2], new_size[2]),
                            ]
                        elif len(param.data.size()) == 4:
                            param.data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                                : min(old_size[2], new_size[2]),
                                : min(old_size[3], new_size[3]),
                            ] = old_net_dict[key].data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                                : min(old_size[2], new_size[2]),
                                : min(old_size[3], new_size[3]),
                            ]
                        elif len(param.data.size()) == 5:
                            param.data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                                : min(old_size[2], new_size[2]),
                                : min(old_size[3], new_size[3]),
                                : min(old_size[4], new_size[4]),
                            ] = old_net_dict[key].data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                                : min(old_size[2], new_size[2]),
                                : min(old_size[3], new_size[3]),
                                : min(old_size[4], new_size[4]),
                            ]

        return new_net

    def shrink_preserve_parameters(self, old_net, new_net):
        """Returns shrunk new neural network with copied parameters from old network.

        :param old_net: Old neural network
        :type old_net: nn.Module()
        :param new_net: New neural network
        :type new_net: nn.Module()
        """
        old_net_dict = dict(old_net.named_parameters())

        for key, param in new_net.named_parameters():
            if key in old_net_dict.keys():
                if old_net_dict[key].data.size() == param.data.size():
                    param.data = old_net_dict[key].data
                else:
                    if "norm" not in key:
                        old_size = old_net_dict[key].data.size()
                        new_size = param.data.size()
                        min_0 = min(old_size[0], new_size[0])
                        if len(param.data.size()) == 1:
                            param.data[:min_0] = old_net_dict[key].data[:min_0]
                        else:
                            min_1 = min(old_size[1], new_size[1])
                            param.data[:min_0, :min_1] = old_net_dict[key].data[
                                :min_0, :min_1
                            ]
        return new_net
