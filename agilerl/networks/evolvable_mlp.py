import copy
import math
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agilerl.networks.custom_activation import GumbelSoftmax


class NoisyLinear(nn.Module):
    """The Noisy Linear Neural Network class.

    :param in_features: Input features size
    :type in_features: int
    :param out_features: Output features size
    :type out_features: int
    :param std_init: Standard deviation, defaults to 0.4
    :type std_init: float, optional
    """

    def __init__(self, in_features, out_features, std_init=0.4):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer(
            "weight_epsilon", torch.FloatTensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        """Returns output of neural network.

        :param x: Neural network input
        :type x: torch.Tensor()
        """
        weight_epsilon = self.weight_epsilon.to(x.device)
        bias_epsilon = self.bias_epsilon.to(x.device)

        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        """Resets neural network parameters."""
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.weight_sigma.size(1))
        )

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        """Resets neural network noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        """Returns noisy tensor.

        :param size: Tensor of same size as noisy output
        :type size: torch.Tensor()
        """
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class EvolvableMLP(nn.Module):
    """The Evolvable Multi-layer Perceptron class.

    :param num_inputs: Input layer dimension
    :type num_inputs: int
    :param num_outputs: Output layer dimension
    :type num_outputs: int
    :param hidden_size: Hidden layer(s) size
    :type hidden_size: list[int]
    :param num_atoms: Number of atoms for Rainbow DQN, defaults to 50
    :type num_atoms: int, optional
    :param mlp_activation: Activation layer, defaults to 'relu'
    :type mlp_activation: str, optional
    :param mlp_output_activation: Output activation layer, defaults to None
    :type mlp_output_activation: str, optional
    :param min_hidden_layers: Minimum number of hidden layers the network will shrink down to, defaults to 1
    :type min_hidden_layers: int, optional
    :param max_hidden_layers: Maximum number of hidden layers the network will expand to, defaults to 3
    :type max_hidden_layers: int, optional
    :param min_mlp_nodes: Minimum number of nodes a layer can have within the network, defaults to 64
    :type min_mlp_nodes: int, optional
    :param max_mlp_nodes: Maximum number of nodes a layer can have within the network, defaults to 500
    :type max_mlp_nodes: int, optional
    :param layer_norm: Normalization between layers, defaults to True
    :type layer_norm: bool, optional
    :param output_vanish: Vanish output by multiplying by 0.1, defaults to True
    :type output_vanish: bool, optional
    :param init_layers: Initialise network layers, defaults to True
    :type init_layers: bool, optional
    :param support: Atoms support tensor, defaults to None
    :type support: torch.Tensor(), optional
    :param rainbow: Using Rainbow DQN, defaults to False
    :type rainbow: bool, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        hidden_size: List[int],
        num_atoms=50,
        mlp_activation="ReLU",
        mlp_output_activation=None,
        min_hidden_layers=1,
        max_hidden_layers=3,
        min_mlp_nodes=64,
        max_mlp_nodes=500,
        layer_norm=True,
        output_vanish=True,
        init_layers=True,
        support=None,
        rainbow=False,
        device="cpu",
        accelerator=None,
    ):
        super().__init__()

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

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.mlp_activation = mlp_activation
        self.mlp_output_activation = mlp_output_activation
        self.min_hidden_layers = min_hidden_layers
        self.max_hidden_layers = max_hidden_layers
        self.min_mlp_nodes = min_mlp_nodes
        self.max_mlp_nodes = max_mlp_nodes
        self.layer_norm = layer_norm
        self.output_vanish = False if rainbow else output_vanish
        self.init_layers = False if rainbow else init_layers
        self.hidden_size = hidden_size
        self.num_atoms = num_atoms
        self.support = support
        self.rainbow = rainbow
        self.device = device
        self.accelerator = accelerator

        self.feature_net, self.value_net, self.advantage_net = self.create_net()

    def get_activation(self, activation_names):
        """Returns activation function for corresponding activation name.

        :param activation_names: Activation function name
        :type activation_names: str
        """
        activation_functions = {
            "Tanh": nn.Tanh,
            "Identity": nn.Identity,
            "ReLU": nn.ReLU,
            "ELU": nn.ELU,
            "Softsign": nn.Softsign,
            "Sigmoid": nn.Sigmoid,
            "GumbelSoftmax": GumbelSoftmax,
            "Softplus": nn.Softplus,
            "Softmax": nn.Softmax,
            "LeakyReLU": nn.LeakyReLU,
            "PReLU": nn.PReLU,
            "GELU": nn.GELU,
        }

        return (
            activation_functions[activation_names](dim=-1)
            if activation_names == "Softmax"
            else activation_functions[activation_names]()
        )

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def create_mlp(
        self,
        input_size,
        output_size,
        hidden_size,
        output_vanish,
        output_activation,
        noisy=False,
    ):
        """Creates and returns multi-layer perceptron."""
        net_dict = OrderedDict()
        if noisy:
            net_dict["linear_layer_0"] = NoisyLinear(input_size, hidden_size[0])
        else:
            net_dict["linear_layer_0"] = nn.Linear(input_size, hidden_size[0])
        if self.init_layers:
            net_dict["linear_layer_0"] = self.layer_init(net_dict["linear_layer_0"])
        if self.layer_norm:
            net_dict["layer_norm_0"] = nn.LayerNorm(hidden_size[0])
        net_dict["activation_0"] = self.get_activation(self.mlp_activation)
        if len(hidden_size) > 1:
            for l_no in range(1, len(hidden_size)):
                if noisy:
                    net_dict[f"linear_layer_{str(l_no)}"] = NoisyLinear(
                        hidden_size[l_no - 1], hidden_size[l_no]
                    )
                else:
                    net_dict[f"linear_layer_{str(l_no)}"] = nn.Linear(
                        hidden_size[l_no - 1], hidden_size[l_no]
                    )
                if self.init_layers:
                    net_dict[f"linear_layer_{str(l_no)}"] = self.layer_init(
                        net_dict[f"linear_layer_{str(l_no)}"]
                    )
                if self.layer_norm:
                    net_dict[f"layer_norm_{str(l_no)}"] = nn.LayerNorm(
                        hidden_size[l_no]
                    )
                net_dict[f"activation_{str(l_no)}"] = self.get_activation(
                    self.mlp_activation
                )
        if noisy:
            output_layer = NoisyLinear(hidden_size[-1], output_size)
        else:
            output_layer = nn.Linear(hidden_size[-1], output_size)
        if self.init_layers:
            output_layer = self.layer_init(output_layer)
        if output_vanish:
            output_layer.weight.data.mul_(0.1)
            output_layer.bias.data.mul_(0.1)
        net_dict["linear_layer_output"] = output_layer
        if output_activation is not None:
            net_dict["activation_output"] = self.get_activation(output_activation)
        net = nn.Sequential(net_dict)
        return net

    def create_net(self):
        """Creates and returns neural network."""
        feature_net = self.create_mlp(
            input_size=self.num_inputs,
            output_size=self.hidden_size[-1] if self.rainbow else self.num_outputs,
            hidden_size=self.hidden_size,
            output_vanish=False,
            output_activation=self.mlp_output_activation,
        )
        if self.accelerator is None:
            feature_net = feature_net.to(self.device)

        value_net, advantage_net = None, None
        if self.rainbow:
            value_net = self.create_mlp(
                input_size=self.hidden_size[-1],
                output_size=self.num_atoms,
                hidden_size=[self.hidden_size[-1]],
                output_vanish=self.output_vanish,
                output_activation=None,
                noisy=True,
            )
            advantage_net = self.create_mlp(
                input_size=self.hidden_size[-1],
                output_size=self.num_atoms * self.num_outputs,
                hidden_size=[self.hidden_size[-1]],
                output_vanish=self.output_vanish,
                output_activation=None,
                noisy=True,
            )
            if self.accelerator is None:
                self.value_net, self.advantage_net = (
                    value_net.to(self.device),
                    advantage_net.to(self.device),
                )

        return feature_net, value_net, advantage_net

    def reset_noise(self):
        """Resets noise of value and advantage networks."""
        if self.rainbow:
            for layer in self.value_net:
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()
            for layer in self.advantage_net:
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()

    def forward(self, x, q=True):
        """Returns output of neural network.

        :param x: Neural network input
        :type x: torch.Tensor() or np.array
        :param q: Return Q value if using rainbow, defaults to True
        :type q: bool, optional
        """
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(np.array(x))
            if self.accelerator is None:
                x = x.to(self.device)

        batch_size = x.size(0)

        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = self.feature_net(x)

        if self.rainbow:
            value = self.value_net(x)
            advantage = self.advantage_net(x)

            value = value.view(batch_size, 1, self.num_atoms)
            advantage = advantage.view(batch_size, self.num_outputs, self.num_atoms)

            x = value + advantage - advantage.mean(1, keepdim=True)
            x = F.softmax(x.view(-1, self.num_atoms), dim=-1).view(
                -1, self.num_outputs, self.num_atoms
            )
            x = x.clamp(min=1e-3)

            if q:
                x = torch.sum(x * self.support, dim=2)

        return x

    @property
    def init_dict(self):
        """Returns model information in dictionary."""
        init_dict = {
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "hidden_size": self.hidden_size,
            "num_atoms": self.num_atoms,
            "mlp_activation": self.mlp_activation,
            "mlp_output_activation": self.mlp_output_activation,
            "min_hidden_layers": self.min_hidden_layers,
            "max_hidden_layers": self.max_hidden_layers,
            "min_mlp_nodes": self.min_mlp_nodes,
            "max_mlp_nodes": self.max_mlp_nodes,
            "layer_norm": self.layer_norm,
            "init_layers": self.init_layers,
            "output_vanish": self.output_vanish,
            "support": self.support,
            "rainbow": self.rainbow,
            "device": self.device,
            "accelerator": self.accelerator,
        }
        return init_dict

    def add_mlp_layer(self):
        """Adds a hidden layer to neural network."""
        # add layer to hyper params
        if len(self.hidden_size) < self.max_hidden_layers:  # HARD LIMIT
            self.hidden_size += [self.hidden_size[-1]]
            self.recreate_nets()
        else:
            self.add_mlp_node()

    def remove_mlp_layer(self):
        """Removes a hidden layer from neural network."""
        if len(self.hidden_size) > self.min_hidden_layers:  # HARD LIMIT
            self.hidden_size = self.hidden_size[:-1]
            self.recreate_nets(shrink_params=True)
        else:
            self.add_mlp_node()

    def add_mlp_node(self, hidden_layer=None, numb_new_nodes=None):
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
            self.recreate_nets()

        return {"hidden_layer": hidden_layer, "numb_new_nodes": numb_new_nodes}

    def remove_mlp_node(self, hidden_layer=None, numb_new_nodes=None):
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

        if (
            self.hidden_size[hidden_layer] - numb_new_nodes > self.min_mlp_nodes
        ):  # HARD LIMIT
            self.hidden_size[hidden_layer] -= numb_new_nodes
            self.recreate_nets(shrink_params=True)

        return {"hidden_layer": hidden_layer, "numb_new_nodes": numb_new_nodes}

    def recreate_nets(self, shrink_params=False):
        """Recreates neural networks."""
        new_feature_net, new_value_net, new_advantage_net = self.create_net()
        if shrink_params:
            new_feature_net = self.shrink_preserve_parameters(
                old_net=self.feature_net, new_net=new_feature_net
            )
            if self.rainbow:
                new_value_net = self.shrink_preserve_parameters(
                    old_net=self.value_net, new_net=new_value_net
                )
                new_advantage_net = self.shrink_preserve_parameters(
                    old_net=self.advantage_net, new_net=new_advantage_net
                )
        else:
            new_feature_net = self.preserve_parameters(
                old_net=self.feature_net, new_net=new_feature_net
            )
            if self.rainbow:
                new_value_net = self.preserve_parameters(
                    old_net=self.value_net, new_net=new_value_net
                )
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
        clone = EvolvableMLP(**copy.deepcopy(self.init_dict))
        clone.load_state_dict(self.state_dict())
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
                        else:
                            param.data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
                            ] = old_net_dict[key].data[
                                : min(old_size[0], new_size[0]),
                                : min(old_size[1], new_size[1]),
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
