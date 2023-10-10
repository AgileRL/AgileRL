import copy
import math
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agilerl.networks.custom_architecture import GumbelSoftmax


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


class EvolvableCNN(nn.Module):
    """The Evolvable Convolutional Neural Network class.

    :param input_shape: Input shape
    :type input_shape: List[int]
    :param channel_size: CNN channel size
    :type channel_size: List[int]
    :param kernel_size: Comvolution kernel size
    :type kernel_size: List[int]
    :param stride_size: Convolution stride size
    :type stride_size: List[int]
    :param hidden_size: Hidden layer(s) size
    :type hidden_size: List[int]
    :param num_actions: Action dimension
    :type num_actions: int
    :param num_atoms: Number of atoms for Rainbow DQN, defaults to 50
    :type num_atoms: int, optional
    :param mlp_activation: MLP activation layer, defaults to 'relu'
    :type mlp_activation: str, optional
    :param cnn_activation: CNN activation layer, defaults to 'relu'
    :type cnn_activation: str, optional
    :param n_agents: Number of agents, defaults to None
    :type n_agents: int, optional
    :param multi: Boolean flag to indicate if this is a multi-agent problem, defaults to False
    :type multi: bool, optional
    :param layer_norm: Normalization between layers, defaults to False
    :type layer_norm: bool, optional
    :param stored_values: Stored network weights, defaults to None
    :type stored_values: numpy.array(), optional
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
    :type accelerator: Hugging Face accelerate.Accelerator(), optional
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
        mlp_activation="relu",
        cnn_activation="relu",
        mlp_output_activation="relu",
        n_agents=None,
        multi=False,
        layer_norm=False,
        stored_values=None,
        support=None,
        rainbow=False,
        critic=False,
        normalize=True,
        device="cpu",
        accelerator=None,
    ):
        super().__init__()

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

        if stored_values is not None:
            self.inject_parameters(pvec=stored_values, without_layer_norm=False)

    def get_activation(self, activation_names):
        """Returns activation function for corresponding activation name.

        :param activation_names: Activation function name
        :type activation_names: str
        """
        activation_functions = {
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "softsign": nn.Softsign,
            "sigmoid": nn.Sigmoid,
            "softmax": nn.Softmax,
            "gumbel_softmax": GumbelSoftmax,
            "softplus": nn.Softplus,
            "lrelu": nn.LeakyReLU,
            "prelu": nn.PReLU,
        }
        return (
            activation_functions[activation_names](dim=1)
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
                self.feature_net, self.value_net, self.advantage_net = (
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
                    self.feature_net,
                    self.value_net,
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
            x = F.softmax(value, dim=-1)

        return x

    def get_model_dict(self):
        """Returns dictionary with model information and weights."""
        model_dict = self.init_dict
        model_dict.update(
            {"stored_values": self.extract_parameters(without_layer_norm=False)}
        )
        return model_dict

    def count_parameters(self, without_layer_norm=False):
        """Returns number of parameters in neural network.

        :param without_layer_norm: Exclude normalization layers, defaults to False
        :type without_layer_norm: bool, optional
        """
        count = 0
        for name, param in self.named_parameters():
            if not without_layer_norm or "layer_norm" not in name:
                count += param.data.cpu().numpy().flatten().shape[0]
        return count

    def extract_grad(self, without_layer_norm=False):
        """Returns current pytorch gradient in same order as genome's flattened
        parameter vector.

        :param without_layer_norm: Exclude normalization layers, defaults to False
        :type without_layer_norm: bool, optional
        """
        tot_size = self.count_parameters(without_layer_norm)
        pvec = np.zeros(tot_size, np.float32)
        count = 0
        for name, param in self.named_parameters():
            if not without_layer_norm or "layer_norm" not in name:
                sz = param.grad.data.cpu().numpy().flatten().shape[0]
                pvec[count : count + sz] = param.grad.data.cpu().numpy().flatten()
                count += sz
        return pvec.copy()

    def extract_parameters(self, without_layer_norm=False):
        """Returns current flattened neural network weights.

        :param without_layer_norm: Exclude normalization layers, defaults to False
        :type without_layer_norm: bool, optional
        """
        tot_size = self.count_parameters(without_layer_norm)
        pvec = np.zeros(tot_size, np.float32)
        count = 0
        for name, param in self.named_parameters():
            if not without_layer_norm or "layer_norm" not in name:
                sz = param.data.cpu().detach().numpy().flatten().shape[0]
                pvec[count : count + sz] = param.data.cpu().detach().numpy().flatten()
                count += sz
        return copy.deepcopy(pvec)

    def inject_parameters(self, pvec, without_layer_norm=False):
        """Injects a flat vector of neural network parameters into the model's current
        neural network weights.

        :param pvec: Network weights
        :type pvec: np.array()
        :param without_layer_norm: Exclude normalization layers, defaults to False
        :type without_layer_norm: bool, optional
        """
        count = 0

        for name, param in self.named_parameters():
            if not without_layer_norm or "layer_norm" not in name:
                sz = param.data.cpu().numpy().flatten().shape[0]
                raw = pvec[count : count + sz]
                reshaped = raw.reshape(param.data.cpu().numpy().shape)
                param.data = torch.from_numpy(copy.deepcopy(reshaped)).type(
                    torch.FloatTensor
                )
                count += sz
        return pvec

    @property
    def short_dict(self):
        """Returns shortened version of model information in dictionary."""
        short_dict = {
            "channel_size": self.channel_size,
            "kernel_size": self.kernel_size,
            "stride_size": self.stride_size,
            "hidden_size": self.hidden_size,
            "num_atoms": self.num_atoms,
            "mlp_activation": self.mlp_activation,
            "cnn_activation": self.cnn_activation,
            "layer_norm": self.layer_norm,
            "support": self.support,
        }
        return short_dict

    @property
    def init_dict(self):
        """Returns model information in dictionary."""
        initdict = {
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
            "multi": self.multi,
            "layer_norm": self.layer_norm,
            "critic": self.critic,
            "rainbow": self.rainbow,
            "device": self.device,
            "accelerator": self.accelerator,
        }
        return initdict

    def add_mlp_layer(self):
        """Adds a hidden layer to Multi-layer Perceptron."""
        if len(self.hidden_size) < 3:  # HARD LIMIT
            self.hidden_size += [self.hidden_size[-1]]

            self.recreate_nets()
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

        if self.hidden_size[hidden_layer] + numb_new_nodes <= 1024:  # HARD LIMIT
            self.hidden_size[hidden_layer] += numb_new_nodes

            self.recreate_nets()
        return {"hidden_layer": hidden_layer, "numb_new_nodes": numb_new_nodes}

    def add_cnn_layer(self):
        """Adds a hidden layer to Convolutional Neural Network."""
        if self.multi:
            if len(self.channel_size) < 6:  # HARD LIMIT
                self.channel_size += [self.channel_size[-1]]
                self.kernel_size += [(1, 3, 3)]
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

        else:
            if len(self.channel_size) < 6:  # HARD LIMIT
                self.channel_size += [self.channel_size[-1]]
                self.kernel_size += [3]

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
        """Adds channel to hidden layer of Convolutional Neural Network.

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

        if self.channel_size[hidden_layer] + numb_new_channels <= 256:  # HARD LIMIT
            self.channel_size[hidden_layer] += numb_new_channels

            self.recreate_nets()

        return {"hidden_layer": hidden_layer, "numb_new_channels": numb_new_channels}

    def recreate_nets(self):
        """Recreates neural networks."""
        new_feature_net, new_value_net, new_advantage_net = self.create_nets()
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
                        else:
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
