from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agilerl.modules.base import EvolvableModule, MutationType, mutation
from agilerl.modules.custom_components import NoisyLinear
from agilerl.typing import ArrayOrTensor, DeviceType
from agilerl.utils.evolvable_networks import get_activation, layer_init


def create_mlp(
    input_size: int,
    output_size: int,
    hidden_size: List[int],
    output_vanish: bool,
    output_activation: Optional[str],
    noisy: bool = False,
    rainbow_feature_net: bool = False,
    init_layers: bool = True,
    layer_norm: bool = False,
    mlp_activation: str = "ReLU",
    mlp_output_activation: Optional[str] = None,
    noise_std: float = 0.1,
    rainbow: bool = False,
    device: DeviceType = "cpu",
    name: str = "mlp",
) -> nn.Sequential:
    """Creates and returns multi-layer perceptron.

    :param input_size: Number of input features.
    :type input_size: int
    :param output_size: Number of output features.
    :type output_size: int
    :param hidden_size: List of hidden layer sizes.
    :type hidden_size: List[int]
    :param output_vanish: Whether to initialize output layer weights to a small value.
    :type output_vanish: bool
    :param output_activation: Activation function for output layer.
    :type output_activation: Optional[str]
    :param noisy: Whether to use noisy layers.
    :type noisy: bool, optional
    :param rainbow_feature_net: Whether to use a rainbow feature network.
    :type rainbow_feature_net: bool, optional
    :param init_layers: Whether to initialize the layers.
    :type init_layers: bool, optional
    :param layer_norm: Whether to use layer normalization.
    :type layer_norm: bool, optional
    :param mlp_activation: Activation function for hidden layers.
    :type mlp_activation: str, optional
    :param mlp_output_activation: Activation function for output layer.
    :type mlp_output_activation: Optional[str], optional
    :param noise_std: Standard deviation of noise for noisy layers.
    :type noise_std: float, optional
    :param rainbow: Whether to use a rainbow network.
    :type rainbow: bool, optional
    :param name: Name of the network.
    :type name: str, default "mlp"

    :return: Multi-layer perceptron.
    :rtype: nn.Sequential
    """
    net_dict = OrderedDict()

    # Initialize the first block
    if noisy:
        net_dict[f"{name}_linear_layer_0"] = NoisyLinear(
            input_size, hidden_size[0], std_init=noise_std, device=device
        )
    else:
        net_dict[f"{name}_linear_layer_0"] = nn.Linear(
            input_size, hidden_size[0], device=device
        )

    if init_layers:
        net_dict[f"{name}_linear_layer_0"] = layer_init(
            net_dict[f"{name}_linear_layer_0"]
        )

    if layer_norm:
        net_dict[f"{name}_layer_norm_0"] = nn.LayerNorm(hidden_size[0], device=device)
    net_dict[f"{name}_activation_0"] = get_activation(
        activation_name=(
            mlp_output_activation
            if (len(hidden_size) == 1 and rainbow_feature_net)
            else mlp_activation
        )
    )

    if len(hidden_size) > 1:
        for l_no in range(1, len(hidden_size)):
            # Add linear layer
            if noisy:
                net_dict[f"{name}_linear_layer_{str(l_no)}"] = NoisyLinear(
                    hidden_size[l_no - 1], hidden_size[l_no], noise_std, device=device
                )
            else:
                net_dict[f"{name}_linear_layer_{str(l_no)}"] = nn.Linear(
                    hidden_size[l_no - 1], hidden_size[l_no], device=device
                )
            # Initialize layer weights
            if init_layers:
                net_dict[f"{name}_linear_layer_{str(l_no)}"] = layer_init(
                    net_dict[f"{name}_linear_layer_{str(l_no)}"]
                )
            # Add layer normalization
            if layer_norm:
                net_dict[f"{name}_layer_norm_{str(l_no)}"] = nn.LayerNorm(
                    hidden_size[l_no], device=device
                )
            # Add activation function
            net_dict[f"{name}_activation_{str(l_no)}"] = get_activation(
                mlp_activation if not rainbow_feature_net else mlp_output_activation
            )

    if not rainbow_feature_net:
        if noisy:
            output_layer = NoisyLinear(
                hidden_size[-1], output_size, noise_std, device=device
            )
        else:
            output_layer = nn.Linear(hidden_size[-1], output_size, device=device)

        if init_layers:
            output_layer = layer_init(output_layer)

        if output_vanish:
            if rainbow:
                output_layer.weight_mu.data.mul_(0.1)
                output_layer.bias_mu.data.mul_(0.1)
                output_layer.weight_sigma.data.mul_(0.1)
                output_layer.bias_sigma.data.mul_(0.1)
            else:
                output_layer.weight.data.mul_(0.1)
                output_layer.bias.data.mul_(0.1)

        net_dict[f"{name}_linear_layer_output"] = output_layer
        if output_activation is not None:
            net_dict[f"{name}_activation_output"] = get_activation(
                activation_name=output_activation
            )

    net = nn.Sequential(net_dict)
    return net


class EvolvableMLP(EvolvableModule):
    """The Evolvable Multi-layer Perceptron class.

    :param num_inputs: Input layer dimension
    :type num_inputs: int
    :param num_outputs: Output layer dimension
    :type num_outputs: int
    :param hidden_size: Hidden layer(s) size
    :type hidden_size: List[int]
    :param num_atoms: Number of atoms for Rainbow DQN, defaults to 51
    :type num_atoms: Optional[int]
    :param mlp_activation: Activation layer, defaults to 'ReLU'
    :type mlp_activation: Optional[str]
    :param mlp_output_activation: Output activation layer, defaults to None
    :type mlp_output_activation: Optional[str]
    :param min_hidden_layers: Minimum number of hidden layers the network will shrink down to, defaults to 1
    :type min_hidden_layers: Optional[int]
    :param max_hidden_layers: Maximum number of hidden layers the network will expand to, defaults to 3
    :type max_hidden_layers: Optional[int]
    :param min_mlp_nodes: Minimum number of nodes a layer can have within the network, defaults to 64
    :type min_mlp_nodes: Optional[int]
    :param max_mlp_nodes: Maximum number of nodes a layer can have within the network, defaults to 500
    :type max_mlp_nodes: Optional[int]
    :param layer_norm: Normalization between layers, defaults to True
    :type layer_norm: Optional[bool]
    :param output_vanish: Vanish output by multiplying by 0.1, defaults to True
    :type output_vanish: Optional[bool]
    :param init_layers: Initialise network layers, defaults to True
    :type init_layers: Optional[bool]
    :param support: Atoms support tensor, defaults to None
    :type support: Optional[torch.Tensor]
    :param rainbow: Using Rainbow DQN, defaults to False
    :type rainbow: Optional[bool]
    :param noise_std: Noise standard deviation, defaults to 0.5
    :type noise_std: Optional[float]
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: Optional[str]
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: Optional[accelerate.Accelerator]
    """

    arch: str = "mlp"

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        hidden_size: List[int],
        num_atoms: int = 51,
        mlp_activation: str = "ReLU",
        mlp_output_activation: str = None,
        min_hidden_layers: int = 1,
        max_hidden_layers: int = 3,
        min_mlp_nodes: int = 64,
        max_mlp_nodes: int = 500,
        layer_norm: bool = True,
        output_vanish: bool = True,
        init_layers: bool = True,
        support: torch.Tensor = None,
        rainbow: bool = False,
        noise_std: float = 0.5,
        device: str = "cpu",
        arch: str = "mlp",
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

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.mlp_activation = mlp_activation
        self.mlp_output_activation = mlp_output_activation
        self.min_hidden_layers = min_hidden_layers
        self.max_hidden_layers = max_hidden_layers
        self.min_mlp_nodes = min_mlp_nodes
        self.max_mlp_nodes = max_mlp_nodes
        self.layer_norm = layer_norm
        self.output_vanish = output_vanish
        self.init_layers = init_layers
        self.hidden_size = hidden_size
        self.num_atoms = num_atoms
        self.support = support
        self.rainbow = rainbow
        self.noise_std = noise_std
        self._net_config = {
            "arch": self.arch,
            "hidden_size": self.hidden_size,
            "mlp_activation": self.mlp_activation,
            "mlp_output_activation": self.mlp_output_activation,
            "min_hidden_layers": self.min_hidden_layers,
            "max_hidden_layers": self.max_hidden_layers,
            "min_mlp_nodes": self.min_mlp_nodes,
            "max_mlp_nodes": self.max_mlp_nodes,
        }

        self.feature_net, self.value_net, self.advantage_net = self.build_networks()

    @property
    def net_config(self) -> Dict[str, Any]:
        return self._net_config

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
            "noise_std": self.noise_std,
            "device": self.device,
        }
        return init_dict

    @property
    def activation(self) -> str:
        """Returns activation function."""
        return self.mlp_activation

    @activation.setter
    def activation(self, activation: str) -> None:
        """Set activation function."""
        self.mlp_activation = activation

    @mutation(MutationType.ACTIVATION)
    def change_activation(self, activation: str, output: bool = False) -> None:
        """Set the activation function for the network.

        :param activation: Activation function to use.
        :type activation: str
        :param output: Flag indicating whether to set the output activation function, defaults to False
        :type output: bool, optional

        :return: Activation function
        :rtype: str
        """
        if output:
            self.mlp_output_activation = activation

        self.mlp_activation = activation
        self.recreate_network()

    def build_networks(
        self,
    ) -> Tuple[nn.Module, Optional[nn.Module], Optional[nn.Module]]:
        """Creates and returns the neural networks for feature, value, and advantage.

        :return: Neural networks for feature and, optionally, value and advantage
        :rtype: Tuple[nn.Module, Optional[nn.Module], Optional[nn.Module]]"""
        if not self.rainbow:
            feature_net = create_mlp(
                input_size=self.num_inputs,
                output_size=self.num_outputs,
                hidden_size=self.hidden_size,
                output_vanish=self.output_vanish,
                output_activation=self.mlp_output_activation,
                layer_norm=self.layer_norm,
                device=self.device,
            )

        value_net, advantage_net = None, None

        if self.rainbow:
            feature_net = create_mlp(
                input_size=self.num_inputs,
                output_size=self.hidden_size[0],
                hidden_size=[self.hidden_size[0]],
                output_vanish=False,
                output_activation=self.mlp_activation,
                rainbow_feature_net=True,
                init_layers=self.init_layers,
                layer_norm=self.layer_norm,
                mlp_activation=self.mlp_activation,
                mlp_output_activation=self.mlp_output_activation,
                noise_std=self.noise_std,
                rainbow=True,
                device=self.device,
            )
            value_net = create_mlp(
                input_size=self.hidden_size[0],
                output_size=self.num_atoms,
                hidden_size=self.hidden_size[1:],
                output_vanish=self.output_vanish,
                output_activation=None,
                noisy=True,
                init_layers=self.init_layers,
                layer_norm=self.layer_norm,
                mlp_activation=self.mlp_activation,
                mlp_output_activation=self.mlp_output_activation,
                noise_std=self.noise_std,
                rainbow=True,
                device=self.device,
            )
            advantage_net = create_mlp(
                input_size=self.hidden_size[0],
                output_size=self.num_atoms * self.num_outputs,
                hidden_size=self.hidden_size[1:],
                output_vanish=self.output_vanish,
                output_activation=None,
                noisy=True,
                init_layers=self.init_layers,
                layer_norm=self.layer_norm,
                mlp_activation=self.mlp_activation,
                mlp_output_activation=self.mlp_output_activation,
                noise_std=self.noise_std,
                rainbow=True,
                device=self.device,
            )

        return feature_net, value_net, advantage_net

    # def reset_noise(self) -> None:
    #     """Resets noise of value and advantage networks."""
    #     networks = [self.value_net]
    #     if self.rainbow:
    #         networks.append(self.advantage_net)

    #     EvolvableModule.reset_noise(*networks)

    def forward(
        self, x: ArrayOrTensor, q: bool = True, log: bool = False
    ) -> torch.Tensor:
        """Returns output of neural network.

        :param x: Neural network input
        :type x: torch.Tensor() or np.array
        :param q: Return Q value if using rainbow, defaults to True
        :type q: bool, optional
        :param log: Return log softmax instead of softmax, defaults to False
        :type log: bool, optional

        :return: Neural network output
        :rtype: torch.Tensor
        """
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(np.array(x))
            if self.accelerator is None:
                x = x.to(self.device)

        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        # Forward pass through feature network
        x = self.feature_net(x)

        # If using rainbow we need to calculate the Q value through (maybe log) softmax
        if self.rainbow:
            value: torch.Tensor = self.value_net(x)
            advantage: torch.Tensor = self.advantage_net(x)
            value = value.view(-1, 1, self.num_atoms)
            advantage = advantage.view(-1, self.num_outputs, self.num_atoms)
            x = value + advantage - advantage.mean(1, keepdim=True)
            if log:
                x = F.log_softmax(x, dim=2)
                return x
            else:
                x = F.softmax(x, dim=2)

            # Output at this point has shape -> (batch_size, actions, num_support)
            if q:
                x = torch.sum(x * self.support.expand_as(x), dim=2)

        return x

    @mutation(MutationType.LAYER)
    def add_mlp_layer(self):
        """Adds a hidden layer to neural network."""
        # add layer to hyper params
        if len(self.hidden_size) < self.max_hidden_layers:  # HARD LIMIT
            self.hidden_size += [self.hidden_size[-1]]
        else:
            self.add_mlp_node()

    @mutation(MutationType.LAYER)
    def remove_mlp_layer(self):
        """Removes a hidden layer from neural network."""
        if len(self.hidden_size) > self.min_hidden_layers:  # HARD LIMIT
            self.hidden_size = self.hidden_size[:-1]
        else:
            self.add_mlp_node()

    @mutation(MutationType.NODE)
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

        return {"hidden_layer": hidden_layer, "numb_new_nodes": numb_new_nodes}

    @mutation(MutationType.NODE)
    def remove_mlp_node(
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
        """Recreates neural networks, shrinking parameters if necessary.

        :param shrink_params: Shrink parameters of neural networks, defaults to False
        :type shrink_params: bool, optional
        """
        new_feature_net, new_value_net, new_advantage_net = self.build_networks()

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
