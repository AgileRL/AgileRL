import copy
from typing import List, Optional, Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import accelerate

from agilerl.typing import ArrayOrTensor
from agilerl.modules.base import EvolvableModule, MutationType, register_mutation_fn
from agilerl.utils.evolvable_networks import create_mlp

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
        gpt_activations: bool = False,
        accelerator: Optional[accelerate.Accelerator] = None,
        arch: str = "mlp",
        name: Optional[str] = None
        ):
        super().__init__(gpt=gpt_activations)

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

        self.arch = arch

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.mlp_activation = mlp_activation
        self.mlp_output_activation = mlp_output_activation
        self.gpt_activations = gpt_activations
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
        self.device = device
        self.accelerator = accelerator
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

    def build_networks(self) -> Tuple[nn.Module, Optional[nn.Module], Optional[nn.Module]]:
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
                gpt_activations=self.gpt_activations,
                layer_norm=self.layer_norm
            )
            if self.accelerator is None:
                feature_net = feature_net.to(self.device)

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
                gpt_activations=self.gpt_activations,
                mlp_activation=self.mlp_activation,
                mlp_output_activation=self.mlp_output_activation,
                noise_std=self.noise_std,
                rainbow=True
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
                gpt_activations=self.gpt_activations,
                mlp_activation=self.mlp_activation,
                mlp_output_activation=self.mlp_output_activation,
                noise_std=self.noise_std,
                rainbow=True
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
                gpt_activations=self.gpt_activations,
                mlp_activation=self.mlp_activation,
                mlp_output_activation=self.mlp_output_activation,
                noise_std=self.noise_std,
                rainbow=True
            )
            if self.accelerator is None:
                value_net, advantage_net, feature_net = (
                    value_net.to(self.device),
                    advantage_net.to(self.device),
                    feature_net.to(self.device),
                )

        return feature_net, value_net, advantage_net

    def reset_noise(self) -> None:
        """Resets noise of value and advantage networks."""
        networks = [self.value_net]
        if self.rainbow:
            networks.append(self.advantage_net)
        
        EvolvableModule.reset_noise(*networks)

    def forward(self, x: ArrayOrTensor, q: bool = True, log: bool = False) -> torch.Tensor:
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

    @register_mutation_fn(MutationType.LAYER)
    def add_mlp_layer(self):
        """Adds a hidden layer to neural network."""
        # add layer to hyper params
        if len(self.hidden_size) < self.max_hidden_layers:  # HARD LIMIT
            self.hidden_size += [self.hidden_size[-1]]
            self.recreate_nets()
        else:
            self.add_mlp_node()

    @register_mutation_fn(MutationType.LAYER)
    def remove_mlp_layer(self):
        """Removes a hidden layer from neural network."""
        if len(self.hidden_size) > self.min_hidden_layers:  # HARD LIMIT
            self.hidden_size = self.hidden_size[:-1]
            self.recreate_nets(shrink_params=True)
        else:
            self.add_mlp_node()

    @register_mutation_fn(MutationType.NODE)
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

    @register_mutation_fn(MutationType.NODE)
    def remove_mlp_node(self, hidden_layer: Optional[int] = None, numb_new_nodes: Optional[int] = None) -> Dict[str, int]:
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
            self.recreate_nets(shrink_params=True)

        return {"hidden_layer": hidden_layer, "numb_new_nodes": numb_new_nodes}

    def recreate_nets(self, shrink_params: bool = False) -> None:
        """Recreates neural networks, shrinking parameters if necessary.
        
        :param shrink_params: Shrink parameters of neural networks, defaults to False
        :type shrink_params: bool, optional
        """
        new_feature_net, new_value_net, new_advantage_net = self.build_networks()
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

    def clone(self) -> "EvolvableMLP":
        """Returns clone of neural net with identical parameters.
        
        :return: Clone of neural network
        :rtype: EvolvableMLP
        """
        clone = EvolvableMLP(**copy.deepcopy(self.init_dict))
        clone.load_state_dict(self.state_dict())
        return clone

