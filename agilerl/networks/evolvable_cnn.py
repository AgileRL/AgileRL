from typing import List, Optional, Tuple, Union, Dict, Any
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator

from agilerl.typing import ArrayOrTensor
from agilerl.networks.base import EvolvableModule, MutationType, register_mutation_fn
from agilerl.utils.evolvable_networks import (
    get_activation,
    calc_max_kernel_sizes,
    create_conv_block,
    create_mlp
)

class EvolvableCNN(EvolvableModule):
    """The Evolvable Convolutional Neural Network class. It supports the evolution of the CNN architecture
    by adding or removing convolutional layers, changing the number of channels in each layer, changing the
    kernel size and stride size of each layer, and changing the number of nodes in the fully connected layer.

    :param input_shape: Input shape
    :type input_shape: list[int]
    :param channel_size: CNN channel size
    :type channel_size: list[int]
    :param kernel_size: Convolution kernel size
    :type kernel_size: list[int]
    :param stride_size: Convolution stride size
    :type stride_size: list[int]
    :param hidden_size: Hidden layer(s) size
    :type hidden_size: list[int]
    :param num_outputs: Action dimension
    :type num_outputs: int
    :param num_atoms: Number of atoms for Rainbow DQN, defaults to 51
    :type num_atoms: int, optional
    :param mlp_output_activation: MLP output activation layer, defaults to None
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
    :param noise_std: Noise standard deviation, defaults to 0.5
    :type noise_std: float, optional
    :param critic: CNN is a critic network, defaults to False
    :type critic: bool, optional
    :param feature_extractor: CNN is a feature extractor, defaults to False
    :type feature_extractor: bool, optional
    :param normalize: Normalize CNN inputs, defaults to True
    :type normalize: bool, optional
    :param init_layers: Initialise network layers, defaults to True
    :type init_layers: bool, optional
    :param output_vanish: Vanish output by multiplying by 0.1, defaults to False
    :type output_vanish: bool, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: Optional[Accelerator]
    """

    def __init__(
        self,
        input_shape: List[int],
        channel_size: List[int],
        kernel_size: List[int],
        stride_size: List[int],
        hidden_size: List[int],
        num_outputs: int,
        num_atoms: int = 51,
        latent_dim: Optional[int] = None,
        mlp_output_activation: Optional[str] = None,
        mlp_activation: str = "ReLU",
        cnn_activation: str = "ReLU",
        min_hidden_layers: int = 1,
        max_hidden_layers: int = 3,
        min_mlp_nodes: int = 64,
        max_mlp_nodes: int = 1024,
        min_cnn_hidden_layers: int = 1,
        max_cnn_hidden_layers: int = 6,
        min_channel_size: int = 32,
        max_channel_size: int = 256,
        n_agents: Optional[int] = None,
        layer_norm: bool = False,
        support: Optional[torch.Tensor] = None,
        rainbow: bool = False,
        noise_std: float = 0.5,
        critic: bool = False,
        normalize: bool = True,
        init_layers: bool = True,
        output_vanish: bool = False,
        device: str = "cpu",
        accelerator: Optional[Accelerator] = None,
        arch: str = "cnn"
        ) -> None:
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
            num_outputs > 0
        ), "'num_outputs' cannot be less than or equal to zero, please enter a valid integer."
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
        if n_agents is not None:
            assert isinstance(n_agents, int) and n_agents > 0, "'n_agents' must be an integer and greater than 0."

        self.arch = arch
        self.input_shape = input_shape
        self.channel_size = channel_size
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.hidden_size = hidden_size
        self.num_outputs = num_outputs
        self.latent_dim = latent_dim
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
        self.init_layers = init_layers
        self.device = device
        self.accelerator = accelerator
        self.n_agents = n_agents
        self.noise_std = noise_std
        self.output_vanish = output_vanish
        self._net_config = {
            "arch": self.arch,
            "channel_size": self.channel_size,
            "kernel_size": self.kernel_size,
            "stride_size": self.stride_size,
            "normalize": self.normalize,
            "cnn_activation": self.cnn_activation,
            "hidden_size": self.hidden_size,
            "mlp_activation": self.mlp_activation,
            "mlp_output_activation": self.mlp_output_activation,
            "min_cnn_hidden_layers": self.min_cnn_hidden_layers,
            "max_cnn_hidden_layers": self.max_cnn_hidden_layers,
            "min_channel_size": self.min_channel_size,
            "max_channel_size": self.max_channel_size,
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
            "input_shape": self.input_shape,
            "channel_size": self.channel_size,
            "kernel_size": self.kernel_size,
            "stride_size": self.stride_size,
            "hidden_size": self.hidden_size,
            "num_outputs": self.num_outputs,
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
            "layer_norm": self.layer_norm,
            "critic": self.critic,
            "rainbow": self.rainbow,
            "noise_std": self.noise_std,
            "output_vanish": self.output_vanish,
            "device": self.device,
        }
        return init_dict

    def create_cnn(
        self,
        in_channels: int,
        channel_size: List[int],
        kernel_size: List[int],
        stride_size: List[int],
        name: str
    ) -> nn.Sequential:
        """
        Creates and returns a convolutional neural network.

        :param in_channels: The number of input channels.
        :type in_channels: int
        :param channel_size: A list of integers representing the number of channels in each convolutional layer.
        :type channel_size: List[int]
        :param kernel_size: A list of integers representing the kernel size of each convolutional layer.
        :type kernel_size: List[int]
        :param stride_size: A list of integers representing the stride size of each convolutional layer.
        :type stride_size: List[int]
        :param name: The name of the CNN.
        :type name: str
        :param features_dim: The dimension of the features output by the CNN. Defaults to None.
        :type features_dim: Optional[int], optional

        :return: The created convolutional neural network.
        :rtype: nn.Sequential
        """
        # Build the main convolutional block
        net_dict = create_conv_block(
            in_channels=in_channels,
            channel_size=channel_size,
            kernel_size=kernel_size,
            stride_size=stride_size,
            name=name,
            critic=self.critic,
            init_layers=self.init_layers,
            layer_norm=self.layer_norm,
            activation_fn=self.cnn_activation,
            n_agents=self.n_agents,
        )

        # If the CNN is a critic or feature extractor, add a linear layer to flatten the output
        if self.critic or self.latent_dim is not None:
            # Features dim is different for critic and feature extractor
            features_dim = self.latent_dim if self.latent_dim is not None else self.hidden_size[0]

            net_dict[f"{name}_flatten"] = nn.Flatten()
            if self.n_agents is not None:
                sample_input = (
                    torch.zeros(1, *self.input_shape)
                    .unsqueeze(2)
                    .repeat(1, 1, self.n_agents, 1, 1)
                )
            else:
                sample_input = torch.zeros((1, *self.input_shape))

            with torch.no_grad():
                flattened_size = nn.Sequential(net_dict)(sample_input).shape[1]

            net_dict[f"{name}_linear_output"] = nn.Linear(flattened_size, features_dim)
            net_dict[f"{name}_output_activation"] = get_activation(
                self.cnn_activation
            )

        return nn.Sequential(net_dict)

    def build_networks(self) -> Tuple[nn.Module, nn.Module, Optional[nn.Module]]:
        """
        Creates and returns neural networks from the current configuration.

        :return: A tuple containing the feature network and the value network.
        :rtype: Tuple[nn.Module, nn.Module]
        """
        feature_net = self.create_cnn(
            self.input_shape[0],
            self.channel_size,
            self.kernel_size,
            self.stride_size,
            name="feature",
        )

        with torch.no_grad():
            if self.n_agents is not None:
                if self.critic:
                    critic_input = (
                        torch.zeros(1, *self.input_shape)
                        .unsqueeze(2)
                        .repeat(1, 1, self.n_agents, 1, 1)
                    )
                    cnn_output: torch.Tensor = feature_net(critic_input)
                    input_size = cnn_output.view(1, -1).size(1)
                else:
                    cnn_output = feature_net(
                        torch.zeros(1, *self.input_shape).unsqueeze(2)
                    )
                    input_size = cnn_output.view(1, -1).size(1)
            else:
                sample = torch.zeros((1, *self.input_shape))
                cnn_output = feature_net(sample)
                input_size = cnn_output.view(1, -1).size(1)

        if self.critic:
            input_size += self.num_outputs

        if self.rainbow:
            value_net = create_mlp(
                input_size,
                output_size=self.num_atoms,
                hidden_size=self.hidden_size,
                output_vanish=self.output_vanish,
                name="value",
                output_activation=None,
                noisy=True,
            )
            advantage_net = create_mlp(
                input_size,
                output_size=self.num_atoms * self.num_outputs,
                hidden_size=self.hidden_size,
                output_vanish=self.output_vanish,
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
                value_net = create_mlp(
                    input_size,
                    output_size=1,
                    hidden_size=self.hidden_size,
                    name="value",
                    output_vanish=self.output_vanish,
                    output_activation=self.mlp_output_activation,
                )
            elif self.latent_dim is not None: # If a feature extractor, last hidden layer is the output layer
                value_net = create_mlp(
                    input_size,
                    output_size=self.latent_dim,
                    hidden_size=self.hidden_size,
                    name="feature_head",
                    output_vanish=self.output_vanish,
                    output_activation=self.mlp_output_activation,
                )
            else:
                value_net = create_mlp(
                    input_size,
                    output_size=self.num_outputs,
                    hidden_size=self.hidden_size,
                    name="value",
                    output_vanish=self.output_vanish,
                    output_activation=self.mlp_output_activation,
                )
            advantage_net = None
            if self.accelerator is None:
                feature_net = feature_net.to(self.device)
                value_net = value_net.to(self.device)

        self.cnn_output_size = cnn_output.shape

        return feature_net, value_net, advantage_net

    def reset_noise(self) -> None:
        """Resets noise of value and advantage networks."""
        networks = [self.value_net]
        if self.rainbow:
            networks.append(self.advantage_net)
        
        EvolvableModule.reset_noise(*networks)

    def forward(
            self,
            x: ArrayOrTensor,
            xc: Optional[ArrayOrTensor] = None,
            q: bool = True,
            log: bool = False
            ) -> torch.Tensor:
        """Returns output of neural network.

        :param x: Neural network input
        :type x: torch.Tensor() or np.array
        :param xc: Actions to be evaluated by critic, defaults to None
        :type xc: torch.Tensor() or np.array, optional
        :param q: Return Q value if using rainbow, defaults to True
        :type q: bool, optional
        :param log: Return log softmax if using rainbow, defaults to False
        :type log: bool, optional
        :return: Output of the neural network
        :rtype: torch.Tensor
        """
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
            x = x.to(self.device)

        if x.dtype != torch.float32:
            x = x.type(torch.float32)

        batch_size = x.size(0)

        if self.normalize:
            x = x / 255.0

        x: torch.Tensor = self.feature_net(x)
        x = x.reshape(batch_size, -1)

        if self.critic:
            x = torch.cat([x, xc], dim=1)

        value: torch.Tensor = self.value_net(x)

        if self.rainbow:
            advantage: torch.Tensor = self.advantage_net(x)
            value = value.view(batch_size, 1, self.num_atoms)
            advantage = advantage.view(batch_size, self.num_outputs, self.num_atoms)
            x = value + advantage - advantage.mean(1, keepdim=True)
            if log:
                x = F.log_softmax(x.view(-1, self.num_atoms), dim=-1).view(
                    -1, self.num_outputs, self.num_atoms
                )
                return x
            else:
                x = F.softmax(x.view(-1, self.num_atoms), dim=-1).view(
                    -1, self.num_outputs, self.num_atoms
                )
                x = x.clamp(min=1e-3)

            if q:
                x = torch.sum(x * self.support, dim=2)
        else:
            x = value

        return x

    @register_mutation_fn(MutationType.LAYER)
    def add_mlp_layer(self) -> None:
        """Adds a hidden layer to the fully connected layer."""
        if len(self.hidden_size) < self.max_hidden_layers:  # HARD LIMIT
            self.hidden_size += [self.hidden_size[-1]]
            self.recreate_nets()
        else:
            self.add_mlp_node()

    @register_mutation_fn(MutationType.LAYER)
    def remove_mlp_layer(self) -> None:
        """Removes a hidden layer from the fully connected layer."""
        if len(self.hidden_size) > self.min_hidden_layers:
            self.hidden_size = self.hidden_size[:-1]
            self.recreate_nets(shrink_params=True)
        else:
            self.add_mlp_node()

    @register_mutation_fn(MutationType.LAYER)
    def add_cnn_layer(self) -> None:
        """Adds a hidden layer to convolutional neural network."""
        max_kernels = calc_max_kernel_sizes(
            self.channel_size, self.kernel_size, self.stride_size, self.input_shape
        )
        if (
            len(self.channel_size) < self.max_cnn_hidden_layers
            and not any(i <= 2 for i in self.cnn_output_size[-2:])
            and max_kernels[-1] > 2
        ):  # HARD LIMIT
            self.channel_size += [self.channel_size[-1]]
            k_size = np.random.randint(2, max_kernels[-1] + 1)
            self.kernel_size += [k_size]
            self.stride_size = self.stride_size + [
                np.random.randint(1, self.stride_size[-1] + 1)
            ]
            self.recreate_nets()
        else:
            self.add_cnn_channel()

    @register_mutation_fn(MutationType.LAYER)
    def remove_cnn_layer(self) -> None:
        """Removes a hidden layer from convolutional neural network."""
        if len(self.channel_size) > self.min_cnn_hidden_layers:
            self.channel_size = self.channel_size[:-1]
            self.kernel_size = self.kernel_size[:-1]
            self.stride_size = self.stride_size[:-1]
            self.recreate_nets(shrink_params=True)
        else:
            self.add_cnn_channel()

    @register_mutation_fn(MutationType.NODE)
    def add_mlp_node(
            self,
            hidden_layer: Optional[int] = None,
            numb_new_nodes: Optional[int] = None
            ) -> Dict[str, int]:
        """Adds nodes to the hidden layer of the Multi-layer Perceptron.

        :param hidden_layer: Depth of the hidden layer to add nodes to, defaults to None
        :type hidden_layer: int, optional
        :param numb_new_nodes: Number of nodes to add to the hidden layer, defaults to None
        :type numb_new_nodes: int, optional
        :return: Dictionary containing the hidden layer and number of new nodes added
        :rtype: dict
        """
        if hidden_layer is None:
            hidden_layer = np.random.randint(0, len(self.hidden_size), 1)[0]
        else:
            hidden_layer = min(hidden_layer, len(self.hidden_size) - 1)
        if numb_new_nodes is None:
            numb_new_nodes = np.random.choice([32, 64, 128], 1)[0]

        # HARD LIMIT
        if self.hidden_size[hidden_layer] + numb_new_nodes <= self.max_mlp_nodes:
            self.hidden_size[hidden_layer] += numb_new_nodes
            self.recreate_nets()

        return {"hidden_layer": hidden_layer, "numb_new_nodes": numb_new_nodes}

    @register_mutation_fn(MutationType.NODE)
    def remove_mlp_node(
            self,
            hidden_layer: Optional[int] = None,
            numb_new_nodes: Optional[int] = None
            ) -> Dict[str, int]:
        """Removes nodes from hidden layer of fully connected layer.

        :param hidden_layer: Depth of hidden layer to remove nodes from, defaults to None
        :type hidden_layer: int, optional
        :param numb_new_nodes: Number of nodes to remove from hidden layer, defaults to None
        :type numb_new_nodes: int, optional
        :return: Dictionary containing the hidden layer index and the number of nodes removed
        :rtype: Dict[str, Union[int, None]]
        """
        if hidden_layer is None:
            hidden_layer = np.random.randint(0, len(self.hidden_size), 1)[0]
        else:
            hidden_layer = min(hidden_layer, len(self.hidden_size) - 1)
        if numb_new_nodes is None:
            numb_new_nodes = np.random.choice([16, 32, 64], 1)[0]

        # HARD LIMIT
        if self.hidden_size[hidden_layer] - numb_new_nodes > self.min_mlp_nodes:
            self.hidden_size[hidden_layer] = self.hidden_size[hidden_layer] - numb_new_nodes
            self.recreate_nets(shrink_params=True)

        return {"hidden_layer": hidden_layer, "numb_new_nodes": numb_new_nodes}

    @register_mutation_fn(MutationType.NODE)
    def change_cnn_kernel(self) -> None:
        """Randomly alters convolution kernel of random CNN layer."""
        max_kernels = calc_max_kernel_sizes(
            self.channel_size, self.kernel_size, self.stride_size, self.input_shape
        )
        if len(self.channel_size) > 1:
            hidden_layer = np.random.randint(1, min(4, len(self.channel_size)), 1)[0]
            self.kernel_size[hidden_layer] = np.random.randint(1, max_kernels[hidden_layer] + 1)
            self.recreate_nets()
        else:
            self.add_cnn_layer()

    @register_mutation_fn(MutationType.NODE)
    def add_cnn_channel(
            self,
            hidden_layer: Optional[int] = None,
            numb_new_channels: Optional[int] = None
            ) -> Dict[str, int]:
        """Adds channel to hidden layer of convolutional neural network.

        :param hidden_layer: Depth of hidden layer to add channel to, defaults to None
        :type hidden_layer: int, optional
        :param numb_new_channels: Number of channels to add to hidden layer, defaults to None
        :type numb_new_channels: int, optional
        :return: Dictionary containing the hidden layer and number of new channels added
        :rtype: dict[str, int]
        """
        if hidden_layer is None:
            hidden_layer = np.random.randint(0, len(self.channel_size), 1)[0]
        else:
            hidden_layer = min(hidden_layer, len(self.channel_size) - 1)
        if numb_new_channels is None:
            numb_new_channels = np.random.choice([8, 16, 32], 1)[0]

        # HARD LIMIT
        if self.channel_size[hidden_layer] + numb_new_channels <= self.max_channel_size:
            self.channel_size[hidden_layer] += numb_new_channels

            self.recreate_nets()

        return {"hidden_layer": hidden_layer, "numb_new_channels": numb_new_channels}

    @register_mutation_fn(MutationType.NODE)
    def remove_cnn_channel(
            self,
            hidden_layer: Optional[int] = None,
            numb_new_channels: Optional[int] = None
            ) -> Dict[str, int]:
        """Remove channel from hidden layer of convolutional neural network.

        :param hidden_layer: Depth of hidden layer to add channel to, defaults to None
        :type hidden_layer: int, optional
        :param numb_new_channels: Number of channels to add to hidden layer, defaults to None
        :type numb_new_channels: int, optional
        :return: Dictionary containing the hidden layer and number of new channels
        :rtype: Dict[str, Union[int, None]]
        """
        if hidden_layer is None:
            hidden_layer = np.random.randint(0, len(self.channel_size), 1)[0]
        else:
            hidden_layer = min(hidden_layer, len(self.channel_size) - 1)

        if numb_new_channels is None:
            numb_new_channels = np.random.choice([8, 16, 32], 1)[0]

        # HARD LIMIT
        if self.channel_size[hidden_layer] - numb_new_channels > self.min_channel_size:
            self.channel_size[hidden_layer] -= numb_new_channels
            self.recreate_nets(shrink_params=True)

        return {"hidden_layer": hidden_layer, "numb_new_channels": numb_new_channels}

    def recreate_nets(self, shrink_params: bool = False) -> None:
        """Recreates neural networks.

        :param shrink_params: Flag indicating whether to shrink the parameters, defaults to False
        :type shrink_params: bool, optional
        """
        new_feature_net, new_value_net, new_advantage_net = self.build_networks()
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

    def clone(self) -> "EvolvableCNN":
        """Returns clone of neural net with identical parameters.
        
        :return: Clone of neural network
        :rtype: EvolvableCNN
        """
        clone = super().clone()
        clone.rainbow = self.rainbow
        clone.critic = self.critic
        return clone
