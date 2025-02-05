from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator

from agilerl.modules.base import EvolvableModule, MutationType, mutation
from agilerl.modules.cnn import EvolvableCNN
from agilerl.modules.custom_components import GumbelSoftmax, NoisyLinear
from agilerl.typing import ArrayOrTensor
from agilerl.utils.evolvable_networks import (
    get_activation,
    get_conv_layer,
    get_normalization,
    get_pooling,
    layer_init,
)

LayerInfo = Dict[str, Dict[int, str]]


class MakeEvolvable(EvolvableModule):
    """Wrapper to make a neural network evolvable,

    .. warning::
        This class will be deprecated in a future release. We recommend users to define evolvable networks through
        the ``EvolvableModule`` and ``EvolvableNetwork class hierarchies. Please refer to :ref:`custom_network_architectures`
        for more information on how to do this.

    :param network: Input neural network
    :type network: nn.Module
    :param input_tensor: Example input tensor so forward pass can be made to detect the network architecture
    :type input_tensor: torch.Tensor
    :param num_atoms: Number of atoms for Rainbow DQN, defaults to 51
    :type num_atoms: int, optional
    :param secondary_input_tensor: Second input tensor if network performs forward pass with two tensors, for example, \
        off-policy algorithms that use a critic(s) with environments that have RGB image observations and thus require CNN \
        architecture, defaults to None
    :type secondary_input_tensor: torch.Tensor, optional
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
    :param output_vanish: Vanish output by multiplying by 0.1, defaults to False
    :type output_vanish: bool, optional
    :param init_layers: Initialise network layers, defaults to False
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

    mlp_layer_info: LayerInfo
    cnn_layer_info: LayerInfo

    def __init__(
        self,
        network: nn.Module,
        input_tensor: torch.Tensor,
        secondary_input_tensor: Optional[torch.Tensor] = None,
        num_atoms: int = 51,
        min_hidden_layers: int = 1,
        max_hidden_layers: int = 3,
        min_mlp_nodes: int = 64,
        max_mlp_nodes: int = 1024,
        min_cnn_hidden_layers: int = 1,
        max_cnn_hidden_layers: int = 6,
        min_channel_size: int = 32,
        max_channel_size: int = 256,
        output_vanish: bool = False,
        init_layers: bool = False,
        support: Optional[torch.Tensor] = None,
        rainbow: bool = False,
        device: str = "cpu",
        accelerator: Optional[Accelerator] = None,
        **kwargs: dict,
    ):
        super().__init__(device)
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
        if not kwargs:
            assert isinstance(
                network, nn.Module
            ), f"'network' must be of type 'nn.Module'.{type(network)}"

        self.init_layers = init_layers
        self.min_hidden_layers = min_hidden_layers
        self.max_hidden_layers = max_hidden_layers
        self.min_mlp_nodes = min_mlp_nodes
        self.max_mlp_nodes = max_mlp_nodes
        self.min_cnn_hidden_layers = min_cnn_hidden_layers
        self.max_cnn_hidden_layers = max_cnn_hidden_layers
        self.min_channel_size = min_channel_size
        self.max_channel_size = max_channel_size
        self.output_vanish = output_vanish
        self.device = device
        self.accelerator = accelerator

        #### Rainbow attributes
        self.rainbow = rainbow  #### add in as a doc string
        self.num_atoms = num_atoms
        self.support = support

        # Set the layer counters
        self.conv_counter = -1
        self.lin_counter = -1
        self.extra_critic_dims = (
            secondary_input_tensor.shape[-1]
            if secondary_input_tensor is not None
            else None
        )

        # Set placeholder attributes (needed for init_dict function to work)
        self.has_conv_layers = False
        self.input_tensor = input_tensor.to(self.device)
        self.secondary_input_tensor = (
            secondary_input_tensor.to(self.device)
            if secondary_input_tensor is not None
            else secondary_input_tensor
        )
        (
            self.in_channels,
            self.channel_size,
            self.kernel_size,
            self.stride_size,
            self.padding,
        ) = (None, None, None, None, None)

        # If first instance, network used to instantiate, upon cloning, init_dict used instead
        if not kwargs:
            self.detect_architecture(
                network.to(self.device), self.input_tensor, self.secondary_input_tensor
            )
        else:
            for key, value in kwargs.items():
                setattr(self, key, value)

        self.feature_net, self.value_net, self.advantage_net = self.build_networks()

        # Need to remove CNN mutation methods if network is not a CNN
        if not self.has_conv_layers:
            self._layer_mutation_methods = [
                method for method in self._layer_mutation_methods if "cnn" not in method
            ]

            self._node_mutation_methods = [
                method for method in self._node_mutation_methods if "cnn" not in method
            ]

    @property
    def activation(self) -> str:
        """Returns the activation function."""
        return self.mlp_activation

    @property
    def output_activation(self) -> str:
        """Returns the output activation function."""
        return self.mlp_output_activation

    def get_output_dense(self) -> nn.Module:
        """Returns the output dense layer."""
        final_layer = self.value_net if self.value_net is not None else self.feature_net
        return getattr(
            final_layer,
            f"{'value' if self.value_net is not None else 'feature'}_linear_layer_output",
        )

    def init_weights_gaussian(self, std_coeff: float = 4.0, output_coeff: float = 2.0):
        """Initialise network weights using Gaussian distribution.

        :param std_coeff: Standard deviation coefficient, defaults to 4.0
        :type std_coeff: float, optional
        :param output_coeff: Output coefficient, defaults to 2.0
        :type output_coeff: float, optional
        """
        layers = [module for module in self.feature_net.children()]
        if self.arch == "cnn":
            layers += [module for module in self.value_net.children()]

        # Initialize network layers
        l_no = 0
        for i, layer in enumerate(layers):
            if i < len(layers) - 1:
                if isinstance(layer, nn.Linear):
                    EvolvableModule.init_weights_gaussian(layer, std_coeff=std_coeff)
                    l_no += 1
            else:
                EvolvableModule.init_weights_gaussian(layer, std_coeff=output_coeff)

    def forward(self, x: ArrayOrTensor, xc: ArrayOrTensor = None, q: bool = True):
        """Returns output of neural network.

        :param x: Neural network input
        :type x: torch.Tensor() or np.array
        :param xc: Actions to be evaluated by critic, defaults to None
        :type xc: torch.Tensor() or np.array, optional
        :param q: Return Q value if using rainbow, defaults to True
        :type q: bool, optional
        """
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(np.array(x))

        if self.accelerator is None:
            x = x.to(self.device)

        if x.dtype != torch.float32:
            x = x.type(torch.float32)

        batch_size = x.size(0)

        x = self.feature_net(x)

        # Check if there is a cnn
        if self.cnn_layer_info:
            x = x.reshape(batch_size, -1)
            # Ensure dtype is float32

            # Concatenate actions if passed to network as a separate tensor
            if xc is not None:
                if self.accelerator is None:
                    xc = xc.to(self.device)
                x = torch.cat([x, xc], dim=1)

            value = self.value_net(x)

        # add in cnn functionality
        if self.rainbow:
            advantage: torch.Tensor = self.advantage_net(x)
            if not self.cnn_layer_info:
                value: torch.Tensor = self.value_net(x)
                value = value.view(-1, 1, self.num_atoms)
                advantage = advantage.view(-1, self.num_outputs, self.num_atoms)
                x = value + advantage - advantage.mean(1, keepdim=True)
                x = F.softmax(x, dim=-1)
            else:
                value = value.view(batch_size, 1, self.num_atoms)
                advantage = advantage.view(batch_size, self.num_outputs, self.num_atoms)

                x = value + advantage - advantage.mean(1, keepdim=True)
                x = F.softmax(x.view(-1, self.num_atoms), dim=-1).view(
                    -1, self.num_outputs, self.num_atoms
                )
            x = x.clamp(min=1e-3)

            if q:
                x = torch.sum(x * self.support, dim=2)
        else:
            if self.cnn_layer_info:
                x = value

        return x

    def detect_architecture(
        self,
        network: nn.Module,
        input_tensor: torch.Tensor,
        secondary_input_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        """Detect the architecture of a neural network.

        :param network: Neural network whose architecture is being detected
        :type network: nn.Module
        :param input_tensor: Tensor used to perform forward pass to detect layers
        :type input_tensor: torch.Tensor
        :param secondary_input_tensor: Second tensor used to perform forward pass if forward
            method of neural network takes two tensors as arguments, defaults to None
        :type secondary_input_tensor: torch.Tensor, optional
        """
        in_features_list = []
        out_features_list = []
        in_channel_list = []
        out_channel_list = []
        kernel_size_list = []
        stride_size_list = []
        padding_list = []

        cnn_layer_info = dict()
        mlp_layer_info = dict()

        def register_hooks(module):
            def forward_hook(
                module: nn.Module, input: torch.Tensor, output: torch.Tensor
            ):
                # Convolutional layer detection
                if isinstance(module, nn.modules.conv._ConvNd):
                    self.has_conv_layers = True
                    self.conv_counter += 1

                    if "conv_layer_type" not in cnn_layer_info.keys():
                        cnn_layer_info["conv_layer_type"] = str(
                            module.__class__.__name__
                        )

                    in_channel_list.append(module.in_channels)
                    out_channel_list.append(module.out_channels)
                    kernel_size_list.append(module.kernel_size)
                    stride_size_list.append(module.stride)
                    padding_list.append(module.padding)

                # Linear layer detection
                elif isinstance(module, nn.Linear):
                    self.lin_counter += 1
                    in_features_list.append(module.in_features)
                    out_features_list.append(module.out_features)

                # Normalization layer detection
                elif isinstance(
                    module,
                    (
                        nn.BatchNorm2d,
                        nn.BatchNorm3d,
                        nn.InstanceNorm2d,
                        nn.InstanceNorm3d,
                        nn.LayerNorm,
                    ),
                ):
                    if len(output.shape) <= 2:
                        if "norm_layers" not in mlp_layer_info.keys():
                            mlp_layer_info["norm_layers"] = dict()

                        mlp_layer_info["norm_layers"][self.lin_counter] = str(
                            module.__class__.__name__
                        )
                    else:
                        if "norm_layers" not in cnn_layer_info.keys():
                            cnn_layer_info["norm_layers"] = dict()

                        cnn_layer_info["norm_layers"][self.conv_counter] = str(
                            module.__class__.__name__
                        )

                # Pooling layer detection
                elif isinstance(
                    module, (nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool2d, nn.AvgPool3d)
                ):
                    if "pooling_layers" not in cnn_layer_info.keys():
                        cnn_layer_info["pooling_layers"] = dict()
                    cnn_layer_info["pooling_layers"][self.conv_counter] = dict()
                    cnn_layer_info["pooling_layers"][self.conv_counter]["name"] = str(
                        module.__class__.__name__
                    )
                    cnn_layer_info["pooling_layers"][self.conv_counter][
                        "kernel"
                    ] = module.kernel_size
                    cnn_layer_info["pooling_layers"][self.conv_counter][
                        "stride"
                    ] = module.stride
                    cnn_layer_info["pooling_layers"][self.conv_counter][
                        "padding"
                    ] = module.padding

                # Skip nn.Flatten layer as this is added when building the CNN to connect
                # the convolutional layers with the fully-connected layers
                elif isinstance(module, nn.Flatten):
                    pass

                # Detect activation layer (supported currently by AgileRL)
                elif isinstance(
                    module,
                    (
                        nn.Tanh,
                        nn.Identity,
                        nn.ReLU,
                        nn.ELU,
                        nn.Softsign,
                        nn.Sigmoid,
                        GumbelSoftmax,
                        nn.Softplus,
                        nn.Softmax,
                        nn.LeakyReLU,
                        nn.PReLU,
                        nn.GELU,
                    ),
                ):
                    if len(output.shape) <= 2:
                        if "activation_layers" not in mlp_layer_info.keys():
                            mlp_layer_info["activation_layers"] = dict()
                        mlp_layer_info["activation_layers"][self.lin_counter] = str(
                            module.__class__.__name__
                        )
                    else:
                        if "activation_layers" not in cnn_layer_info.keys():
                            cnn_layer_info["activation_layers"] = dict()
                        cnn_layer_info["activation_layers"][self.conv_counter] = str(
                            module.__class__.__name__
                        )
                else:
                    raise Exception(
                        f"{module} not currently supported, use an alternative layer."
                    )

            if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not isinstance(module, type(network))
            ):
                hooks.append(module.register_forward_hook(forward_hook))

        hooks = []
        network.apply(register_hooks)

        # Forward pass to collect network data necessary to make network evolvable
        with torch.no_grad():
            if secondary_input_tensor is None:
                network(input_tensor)
            else:
                network(input_tensor, secondary_input_tensor)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Save neural network information as attributes
        self.num_inputs, *self.hidden_size = in_features_list
        *_, self.num_outputs = out_features_list
        if len(self.hidden_size) == 0:
            raise TypeError("Network must have at least one hidden layer.")

        self.mlp_layer_info = mlp_layer_info

        if len(out_features_list) - 1 in mlp_layer_info["activation_layers"].keys():
            self.mlp_output_activation = mlp_layer_info["activation_layers"][
                len(out_features_list) - 1
            ]
        else:
            self.mlp_output_activation = None

        activation_function_set = set(mlp_layer_info["activation_layers"].values())
        if self.mlp_output_activation is not None:
            activation_function_set.remove(self.mlp_output_activation)

        if len(activation_function_set) > 1:
            raise TypeError(
                "All activation functions other than the output layer activation must be the same."
            )
        else:
            self.mlp_activation = list(mlp_layer_info["activation_layers"].values())[0]

        self.cnn_layer_info = cnn_layer_info
        if self.has_conv_layers is True:
            self.arch = "cnn"
            self.in_channels = in_channel_list[0]
            self.channel_size = out_channel_list
            self.kernel_size = kernel_size_list
            self.stride_size = stride_size_list
            self.padding = padding_list
        else:
            self.arch = "mlp"

        # Reset the layer counters
        self.conv_counter = -1
        self.lin_counter = -1

    def create_mlp(
        self,
        input_size: int,
        output_size: int,
        hidden_size: list[int],
        name: str,
        mlp_activation: str,
        mlp_output_activation: Optional[str],
        noisy: bool = False,
        rainbow_feature_net: bool = False,
    ) -> nn.Sequential:
        """Creates and returns multi-layer perceptron.

        :param input_size: Input dimensions to first MLP layer
        :type input_size: int
        :param output_size: Output dimensions from last MLP layer
        :type output_size: int
        :param hidden_size: Hidden layer sizes
        :type hidden_size: list[int]
        :param name: Layer name
        :type name: str
        :param mlp_activation: Activation function for hidden layers
        :type mlp_activation: str
        :param mlp_output_activation: Activation function for output layer
        :type mlp_output_activation: Optional[str]
        :param noisy: Whether to use NoisyLinear layers
        :type noisy: bool
        :param rainbow_feature_net: Whether this is a Rainbow DQN feature network
        :type rainbow_feature_net: bool
        """

        net_dict = OrderedDict()
        if noisy:
            net_dict[f"{name}_linear_layer_0"] = NoisyLinear(input_size, hidden_size[0])
        else:
            net_dict[f"{name}_linear_layer_0"] = nn.Linear(input_size, hidden_size[0])

        if self.init_layers:
            net_dict[f"{name}_linear_layer_0"] = layer_init(
                net_dict[f"{name}_linear_layer_0"]
            )

        if ("norm_layers" in self.mlp_layer_info.keys()) and (
            0 in self.mlp_layer_info["norm_layers"].keys()
        ):
            net_dict[f"{name}_layer_norm_0"] = get_normalization(
                self.mlp_layer_info["norm_layers"][0], hidden_size[0]
            )

        if ("activation_layers" in self.mlp_layer_info.keys()) and (
            0 in self.mlp_layer_info["activation_layers"].keys()
        ):
            net_dict[f"{name}_activation_0"] = get_activation(
                mlp_output_activation
                if (len(hidden_size) == 1 and rainbow_feature_net)
                else mlp_activation
            )

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
                if self.init_layers:
                    net_dict[f"{name}_linear_layer_{str(l_no)}"] = layer_init(
                        net_dict[f"{name}_linear_layer_{str(l_no)}"]
                    )
                if ("norm_layers" in self.mlp_layer_info.keys()) and (
                    l_no in self.mlp_layer_info["norm_layers"].keys()
                ):
                    net_dict[f"{name}_layer_norm_{str(l_no)}"] = get_normalization(
                        self.mlp_layer_info["norm_layers"][l_no], hidden_size[l_no]
                    )
                if l_no in self.mlp_layer_info["activation_layers"].keys():
                    net_dict[f"{name}_activation_{str(l_no)}"] = get_activation(
                        mlp_activation
                        if not rainbow_feature_net
                        else mlp_output_activation
                    )
        if not rainbow_feature_net:
            if noisy:
                output_layer = NoisyLinear(hidden_size[-1], output_size)
            else:
                output_layer = nn.Linear(hidden_size[-1], output_size)
            if self.init_layers:
                output_layer = layer_init(output_layer)

            if self.output_vanish:
                output_layer.weight.data.mul_(0.1)
                output_layer.bias.data.mul_(0.1)

            net_dict[f"{name}_linear_layer_output"] = output_layer
            if mlp_output_activation is not None:
                net_dict[f"{name}_activation_output"] = get_activation(
                    mlp_output_activation
                )

        return nn.Sequential(net_dict)

    def create_cnn(
        self,
        input_size: int,
        channel_size: List[int],
        kernel_size: List[int],
        stride_size: List[int],
        padding: List[int],
        name: str,
    ) -> nn.Sequential:
        """Creates and returns convolutional neural network.

        :param input_size: Channel size of first layer
        :type input_size: int
        :param channel_size: Output channel sizes for each layer
        :type channel_size: list[int]
        :param kernel_size: Kernel sizes
        :type kernel_size: list[int] or list[Tuple[int]]
        :param stride_size: Stride sizes
        :type stride_size: list[int] or list[Tuple[int]]
        :param padding: Convolutional layer padding
        :type padding: list[int] or list[Tuple[int]]
        :param name: Layer name
        :type name: str
        """

        net_dict = OrderedDict()
        # if self.cnn_layer_info["conv_layer_type"] == "Conv3d":
        #     k_size = [
        #         (self.input_tensor.shape[-3], k_size[1], k_size[2])
        #         for k_size in kernel_size
        #     ]
        # else:

        net_dict[f"{name}_conv_layer_0"] = get_conv_layer(
            self.cnn_layer_info["conv_layer_type"],
            in_channels=input_size,
            out_channels=channel_size[0],
            kernel_size=kernel_size[0],
            stride=stride_size[0],
            padding=padding[0],
        )
        if ("norm_layers" in self.cnn_layer_info.keys()) and (
            0 in self.cnn_layer_info["norm_layers"].keys()
        ):
            net_dict[f"{name}_layer_norm_0"] = get_normalization(
                self.cnn_layer_info["norm_layers"][0], channel_size[0]
            )

        if ("activation_layers" in self.cnn_layer_info.keys()) and (
            0 in self.cnn_layer_info["activation_layers"].keys()
        ):
            net_dict[f"{name}_activation_0"] = get_activation(
                self.cnn_layer_info["activation_layers"][0]
            )

        if ("pooling_layers" in self.cnn_layer_info.keys()) and (
            0 in self.cnn_layer_info["pooling_layers"].keys()
        ):
            net_dict[f"{name}_pooling_0"] = get_pooling(
                self.cnn_layer_info["pooling_layers"][0]["name"],
                self.cnn_layer_info["pooling_layers"][0]["kernel"],
                self.cnn_layer_info["pooling_layers"][0]["stride"],
                self.cnn_layer_info["pooling_layers"][0]["padding"],
            )

        if len(channel_size) > 1:
            for l_no in range(1, len(channel_size)):
                net_dict[f"{name}_conv_layer_{str(l_no)}"] = get_conv_layer(
                    self.cnn_layer_info["conv_layer_type"],
                    in_channels=channel_size[l_no - 1],
                    out_channels=channel_size[l_no],
                    kernel_size=kernel_size[l_no],
                    stride=stride_size[l_no],
                    padding=padding[l_no],
                )
                if ("norm_layers" in self.cnn_layer_info.keys()) and (
                    l_no in self.cnn_layer_info["norm_layers"].keys()
                ):
                    net_dict[f"{name}_layer_norm_{str(l_no)}"] = get_normalization(
                        self.cnn_layer_info["norm_layers"][l_no], channel_size[l_no]
                    )

                if ("activation_layers" in self.cnn_layer_info.keys()) and (
                    l_no in self.cnn_layer_info["activation_layers"].keys()
                ):
                    net_dict[f"{name}_activation_{str(l_no)}"] = get_activation(
                        self.cnn_layer_info["activation_layers"][l_no]
                    )

                if ("pooling_layers" in self.cnn_layer_info.keys()) and (
                    l_no in self.cnn_layer_info["pooling_layers"].keys()
                ):
                    net_dict[f"{name}_pooling_{str(l_no)}"] = get_pooling(
                        self.cnn_layer_info["pooling_layers"][l_no]["name"],
                        self.cnn_layer_info["pooling_layers"][l_no]["kernel"],
                        self.cnn_layer_info["pooling_layers"][l_no]["stride"],
                        self.cnn_layer_info["pooling_layers"][l_no]["padding"],
                    )

        return nn.Sequential(net_dict)

    def build_networks(self) -> Tuple[nn.Module, nn.Module, Optional[nn.Module]]:
        """Creates and returns the feature and value net."""

        # Check if any CNN layers otherwise return just a mlp
        if self.cnn_layer_info:
            feature_net = self.create_cnn(
                self.in_channels,
                self.channel_size,
                self.kernel_size,
                self.stride_size,
                self.padding,
                name="feature",
            )
            cnn_output: torch.Tensor = feature_net(
                torch.zeros(*self.input_tensor.shape)
            )
            self.cnn_output_size = cnn_output.shape
            input_size = (cnn_output).to(self.device).view(1, -1).size(1)

            if self.secondary_input_tensor is not None:
                input_size += self.extra_critic_dims

            if self.rainbow:
                value_net = self.create_mlp(
                    input_size,
                    output_size=self.num_atoms,
                    hidden_size=self.hidden_size,
                    name="value",
                    noisy=True,
                    mlp_output_activation=self.mlp_output_activation,
                    mlp_activation=self.mlp_activation,
                )
                advantage_net = self.create_mlp(
                    input_size,
                    output_size=self.num_atoms * self.num_outputs,
                    hidden_size=self.hidden_size,
                    name="advantage",
                    noisy=True,
                    mlp_output_activation=self.mlp_output_activation,
                    mlp_activation=self.mlp_activation,
                )
            else:
                value_net = self.create_mlp(
                    input_size,
                    self.num_outputs,
                    self.hidden_size,
                    name="value",
                    mlp_activation=self.mlp_activation,
                    mlp_output_activation=self.mlp_output_activation,
                )
                advantage_net = None

        else:
            input_size = self.num_inputs
            if self.rainbow:
                feature_net = self.create_mlp(
                    input_size=self.num_inputs,
                    output_size=128,
                    hidden_size=[128],
                    name="feature",
                    rainbow_feature_net=True,
                    mlp_activation=self.mlp_activation,
                    mlp_output_activation="ReLU",
                )
                value_net = self.create_mlp(
                    input_size=128,
                    output_size=self.num_atoms,
                    hidden_size=self.hidden_size,
                    noisy=True,
                    name="value",
                    mlp_output_activation=self.mlp_output_activation,
                    mlp_activation=self.mlp_activation,
                )
                advantage_net = self.create_mlp(
                    input_size=128,
                    output_size=self.num_atoms * self.num_outputs,
                    hidden_size=self.hidden_size,
                    noisy=True,
                    name="advantage",
                    mlp_output_activation=self.mlp_output_activation,
                    mlp_activation=self.mlp_activation,
                )
            else:
                value_net = None
                advantage_net = None
                feature_net = self.create_mlp(
                    input_size,
                    self.num_outputs,
                    self.hidden_size,
                    name="feature",
                    mlp_activation=self.mlp_activation,
                    mlp_output_activation=self.mlp_output_activation,
                )

        if self.accelerator is None:
            feature_net = feature_net.to(self.device)
            value_net = (
                value_net.to(self.device) if value_net is not None else value_net
            )
            advantage_net = (
                advantage_net.to(self.device)
                if advantage_net is not None
                else advantage_net
            )
        return feature_net, value_net, advantage_net

    def get_init_dict(self) -> Dict[str, Any]:
        """Returns model information in dictionary."""
        init_dict = {
            "network": None,
            "input_tensor": self.input_tensor,
            "secondary_input_tensor": self.secondary_input_tensor,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "hidden_size": self.hidden_size,
            "mlp_activation": self.mlp_activation,
            "mlp_output_activation": self.mlp_output_activation,
            "device": self.device,
            "accelerator": self.accelerator,
            "in_channels": self.in_channels,
            "channel_size": self.channel_size,
            "kernel_size": self.kernel_size,
            "stride_size": self.stride_size,
            "padding": self.padding,
            "extra_critic_dims": self.extra_critic_dims,
            "output_vanish": self.output_vanish,
            "init_layers": self.init_layers,
            "has_conv_layer": self.has_conv_layers,
            "arch": self.arch,
            "cnn_layer_info": self.cnn_layer_info,
            "mlp_layer_info": self.mlp_layer_info,
            "num_atoms": self.num_atoms,
            "rainbow": self.rainbow,
            "support": self.support,
        }

        return init_dict

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

    @mutation(MutationType.LAYER)
    def add_mlp_layer(self) -> None:
        """Adds a hidden layer to value network."""
        if len(self.hidden_size) < self.max_hidden_layers:  # HARD LIMIT
            self.hidden_size += [self.hidden_size[-1]]
            self.mlp_layer_info["activation_layers"][
                len(self.hidden_size) - 1
            ] = self.mlp_activation
            if self.mlp_output_activation is not None:
                self.mlp_layer_info["activation_layers"][
                    len(self.hidden_size)
                ] = self.mlp_output_activation
        else:
            self.add_mlp_node()

    @mutation(MutationType.LAYER)
    def remove_mlp_layer(self) -> None:
        """Removes a hidden layer from value network."""
        if len(self.hidden_size) > self.min_hidden_layers:  # HARD LIMIT
            self.hidden_size = self.hidden_size[:-1]
            if len(self.hidden_size) in self.mlp_layer_info["activation_layers"].keys():
                if self.mlp_output_activation is None:
                    self.mlp_layer_info["activation_layers"].pop(len(self.hidden_size))
                else:
                    self.mlp_layer_info["activation_layers"].pop(
                        len(self.hidden_size) + 1
                    )
                    self.mlp_layer_info["activation_layers"][
                        len(self.hidden_size)
                    ] = self.mlp_output_activation
            else:
                if self.mlp_output_activation is not None:
                    self.mlp_layer_info["activation_layers"].pop(
                        len(self.hidden_size) + 1
                    )
                    self.mlp_layer_info["activation_layers"][
                        len(self.hidden_size)
                    ] = self.mlp_output_activation

            if (
                "norm_layers" in self.mlp_layer_info.keys()
                and len(self.hidden_size) in self.mlp_layer_info["norm_layers"]
            ):
                self.mlp_layer_info["norm_layers"].pop(len(self.hidden_size))

        else:
            self.add_mlp_node()

    @mutation(MutationType.NODE)
    def add_mlp_node(
        self, hidden_layer: Optional[int] = None, numb_new_nodes: Optional[int] = None
    ) -> Dict[str, int]:
        """Adds nodes to hidden layer of value network.

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

        if (
            self.hidden_size[hidden_layer] - numb_new_nodes > self.min_mlp_nodes
        ):  # HARD LIMIT
            self.hidden_size[hidden_layer] -= numb_new_nodes

        return {"hidden_layer": hidden_layer, "numb_new_nodes": numb_new_nodes}

    @mutation(MutationType.LAYER)
    def add_cnn_layer(self) -> None:
        """Adds a hidden layer to convolutional neural network."""
        max_kernels = self.calc_max_kernel_sizes()
        stride_size_ranges = self.calc_stride_size_ranges()
        if (
            len(self.channel_size) < self.max_cnn_hidden_layers
            and not any(i <= 2 for i in self.cnn_output_size[-2:])
            and all(i > 0 for i in max_kernels[-1])
        ):  # HARD LIMIT
            self.channel_size += [self.channel_size[-1]]
            k_size = tuple(np.random.randint(1, 1 + k) for k in max_kernels[-1])
            self.kernel_size += [k_size]
            self.padding += [self.padding[-1]]
            stride_size_list = [
                tuple(
                    np.random.randint(tup[0], tup[1] + 1) for _ in self.stride_size[-1]
                )
                for tup in stride_size_ranges
            ]
            self.stride_size = stride_size_list + [
                tuple(1 for _ in self.stride_size[-1])
            ]
            if "activation_layers" not in self.cnn_layer_info.keys():
                self.cnn_layer_info["activation_layers"] = dict()
            self.cnn_layer_info["activation_layers"][
                len(self.channel_size) - 1
            ] = "ReLU"

        else:
            self.add_cnn_channel()

    @mutation(MutationType.LAYER)
    def remove_cnn_layer(self) -> None:
        """Removes a hidden layer from the convolutional neural network."""
        stride_size_ranges = self.calc_stride_size_ranges()
        if len(self.channel_size) > self.min_cnn_hidden_layers:
            self.channel_size = self.channel_size[:-1]
            self.kernel_size = self.kernel_size[:-1]

            stride_size_list = [
                tuple(
                    np.random.randint(tup[0], tup[1] + 1) for _ in self.stride_size[-1]
                )
                for tup in stride_size_ranges
            ]
            self.stride_size = stride_size_list[:-1]

            if "activation_layers" in self.cnn_layer_info.keys():
                if len(self.channel_size) in self.cnn_layer_info["activation_layers"]:
                    self.cnn_layer_info["activation_layers"].pop(len(self.channel_size))
            else:
                self.cnn_layer_info["activation_layers"] = dict()
            if (
                len(self.channel_size) - 1
                not in self.cnn_layer_info["activation_layers"]
            ):
                self.cnn_layer_info["activation_layers"][
                    len(self.channel_size) - 1
                ] = "ReLU"

            if (
                "pooling_layers" in self.cnn_layer_info.keys()
                and len(self.channel_size) in self.cnn_layer_info["pooling_layers"]
            ):
                self.cnn_layer_info["pooling_layers"].pop(len(self.channel_size))

            if (
                "norm_layers" in self.cnn_layer_info.keys()
                and len(self.channel_size) in self.cnn_layer_info["norm_layers"]
            ):
                self.cnn_layer_info["norm_layers"].pop(len(self.channel_size))

        else:
            self.add_cnn_channel()

    @mutation(MutationType.NODE)
    def change_cnn_kernel(self) -> None:
        """Randomly alters convolution kernel of random CNN layer."""
        max_kernels = self.calc_max_kernel_sizes()
        # if self.cnn_layer_info["conv_layer_type"] == "Conv3d":
        if len(self.channel_size) > 1:
            hidden_layer = np.random.randint(1, min(4, len(self.channel_size)), 1)[0]
            kernel_size_values = tuple(
                np.random.choice([3, 4, 5, 7]) for _ in self.kernel_size[-1]
            )
            kernel_size_values = tuple(
                i if i <= max_kernels[-1][idx] else int(max_kernels[-1][idx])
                for idx, i in enumerate(kernel_size_values)
            )
            self.kernel_size[hidden_layer] = kernel_size_values
        else:
            self.add_cnn_layer()

    @mutation(MutationType.NODE)
    def add_cnn_channel(
        self,
        hidden_layer: Optional[int] = None,
        numb_new_channels: Optional[int] = None,
    ) -> Dict[str, int]:
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

        if (
            self.channel_size[hidden_layer] + numb_new_channels <= self.max_channel_size
        ):  # HARD LIMIT
            self.channel_size[hidden_layer] += numb_new_channels

        return {"hidden_layer": hidden_layer, "numb_new_channels": numb_new_channels}

    @mutation(MutationType.NODE, shrink_params=True)
    def remove_cnn_channel(
        self,
        hidden_layer: Optional[int] = None,
        numb_new_channels: Optional[int] = None,
    ) -> Dict[str, int]:
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

        return {"hidden_layer": hidden_layer, "numb_new_channels": numb_new_channels}

    def calc_max_kernel_sizes(self) -> List[Tuple[int, int]]:
        "Calculates the max kernel size for each convolutional layer of the feature net."
        max_kernel_list = []
        if self.cnn_layer_info["conv_layer_type"] != "Conv3d":
            height_in, width_in = self.input_tensor.shape[-2:]
            for idx, _ in enumerate(self.channel_size):
                height_out = 1 + (
                    height_in
                    + 2 * self.padding[idx][0]
                    - 1 * (self.kernel_size[idx][0] - 1)
                    - 1
                ) / (self.stride_size[idx][0])
                width_out = 1 + (
                    width_in
                    + 2 * self.padding[idx][0]
                    - 1 * (self.kernel_size[idx][1] - 1)
                    - 1
                ) / (self.stride_size[idx][1])
                max_kernel_sizes = np.array([height_out * 0.2, width_out * 0.2])
                max_kernel_sizes = np.where(max_kernel_sizes < 0, 0, max_kernel_sizes)
                max_kernel_sizes = np.where(max_kernel_sizes > 10, 10, max_kernel_sizes)
                max_kernel_list.append(tuple(max_kernel_sizes))
                height_in = height_out
                width_in = width_out
        else:
            depth_in, height_in, width_in = self.input_tensor.shape[-3:]
            for idx, _ in enumerate(self.channel_size):
                depth_out = 1 + (
                    depth_in
                    + 2 * self.padding[idx][0]
                    - 1 * (self.kernel_size[idx][0] - 1)
                    - 1
                ) / (self.stride_size[idx][0])
                height_out = 1 + (
                    height_in
                    + 2 * self.padding[idx][1]
                    - 1 * (self.kernel_size[idx][1] - 1)
                    - 1
                ) / (self.stride_size[idx][1])
                width_out = 1 + (
                    width_in
                    + 2 * self.padding[idx][2]
                    - 1 * (self.kernel_size[idx][2] - 1)
                    - 1
                ) / (self.stride_size[idx][2])
                max_kernel_sizes = np.array(
                    [depth_out, height_out * 0.2, width_out * 0.2]
                )
                max_kernel_sizes = np.where(max_kernel_sizes < 0, 0, max_kernel_sizes)
                max_kernel_sizes = np.where(max_kernel_sizes > 10, 10, max_kernel_sizes)
                max_kernel_list.append(tuple(max_kernel_sizes))
                height_in = height_out
                width_in = width_out
                depth_in = depth_out

        return max_kernel_list

    def calc_stride_size_ranges(self) -> List[Tuple[int, int]]:
        "Calculates a range of stride sizes for each convolutional layer of the feature net."
        stride_range_list = []
        if self.cnn_layer_info["conv_layer_type"] != "Conv3d":
            height_in, width_in = self.input_tensor.shape[-2:]
            for idx, _ in enumerate(self.channel_size):
                height_out = 1 + (
                    height_in
                    + 2 * self.padding[idx][0]
                    - 1 * (self.kernel_size[idx][0] - 1)
                    - 1
                ) / (self.stride_size[idx][0])
                width_out = 1 + (
                    width_in
                    + 2 * self.padding[idx][0]
                    - 1 * (self.kernel_size[idx][1] - 1)
                    - 1
                ) / (self.stride_size[idx][1])

                min_stride = min(-(-height_out // 200), -(-width_out // 200))
                max_stride = min(-(-height_out // 75), -(-width_out // 75))

                stride_range_list.append((int(min_stride), int(max_stride)))
                height_in = height_out
                width_in = width_out
        else:
            depth_in, height_in, width_in = self.input_tensor.shape[-3:]
            for idx, _ in enumerate(self.channel_size):
                depth_out = 1 + (
                    depth_in
                    + 2 * self.padding[idx][0]
                    - 1 * (self.kernel_size[idx][0] - 1)
                    - 1
                ) / (self.stride_size[idx][0])
                height_out = 1 + (
                    height_in
                    + 2 * self.padding[idx][1]
                    - 1 * (self.kernel_size[idx][1] - 1)
                    - 1
                ) / (self.stride_size[idx][1])
                width_out = 1 + (
                    width_in
                    + 2 * self.padding[idx][2]
                    - 1 * (self.kernel_size[idx][2] - 1)
                    - 1
                ) / (self.stride_size[idx][2])

                min_stride = min(
                    -(-height_out // 100), -(-width_out // 100), -(-depth_out // 100)
                )
                max_stride = min(
                    -(-height_out // 40), -(-width_out // 40), -(-depth_out // 40)
                )

                stride_range_list.append((int(min_stride), int(max_stride)))
                height_in = height_out
                width_in = width_out
                depth_in = depth_out

        return stride_range_list

    def recreate_network(self, shrink_params: bool = False) -> None:
        """Recreates neural networks.

        :param shrink_params: Boolean flag to shrink parameters
        :type shrink_params: bool
        """
        new_feature_net, new_value_net, new_advantage_net = self.build_networks()

        preserve_params_fn = (
            EvolvableCNN.shrink_preserve_parameters
            if shrink_params
            else EvolvableModule.preserve_parameters
        )

        if self.value_net is not None:
            new_value_net = preserve_params_fn(
                old_net=self.value_net, new_net=new_value_net
            )
        if self.advantage_net is not None:
            new_advantage_net = preserve_params_fn(
                old_net=self.advantage_net, new_net=new_advantage_net
            )
        new_feature_net = preserve_params_fn(
            old_net=self.feature_net, new_net=new_feature_net
        )

        self.feature_net, self.value_net, self.advantage_net = (
            new_feature_net,
            new_value_net,
            new_advantage_net,
        )
