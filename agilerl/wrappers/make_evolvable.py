import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agilerl.networks.custom_components import GumbelSoftmax, NoisyLinear


class MakeEvolvable(nn.Module):
    """Wrapper to make a neural network evolvable

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

    def __init__(
        self,
        network,
        input_tensor,
        secondary_input_tensor=None,
        num_atoms=51,
        min_hidden_layers=1,
        max_hidden_layers=3,
        min_mlp_nodes=64,
        max_mlp_nodes=1024,
        min_cnn_hidden_layers=1,
        max_cnn_hidden_layers=6,
        min_channel_size=32,
        max_channel_size=256,
        output_vanish=False,
        init_layers=False,
        support=None,
        rainbow=False,
        device="cpu",
        accelerator=None,
        **kwargs,
    ):
        super().__init__()
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

        self.feature_net, self.value_net, self.advantage_net = self.create_nets()

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
            advantage = self.advantage_net(x)
            if not self.cnn_layer_info:
                value = self.value_net(x)
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

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        """Initialize the weights of a neural network layer using orthogonal initialization and set the biases to a constant value.

        :param layer: Neural network layer
        :type layer: nn.Module
        :param std: Standard deviation, defaults to sqrt(2)
        :type std: float
        :param bias_const: Bias value, defaults to 0.0
        :type bias_const: float
        """
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_pooling(self, pooling_names, kernel_size, stride, padding):
        """Returns pooling layer for corresponding activation name.

        :param pooling_names: Pooling layer name
        :type pooling_names: str
        :param kernel_size: Pooling layer kernel size
        :type kernel_size: int or Tuple[int]
        :param stride: Pooling layer stride
        :type stride: int or Tuple[int]
        :param padding: Pooling layer padding
        :type padding: int or Tuple[int]
        """
        pooling_functions = {
            "MaxPool2d": nn.MaxPool2d,
            "MaxPool3d": nn.MaxPool3d,
            "AvgPool2d": nn.AvgPool2d,
            "AvgPool3d": nn.AvgPool3d,
        }

        return pooling_functions[pooling_names](kernel_size, stride, padding)

    def get_activation(self, activation_names):
        """Returns activation function for corresponding activation name.

        :param activation_names: Activation function name
        :type activation_names: str
        """
        activation_functions = {
            "Tanh": nn.Tanh,
            "Linear": nn.Identity,
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

    def get_normalization(self, normalization_name, layer_size):
        """Returns normalization layer for corresponding normalization name.

        :param normalization_names: Normalization layer name
        :type normalization_names: str
        :param layer_size: The layer after which the normalization layer will be applied
        :param layer_size: int
        """
        normalization_functions = {
            "BatchNorm2d": nn.BatchNorm2d,
            "BatchNorm3d": nn.BatchNorm3d,
            "InstanceNorm2d": nn.InstanceNorm2d,
            "InstanceNorm3d": nn.InstanceNorm3d,
            "LayerNorm": nn.LayerNorm,
        }

        return normalization_functions[normalization_name](layer_size)

    def get_conv_layer(
        self, conv_layer_name, in_channels, out_channels, kernel_size, stride, padding
    ):
        """Return convolutional layer for corresponding convolutional layer name.

        :param conv_layer_name: Convolutional layer name
        :type conv_layer_name: str
        :param in_channels: Number of input channels to convolutional layer
        :type in_channels: int
        :param out_channels: Number of output channels from convolutional layer
        :type out_channels: int
        :param kernel_size: Kernel size of convolutional layer
        :type kernel_size: int or Tuple[int]
        :param stride: Stride size of convolutional layer
        :type stride: int or Tuple[int]
        :param padding: Convolutional layer padding
        :type padding: int or Tuple[int]
        """

        convolutional_layers = {"Conv2d": nn.Conv2d, "Conv3d": nn.Conv3d}

        return convolutional_layers[conv_layer_name](
            in_channels, out_channels, kernel_size, stride, padding
        )

    def detect_architecture(self, network, input_tensor, secondary_input_tensor=None):
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
            def forward_hook(module, input, output):
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
                        cnn_layer_info["pooling_layers"] = dict()  #
                    cnn_layer_info["pooling_layers"][self.conv_counter] = dict()  #
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
        input_size,
        output_size,
        hidden_size,
        name,
        mlp_activation,
        mlp_output_activation,
        noisy=False,
        rainbow_feature_net=False,
    ):
        """Creates and returns multi-layer perceptron.

        :param input_size: Input dimensions to first MLP layer
        :type input_size: int
        :param output_size: Output dimensions from last MLP layer
        :type output_size: int
        :param hidden_size: Hidden layer sizes
        :type hidden_size: list[int]
        :param name: Layer name
        :type name: str
        ####
        :param noisy:
        :type noisy:
        :param rainbow_feature_net:
        :type rainbow_feature_net:
        """

        net_dict = OrderedDict()
        if noisy:
            net_dict[f"{name}_linear_layer_0"] = NoisyLinear(input_size, hidden_size[0])
        else:
            net_dict[f"{name}_linear_layer_0"] = nn.Linear(input_size, hidden_size[0])

        if self.init_layers:
            net_dict[f"{name}_linear_layer_0"] = self.layer_init(
                net_dict[f"{name}_linear_layer_0"]
            )

        if ("norm_layers" in self.mlp_layer_info.keys()) and (
            0 in self.mlp_layer_info["norm_layers"].keys()
        ):
            net_dict[f"{name}_layer_norm_0"] = self.get_normalization(
                self.mlp_layer_info["norm_layers"][0], hidden_size[0]
            )

        if ("activation_layers" in self.mlp_layer_info.keys()) and (
            0 in self.mlp_layer_info["activation_layers"].keys()
        ):
            net_dict[f"{name}_activation_0"] = self.get_activation(
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
                    net_dict[f"{name}_linear_layer_{str(l_no)}"] = self.layer_init(
                        net_dict[f"{name}_linear_layer_{str(l_no)}"]
                    )
                if ("norm_layers" in self.mlp_layer_info.keys()) and (
                    l_no in self.mlp_layer_info["norm_layers"].keys()
                ):
                    net_dict[f"{name}_layer_norm_{str(l_no)}"] = self.get_normalization(
                        self.mlp_layer_info["norm_layers"][l_no], hidden_size[l_no]
                    )
                if l_no in self.mlp_layer_info["activation_layers"].keys():
                    net_dict[f"{name}_activation_{str(l_no)}"] = self.get_activation(
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
                output_layer = self.layer_init(output_layer)

            if self.output_vanish:
                output_layer.weight.data.mul_(0.1)
                output_layer.bias.data.mul_(0.1)

            net_dict[f"{name}_linear_layer_output"] = output_layer
            if mlp_output_activation is not None:
                net_dict[f"{name}_activation_output"] = self.get_activation(
                    mlp_output_activation
                )

        return nn.Sequential(net_dict)

    def create_cnn(
        self, input_size, channel_size, kernel_size, stride_size, padding, name
    ):
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

        net_dict[f"{name}_conv_layer_0"] = self.get_conv_layer(
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
            net_dict[f"{name}_layer_norm_0"] = self.get_normalization(
                self.cnn_layer_info["norm_layers"][0], channel_size[0]
            )

        if ("activation_layers" in self.cnn_layer_info.keys()) and (
            0 in self.cnn_layer_info["activation_layers"].keys()
        ):
            net_dict[f"{name}_activation_0"] = self.get_activation(
                self.cnn_layer_info["activation_layers"][0]
            )

        if ("pooling_layers" in self.cnn_layer_info.keys()) and (
            0 in self.cnn_layer_info["pooling_layers"].keys()
        ):
            net_dict[f"{name}_pooling_0"] = self.get_pooling(
                self.cnn_layer_info["pooling_layers"][0]["name"],
                self.cnn_layer_info["pooling_layers"][0]["kernel"],
                self.cnn_layer_info["pooling_layers"][0]["stride"],
                self.cnn_layer_info["pooling_layers"][0]["padding"],
            )

        if len(channel_size) > 1:
            for l_no in range(1, len(channel_size)):
                net_dict[f"{name}_conv_layer_{str(l_no)}"] = self.get_conv_layer(
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
                    net_dict[f"{name}_layer_norm_{str(l_no)}"] = self.get_normalization(
                        self.cnn_layer_info["norm_layers"][l_no], channel_size[l_no]
                    )

                if ("activation_layers" in self.cnn_layer_info.keys()) and (
                    l_no in self.cnn_layer_info["activation_layers"].keys()
                ):
                    net_dict[f"{name}_activation_{str(l_no)}"] = self.get_activation(
                        self.cnn_layer_info["activation_layers"][l_no]
                    )

                if ("pooling_layers" in self.cnn_layer_info.keys()) and (
                    l_no in self.cnn_layer_info["pooling_layers"].keys()
                ):
                    net_dict[f"{name}_pooling_{str(l_no)}"] = self.get_pooling(
                        self.cnn_layer_info["pooling_layers"][l_no]["name"],
                        self.cnn_layer_info["pooling_layers"][l_no]["kernel"],
                        self.cnn_layer_info["pooling_layers"][l_no]["stride"],
                        self.cnn_layer_info["pooling_layers"][l_no]["padding"],
                    )

        return nn.Sequential(net_dict)

    def create_nets(self):
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
            cnn_output = feature_net(torch.zeros(*self.input_tensor.shape))
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
                    output_size=self.num_atoms * self.num_outputs,  ####
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

    @property
    def init_dict(self):
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

    def add_mlp_layer(self):
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
            self.recreate_nets()
        else:
            self.add_mlp_node()

    def remove_mlp_layer(self):
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

            self.recreate_nets(shrink_params=True)
        else:
            self.add_mlp_node()

    def add_mlp_node(self, hidden_layer=None, numb_new_nodes=None):
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

    def add_cnn_layer(self):
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

            self.recreate_nets()
        else:
            self.add_cnn_channel()

    def remove_cnn_layer(self):
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

            self.recreate_nets()
        else:
            self.add_cnn_channel()

    def calc_max_kernel_sizes(self):
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

    def calc_stride_size_ranges(self):
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

    def change_cnn_kernel(self):
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

    def reset_noise(self):
        """Resets noise of value and advantage networks."""
        if self.rainbow:
            for layer in self.value_net:
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()
            for layer in self.advantage_net:
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()

    def recreate_nets(self, shrink_params=False):
        """Recreates neural networks.

        :param shrink_params: Boolean flag to shrink parameters
        :type shrink_params: bool
        """
        new_feature_net, new_value_net, new_advantage_net = self.create_nets()

        if shrink_params:
            if self.value_net is not None:
                new_value_net = self.shrink_preserve_parameters(
                    old_net=self.value_net, new_net=new_value_net
                )
            if self.advantage_net is not None:
                new_advantage_net = self.shrink_preserve_parameters(
                    old_net=self.advantage_net, new_net=new_advantage_net
                )
            new_feature_net = self.shrink_preserve_parameters(
                old_net=self.feature_net, new_net=new_feature_net
            )
        else:
            if self.value_net is not None:
                new_value_net = self.preserve_parameters(
                    old_net=self.value_net, new_net=new_value_net
                )
            if self.advantage_net is not None:
                new_advantage_net = self.preserve_parameters(
                    old_net=self.advantage_net, new_net=new_advantage_net
                )
            new_feature_net = self.preserve_parameters(
                old_net=self.feature_net, new_net=new_feature_net
            )

        self.feature_net, self.value_net, self.advantage_net = (
            new_feature_net,
            new_value_net,
            new_advantage_net,
        )

    def clone(self):
        """Returns clone of neural net with identical parameters."""
        clone = MakeEvolvable(**copy.deepcopy(self.init_dict))
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
