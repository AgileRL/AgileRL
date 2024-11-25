# This file contains utility functions for tuning
from typing import Tuple, Union, List, Dict, Optional, Iterable
import math
import numpy as np
import torch
from torch.optim import Optimizer
import torch.nn as nn
from collections import OrderedDict
from accelerate.optimizer import AcceleratedOptimizer
from gymnasium import spaces

from agilerl.modules.custom_components import GumbelSoftmax, NoisyLinear, NewGELU

def unwrap_optimizer(
        optimizer: Union[Optimizer, AcceleratedOptimizer],
        network: Union[nn.Module, Iterable[nn.Module]],
        lr: int):
    """Unwrap the optimizer.

    :param optimizer: Optimizer to unwrap
    :type optimizer: Union[Optimizer, AcceleratedOptimizer]
    :param network: Network to unwrap
    :type network: Union[nn.Module, Iterable[nn.Module]]
    :param lr: Learning rate
    :type lr: int

    :return: Unwrapped optimizer
    :rtype: Optimizer
    """
    if isinstance(optimizer, AcceleratedOptimizer):
        if isinstance(network, (list, tuple)):
            optim_arg = [{"params": net.parameters(), "lr": lr} for net in network]
            unwrapped_optimizer: Optimizer = type(optimizer.optimizer)(optim_arg)
        else:
            unwrapped_optimizer: Optimizer = type(optimizer.optimizer)(network.parameters(), lr=lr)

        unwrapped_optimizer.load_state_dict(optimizer.state_dict())
        return unwrapped_optimizer
    else:
        return optimizer

def contains_moduledict(module: nn.Module) -> bool:
    """Check if a module contains a ModuleDict.

    :param module: Input module
    :type module: nn.Module

    :return: True if module contains a ModuleDict, False otherwise
    :rtype: bool
    """
    for submodule in module.modules():
        if isinstance(submodule, nn.ModuleDict):
            return True
    return False

def get_module_dict(module: nn.Module) -> nn.ModuleDict:
    """Get the ModuleDict from a module.

    :param module: Input module
    :type module: nn.Module

    :return: ModuleDict from module
    :rtype: Dict[str, nn.Module]
    """
    for submodule in module.modules():
        if isinstance(submodule, nn.ModuleDict):
            return submodule
    return None

def get_conv_layer(
        conv_layer_name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int, ...], int],
        stride: Union[Tuple[int, ...], int],
        padding: Union[Tuple[int, ...], int]
    ) -> nn.Module:
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

        :return: Convolutional layer
        :rtype: nn.Module
        """
        convolutional_layers = {"Conv1d": nn.Conv1d, "Conv2d": nn.Conv2d, "Conv3d": nn.Conv3d}

        return convolutional_layers[conv_layer_name](
            in_channels, out_channels, kernel_size, stride, padding
        )

def get_normalization(normalization_name: str, layer_size: int) -> nn.Module:
    """Returns normalization layer for corresponding normalization name.

    :param normalization_names: Normalization layer name
    :type normalization_names: str
    :param layer_size: The layer after which the normalization layer will be applied
    :param layer_size: int

    :return: Normalization layer
    :rtype: nn.Module
    """
    normalization_functions = {
        "BatchNorm2d": nn.BatchNorm2d,
        "BatchNorm3d": nn.BatchNorm3d,
        "InstanceNorm2d": nn.InstanceNorm2d,
        "InstanceNorm3d": nn.InstanceNorm3d,
        "LayerNorm": nn.LayerNorm,
    }

    return normalization_functions[normalization_name](layer_size)

def get_activation(activation_name: Optional[str], gpt: bool = False) -> nn.Module:
    """Returns activation function for corresponding activation name.

    :param activation_names: Activation function name
    :type activation_names: str
    """
    activation_functions = {
        "Tanh": nn.Tanh,
        "ReLU": nn.ReLU,
        "ELU": nn.ELU,
        "Softsign": nn.Softsign,
        "Sigmoid": nn.Sigmoid,
        "GumbelSoftmax": GumbelSoftmax,
        "Softplus": nn.Softplus,
        "Softmax": nn.Softmax,
        "LeakyReLU": nn.LeakyReLU,
        "PReLU": nn.PReLU,
        "GELU": nn.GELU if not gpt else NewGELU,
        "Identity": nn.Identity,
    }

    activation_name = activation_name if activation_name is not None else "Identity"
    return (
        activation_functions[activation_name](dim=-1)
        if activation_name == "Softmax" and not gpt
        else activation_functions[activation_name]()
    )

def get_pooling(
          pooling_name: str,
          kernel_size: Union[Tuple[int, ...], int],
          stride: Union[Tuple[int, ...], int],
          padding: Union[Tuple[int, ...], int]
          ) -> nn.Module:
    """Returns pooling layer for corresponding activation name.

    :param pooling_names: Pooling layer name
    :type pooling_names: str
    :param kernel_size: Pooling layer kernel size
    :type kernel_size: int or Tuple[int]
    :param stride: Pooling layer stride
    :type stride: int or Tuple[int]
    :param padding: Pooling layer padding
    :type padding: int or Tuple[int]

    :return: Pooling layer
    :rtype: nn.Module
    """
    pooling_functions = {
        "MaxPool2d": nn.MaxPool2d,
        "MaxPool3d": nn.MaxPool3d,
        "AvgPool2d": nn.AvgPool2d,
        "AvgPool3d": nn.AvgPool3d,
    }

    return pooling_functions[pooling_name](kernel_size, stride, padding)

LayerType = Union[nn.Module, GumbelSoftmax, NoisyLinear]

def layer_init(layer: LayerType, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """
    Initializes the weights and biases of a layer.

    :param layer: The layer to initialize.
    :type layer: nn.Module
    :param std: The standard deviation for weight initialization. Defaults to np.sqrt(2).
    :type std: float, optional
    :param bias_const: The constant value for bias initialization. Defaults to 0.0.
    :type bias_const: float, optional

    :return: The initialized layer.
    :rtype: nn.Module
    """
    if hasattr(layer, "weight"):
        torch.nn.init.orthogonal_(layer.weight, std)

    elif hasattr(layer, "weight_mu") and hasattr(layer, "weight_sigma"):
        torch.nn.init.orthogonal_(layer.weight_mu, std)
        torch.nn.init.orthogonal_(layer.weight_sigma, std)

    if hasattr(layer, "bias"):
        torch.nn.init.constant_(layer.bias, bias_const)

    elif hasattr(layer, "bias_mu") and hasattr(layer, "bias_sigma"):
        torch.nn.init.constant_(layer.bias_mu, bias_const)
        torch.nn.init.constant_(layer.bias_sigma, bias_const)

    return layer

def calc_max_kernel_sizes(
        channel_size: List[int],
        kernel_size: List[int],
        stride_size: List[int],
        input_shape: List[int]
        ) -> List[int]:
    """Calculates the max kernel size for each convolutional layer of the feature net.

    :param channel_size: List of channel sizes for each convolutional layer
    :type channel_size: list[int]
    :param kernel_size: List of kernel sizes for each convolutional layer
    :type kernel_size: list[int]
    :param stride_size: List of stride sizes for each convolutional layer
    :type stride_size: list[int]
    :param input_shape: Input shape of the network
    :type input_shape: list[int]
    :return: List of max kernel sizes for each convolutional layer
    :rtype: list[int]
    """
    max_kernel_list = []
    height_in, width_in = input_shape[-2:]
    for idx, _ in enumerate(channel_size):
        height_out = 1 + np.floor(
            (height_in + 2 * (0) - kernel_size[idx]) / (stride_size[idx])
        )
        width_out = 1 + np.floor(
            (width_in + 2 * (0) - kernel_size[idx]) / (stride_size[idx])
        )
        max_kernel_size = min(height_out, width_out) * 0.25
        max_kernel_size = int(max_kernel_size)
        if max_kernel_size <= 0:
            max_kernel_size = 1
        elif max_kernel_size > 9:
            max_kernel_size = 9
        max_kernel_list.append(max_kernel_size)
        height_in = height_out
        width_in = width_out

    return max_kernel_list
    
def is_image_space(space: spaces.Space) -> bool:
    """Check if the space is an image space. We ignore dtype and number of channels 
    checks.

    :param space: Input space
    :type space: spaces.Space

    :return: True if the space is an image space, False otherwise
    :rtype: bool
    """
    return isinstance(space, spaces.Box) and len(space.shape) == 3

def create_conv_block(
        in_channels: int,
        channel_size: List[int],
        kernel_size: List[int],
        stride_size: List[int],
        name: str,
        critic: bool = False,
        init_layers: bool = True,
        layer_norm: bool = False,
        activation_fn: str = "ReLU",
        n_agents: Optional[int] = None,
        ) -> Dict[str, nn.Module]:
    """
    Build a convolutional block.

    :param in_channels: Number of input channels.
    :type in_channels: int
    :param channel_size: List of channel sizes for each layer.
    :type channel_size: List[int]
    :param kernel_size: List of kernel sizes for each layer.
    :type kernel_size: List[int]
    :param stride_size: List of stride sizes for each layer.
    :type stride_size: List[int]
    :param name: Name of the block.
    :type name: str
    :param critic: Whether the block is for a critic network. Defaults to False.
    :type critic: bool, optional
    :param init_layers: Whether to initialize the layers. Defaults to True.
    :type init_layers: bool, optional
    :param layer_norm: Whether to use layer normalization. Defaults to False.
    :type layer_norm: bool, optional
    :param activation_fn: Activation function to use. Defaults to "ReLU".
    :type activation_fn: str, optional
    :param block_type: Type of convolutional block. Defaults to "Conv2d".
    :type block_type: Literal["Conv1d", "Conv2d", "Conv3d"], optional
    :param n_agents: Number of agents. Defaults to None.
    :type n_agents: int, optional
    
    :return: 3D convolutional block.
    :rtype: Dict[str, nn.Module]
    """
    block_conv_map = {
        "Conv2d": nn.Conv2d,
        "Conv3d": nn.Conv3d,
    }

    block_batch_norm_map = {
        "Conv2d": nn.BatchNorm2d,
        "Conv3d": nn.BatchNorm3d,
    }

    multi = n_agents is not None
    if multi:
        k_size = (
            (n_agents, kernel_size[0], kernel_size[0])
            if critic
            else (1, kernel_size[0], kernel_size[0])
        )
    else:
        k_size = kernel_size

    net_dict = OrderedDict()
    block_type = "Conv2d" if not multi else "Conv3d"
    net_dict[f"{name}_conv_layer_0"] = block_conv_map[block_type](
        in_channels=in_channels,
        out_channels=channel_size[0],
        kernel_size=k_size if multi else kernel_size[0],
        stride=stride_size[0],
    )
    if init_layers:
        net_dict[f"{name}_conv_layer_0"] = layer_init(
            net_dict[f"{name}_conv_layer_0"]
        )
    if layer_norm:
        net_dict[f"{name}_layer_norm_0"] = block_batch_norm_map[block_type](channel_size[0])
    net_dict[f"{name}_activation_0"] = get_activation(activation_fn)

    if len(channel_size) > 1:
        for l_no in range(1, len(channel_size)):
            k_size = (1, kernel_size[l_no], kernel_size[l_no]) if multi else kernel_size[l_no]
            net_dict[f"{name}_conv_layer_{str(l_no)}"] = block_conv_map[block_type](
                in_channels=channel_size[l_no - 1],
                out_channels=channel_size[l_no],
                kernel_size=k_size,
                stride=stride_size[l_no],
            )
            if init_layers:
                net_dict[f"{name}_conv_layer_{str(l_no)}"] = layer_init(
                    net_dict[f"{name}_conv_layer_{str(l_no)}"]
                )
            if layer_norm:
                net_dict[f"{name}_layer_norm_{str(l_no)}"] = block_batch_norm_map[block_type](
                    channel_size[l_no]
                )
            net_dict[f"{name}_activation_{str(l_no)}"] = get_activation(
                activation_fn
            )
    
    return net_dict

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
    gpt_activations: bool = False,
    mlp_activation: str = "ReLU",
    mlp_output_activation: Optional[str] = None,
    noise_std: float = 0.1,
    rainbow: bool = False,
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
    :param gpt_activations: Whether to use GPT activations.
    :type gpt_activations: bool, optional
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
    if noisy:
        net_dict[f"{name}_linear_layer_0"] = NoisyLinear(input_size, hidden_size[0], noise_std)
    else:
        net_dict[f"{name}_linear_layer_0"] = nn.Linear(input_size, hidden_size[0])

    if init_layers:
        net_dict[f"{name}_linear_layer_0"] = layer_init(net_dict[f"{name}_linear_layer_0"])

    if layer_norm:
        net_dict[f"{name}_layer_norm_0"] = nn.LayerNorm(hidden_size[0])
    net_dict[f"{name}_activation_0"] = get_activation(
        activation_name=mlp_output_activation if (len(hidden_size) == 1 and rainbow_feature_net) else mlp_activation,
        gpt=gpt_activations,
    )

    if len(hidden_size) > 1:
        for l_no in range(1, len(hidden_size)):
            if noisy:
                net_dict[f"{name}_linear_layer_{str(l_no)}"] = NoisyLinear(
                    hidden_size[l_no - 1], hidden_size[l_no], noise_std
                )
            else:
                net_dict[f"{name}_linear_layer_{str(l_no)}"] = nn.Linear(
                    hidden_size[l_no - 1], hidden_size[l_no]
                )
            if init_layers:
                net_dict[f"{name}_linear_layer_{str(l_no)}"] = layer_init(net_dict[f"{name}_linear_layer_{str(l_no)}"])
            if layer_norm:
                net_dict[f"{name}_layer_norm_{str(l_no)}"] = nn.LayerNorm(hidden_size[l_no])
            net_dict[f"{name}_activation_{str(l_no)}"] = get_activation(
                mlp_activation if not rainbow_feature_net else mlp_output_activation,
                gpt=gpt_activations,
            )

    if not rainbow_feature_net:
        if noisy:
            output_layer = NoisyLinear(hidden_size[-1], output_size, noise_std)
        else:
            output_layer = nn.Linear(hidden_size[-1], output_size)

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
                activation_name=output_activation,
                gpt=gpt_activations,
            )

    net = nn.Sequential(net_dict)
    return net

