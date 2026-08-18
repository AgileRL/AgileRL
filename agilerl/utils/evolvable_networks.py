# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any, Literal, Protocol, TypeGuard, TypeVar, overload

import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch._dynamo.eval_frame import OptimizedModule

from agilerl.modules import EvolvableModule, ModuleDict
from agilerl.modules.configs import (
    CnnNetConfig,
    LstmNetConfig,
    MlpNetConfig,
    MultiInputNetConfig,
    NetConfig,
    SimBaNetConfig,
)
from agilerl.modules.custom_components import (
    GumbelSoftmax,
    NewGELU,
    NoisyLinear,
    ResidualBlock,
    SimbaResidualBlock,
)
from agilerl.typing import DeviceType, KernelSizeType, NetConfigType

ConvBlockType = Literal["Conv1d", "Conv2d", "Conv3d"]


# Arity of the size tuple a torch layer expects (e.g. ``tuple[int, int]`` for
# 2d layers). Ties a layer constructor to a normalizer that produces it.
_SizeT = TypeVar("_SizeT")


@overload
def _size_normalizer(dims: Literal[1]) -> Callable[[KernelSizeType], tuple[int]]: ...
@overload
def _size_normalizer(
    dims: Literal[2],
) -> Callable[[KernelSizeType], tuple[int, int]]: ...
@overload
def _size_normalizer(
    dims: Literal[3],
) -> Callable[[KernelSizeType], tuple[int, int, int]]: ...
def _size_normalizer(dims: int) -> Callable[[KernelSizeType], tuple[int, ...]]:
    """Build a normalizer that coerces an int-or-tuple size to ``dims`` arity."""

    def normalize(size: KernelSizeType) -> tuple[int, ...]:
        # A tuple already carries per-dimension sizes; anything else (a Python
        # ``int`` or a numpy integer scalar, which is not sliceable) is a scalar
        # broadcast across every dimension.
        if isinstance(size, tuple):
            return tuple(int(s) for s in size[:dims])
        return (int(size),) * dims

    return normalize


class _ConvConstructor(Protocol[_SizeT]):
    """A torch conv constructor whose size args all share arity ``_SizeT``."""

    def __call__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _SizeT,
        stride: _SizeT,
        padding: _SizeT,
        *,
        device: DeviceType,
    ) -> nn.Module: ...


class _PoolConstructor(Protocol[_SizeT]):
    """A torch pooling constructor whose size args accept int or arity ``_SizeT``."""

    def __call__(
        self,
        kernel_size: int | _SizeT,
        stride: int | _SizeT,
        padding: int | _SizeT,
    ) -> nn.Module: ...


def _build_conv(
    conv_cls: _ConvConstructor[_SizeT],
    to_size: Callable[[KernelSizeType], _SizeT],
    in_channels: int,
    out_channels: int,
    kernel_size: KernelSizeType,
    stride: KernelSizeType,
    padding: KernelSizeType,
    device: DeviceType,
) -> nn.Module:
    """Construct ``conv_cls`` with each size normalized to its expected arity."""
    return conv_cls(
        in_channels,
        out_channels,
        to_size(kernel_size),
        to_size(stride),
        to_size(padding),
        device=device,
    )


def _build_pool(
    pool_cls: _PoolConstructor[_SizeT],
    to_size: Callable[[KernelSizeType], _SizeT],
    kernel_size: KernelSizeType,
    stride: KernelSizeType,
    padding: KernelSizeType,
) -> nn.Module:
    """Construct ``pool_cls``, preserving scalar sizes when all args are ints."""
    if (
        not isinstance(kernel_size, tuple)
        and not isinstance(stride, tuple)
        and not isinstance(padding, tuple)
    ):
        # All scalar (Python int or numpy integer): keep scalar kwargs so the
        # layer's repr matches a reference network built with ints.
        return pool_cls(int(kernel_size), int(stride), int(padding))
    return pool_cls(to_size(kernel_size), to_size(stride), to_size(padding))


def _is_module_dict(model: nn.Module) -> TypeGuard[ModuleDict[nn.Module]]:
    """Narrow a module to a ``ModuleDict`` whose values are plain ``nn.Module``."""
    return isinstance(model, ModuleDict)


@overload
def compile_model(
    model: ModuleDict[EvolvableModule],
    mode: str | None = "default",
) -> ModuleDict[EvolvableModule]: ...


@overload
def compile_model(
    model: nn.Module,
    mode: str | None = "default",
) -> nn.Module: ...


def compile_model(
    model: nn.Module,
    mode: str | None = "default",
) -> nn.Module:
    """Compiles torch model if not already compiled.

    :param model: torch model
    :type model: nn.Module | ModuleDict[EvolvableModule]
    :param mode: torch compile mode, defaults to "default"
    :type mode: str, optional
    :return: compiled model (returned unchanged when ``mode`` is None or the
        model is already compiled)
    :rtype: nn.Module
    """
    if _is_module_dict(model):
        compiled_model = ModuleDict()
        for agent_id, module in model.items():
            compiled_model.add_module(agent_id, compile_model(module, mode))
        compiled_model.last_mutation_attr = model.last_mutation_attr
        return compiled_model

    if not isinstance(model, OptimizedModule) and mode is not None:
        # torch.compile's overloads type a Module input as a bare callable;
        # at runtime compiling an nn.Module always yields an OptimizedModule.
        compiled: Any = torch.compile(model, mode=mode, dynamic=True)
        return compiled

    return model


def is_mlp_net_config(config: NetConfigType) -> bool:
    """Check if the net config is a MLP net config.

    :param config: Net config
    :type config: NetConfigType
    :return: True if the net config is a MLP net config, False otherwise
    :rtype: bool
    """
    return "hidden_size" in config and "num_blocks" not in config


def is_image_space(space: spaces.Space) -> bool:
    """Check if the space is an image space. We ignore dtype and number of channels
    checks.

    :param space: Input space
    :type space: spaces.Space

    :return: True if the space is an image space, False otherwise
    :rtype: bool
    """
    return isinstance(space, spaces.Box) and len(space.shape) == 3


def is_box_space_ndim(space: spaces.Space, ndim: int) -> bool:
    """Check if the space is a Box space with the given number of dimensions.

    :param space: Input space
    :type space: spaces.Space
    :param ndim: Number of dimensions
    :type ndim: int

    :return: True if the space is a Box space with the given number of dimensions, False otherwise
    """
    return isinstance(space, spaces.Box) and len(space.shape) == ndim


def is_vector_space(space: spaces.Space) -> bool:
    """Check if the space is a vector space.

    :param space: Input space
    :type space: spaces.Space

    :return: True if the space is a vector space, False otherwise
    :rtype: bool
    """
    return (isinstance(space, spaces.Box) and len(space.shape) in [0, 1]) or isinstance(
        space, (spaces.Discrete, spaces.MultiDiscrete)
    )


def config_from_dict(config_dict: NetConfigType) -> NetConfig:
    """Get the class of the net config from the dictionary.

    :param config_dict: The dictionary to get the class from.
    :type config_dict: NetConfigType
    :return: The net config class.
    :rtype: NetConfig
    """
    if isinstance(config_dict, NetConfig):
        return config_dict
    config_keys = config_dict.keys()
    if "hidden_state_size" in config_keys:
        config_cls = LstmNetConfig
    elif "hidden_size" in config_keys:
        if "num_blocks" in config_keys:
            config_cls = SimBaNetConfig
        else:
            config_cls = MlpNetConfig
    elif "channel_size" in config_keys:
        config_cls = CnnNetConfig
    elif any(key in MultiInputNetConfig.__dataclass_fields__ for key in config_keys):
        config_cls = MultiInputNetConfig
    else:
        msg = (
            f"Unable to determine net config class from: {config_dict}. "
            "Please verify that the keys correspond to the arguments of the net config class."
        )
        raise ValueError(
            msg,
        )

    return config_cls.from_dict(config_dict)


def tuple_to_dict_space(tuple_space: spaces.Tuple) -> spaces.Dict:
    """Convert a Tuple observation space to a Dict observation space.

    :param tuple_space: Tuple observation space
    :type tuple_space: spaces.Tuple
    :return: Dictionary observation space
    :rtype: spaces.Dict
    """
    return spaces.Dict({str(i): space for i, space in enumerate(tuple_space.spaces)})


def tuple_to_dict_obs(tuple_obs: tuple) -> dict:
    """Convert a tuple observation to a Python dictionary.

    :param tuple_obs: Tuple observation
    :type tuple_obs: tuple
    :return: Dictionary observation
    :rtype: dict
    """
    return {str(i): obs for i, obs in enumerate(tuple_obs)}


def get_default_encoder_config(
    observation_space: spaces.Space,
    simba: bool = False,
    recurrent: bool = False,
    layer_norm: bool = True,
) -> NetConfigType:
    """Get the default configuration for the encoder network based on the observation space.

    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
    :param simba: Whether to use SimBA encoder.
    :type simba: bool
    :param recurrent: Whether to use recurrent encoder.
    :type recurrent: bool
    :param layer_norm: Whether to use layer normalization in MLP encoders.
    :type layer_norm: bool
    :return: Default configuration for the encoder network.
    :rtype: NetConfigType
    """
    default_oa = "ReLU"
    if isinstance(observation_space, (spaces.Dict, spaces.Tuple)):
        config = MultiInputNetConfig(output_activation=default_oa)
    elif is_image_space(observation_space):
        config = CnnNetConfig(
            channel_size=[32, 32],
            kernel_size=[3, 3],
            stride_size=[1, 1],
            output_activation=default_oa,
        )
    elif simba:
        config = SimBaNetConfig(
            hidden_size=128,
            num_blocks=2,
            output_activation=default_oa,
        )
    elif recurrent:
        config = LstmNetConfig(
            hidden_state_size=128,
            num_layers=2,
            output_activation=default_oa,
        )
    else:
        config = MlpNetConfig(
            hidden_size=[64, 64],
            output_activation=default_oa,
            layer_norm=layer_norm,
            output_vanish=False,
        )

    return asdict(config)


def contains_moduledict(module: nn.Module) -> bool:
    """Check if a module contains a ModuleDict.

    :param module: Input module
    :type module: nn.Module

    :return: True if module contains a ModuleDict, False otherwise
    :rtype: bool
    """
    return any(isinstance(submodule, nn.ModuleDict) for submodule in module.modules())


def get_module_dict(module: nn.Module) -> nn.ModuleDict | None:
    """Get the ModuleDict from a module.

    :param module: Input module
    :type module: nn.Module

    :return: ModuleDict from module, or None if the module contains none
    :rtype: nn.ModuleDict | None
    """
    for submodule in module.modules():
        if isinstance(submodule, nn.ModuleDict):
            return submodule
    return None


def get_batch_norm_layer(
    name: str,
    num_features: int,
    device: DeviceType = "cpu",
) -> nn.Module:
    """Return batch normalization layer for corresponding batch normalization name.

    :param name: Batch normalization layer name
    :type name: str
    :param layer_size: The layer after which the batch normalization layer will be applied
    :type layer_size: int

    :return: Batch normalization layer
    :rtype: nn.Module
    """
    batch_norm_layers = {
        "1d": nn.BatchNorm1d,
        "2d": nn.BatchNorm2d,
        "3d": nn.BatchNorm3d,
    }

    return batch_norm_layers[name](num_features, device=device)


def get_conv_layer(
    conv_layer_name: ConvBlockType,
    in_channels: int,
    out_channels: int,
    kernel_size: KernelSizeType,
    stride: KernelSizeType = 1,
    padding: KernelSizeType = 0,
    device: DeviceType = "cpu",
) -> nn.Module:
    """Return convolutional layer for corresponding convolutional layer name.

    :param conv_layer_name: Convolutional layer name
    :type conv_layer_name: str
    :param in_channels: Number of input channels to convolutional layer
    :type in_channels: int
    :param out_channels: Number of output channels from convolutional layer
    :type out_channels: int
    :param kernel_size: Kernel size of convolutional layer
    :type kernel_size: int or tuple[int]
    :param stride: Stride size of convolutional layer
    :type stride: int or tuple[int]
    :param padding: Convolutional layer padding
    :type padding: int or tuple[int]

    :return: Convolutional layer
    :rtype: nn.Module
    """
    args = (in_channels, out_channels, kernel_size, stride, padding)
    if conv_layer_name == "Conv1d":
        return _build_conv(nn.Conv1d, _size_normalizer(1), *args, device=device)
    if conv_layer_name == "Conv2d":
        return _build_conv(nn.Conv2d, _size_normalizer(2), *args, device=device)
    if conv_layer_name == "Conv3d":
        return _build_conv(nn.Conv3d, _size_normalizer(3), *args, device=device)

    msg = f"Invalid convolutional layer {conv_layer_name}. Must be one of 'Conv1d', 'Conv2d', 'Conv3d'."
    raise ValueError(msg)


def get_normalization(
    normalization_name: str,
    layer_size: int,
    device: DeviceType = "cpu",
) -> nn.Module:
    """Return normalization layer for corresponding normalization name.

    :param normalization_names: Normalization layer name
    :type normalization_names: str
    :param layer_size: The layer after which the normalization layer will be applied
    :type layer_size: int

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

    return normalization_functions[normalization_name](layer_size, device=device)


ACTIVATION_FUNCTIONS: dict[str, type[nn.Module]] = {
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
    "GELU": nn.GELU,
    "Identity": nn.Identity,
}


def get_activation(activation_name: str | None, new_gelu: bool = False) -> nn.Module:
    """Return activation function for corresponding activation name.

    :param activation_names: Activation function name
    :type activation_names: str
    """
    activation_functions = dict(ACTIVATION_FUNCTIONS)
    if new_gelu:
        activation_functions["GELU"] = NewGELU

    activation_name = activation_name if activation_name is not None else "Identity"
    if activation_name == "Softmax":
        return nn.Softmax(dim=-1)
    return activation_functions[activation_name]()


def get_pooling(
    pooling_name: str,
    kernel_size: tuple[int, ...] | int,
    stride: tuple[int, ...] | int,
    padding: tuple[int, ...] | int,
) -> nn.Module:
    """Return pooling layer for corresponding activation name.

    :param pooling_names: Pooling layer name
    :type pooling_names: str
    :param kernel_size: Pooling layer kernel size
    :type kernel_size: int or tuple[int]
    :param stride: Pooling layer stride
    :type stride: int or tuple[int]
    :param padding: Pooling layer padding
    :type padding: int or tuple[int]

    :return: Pooling layer
    :rtype: nn.Module
    """
    args = (kernel_size, stride, padding)
    if pooling_name == "MaxPool2d":
        return _build_pool(nn.MaxPool2d, _size_normalizer(2), *args)
    if pooling_name == "MaxPool3d":
        return _build_pool(nn.MaxPool3d, _size_normalizer(3), *args)
    if pooling_name == "AvgPool2d":
        return _build_pool(nn.AvgPool2d, _size_normalizer(2), *args)
    if pooling_name == "AvgPool3d":
        return _build_pool(nn.AvgPool3d, _size_normalizer(3), *args)

    msg = f"Invalid pooling layer {pooling_name}. Must be one of 'MaxPool2d', 'MaxPool3d', 'AvgPool2d', 'AvgPool3d'."
    raise ValueError(msg)


LayerType = nn.Module | GumbelSoftmax | NoisyLinear
LayerT = TypeVar("LayerT", bound=nn.Module)


def layer_init(
    layer: LayerT,
    std: float = np.sqrt(2),
    bias_const: float = 0.0,
) -> LayerT:
    """Initialize the weights and biases of a layer.

    :param layer: The layer to initialize.
    :type layer: nn.Module
    :param std: The standard deviation for weight initialization. Defaults to np.sqrt(2).
    :type std: float, optional
    :param bias_const: The constant value for bias initialization. Defaults to 0.0.
    :type bias_const: float, optional

    :return: The initialized layer.
    :rtype: nn.Module
    """
    weight = getattr(layer, "weight", None)
    weight_mu = getattr(layer, "weight_mu", None)
    weight_sigma = getattr(layer, "weight_sigma", None)
    if isinstance(weight, torch.Tensor):
        torch.nn.init.orthogonal_(weight, std)
    elif isinstance(weight_mu, torch.Tensor) and isinstance(weight_sigma, torch.Tensor):
        torch.nn.init.orthogonal_(weight_mu, std)
        torch.nn.init.orthogonal_(weight_sigma, std)

    bias = getattr(layer, "bias", None)
    bias_mu = getattr(layer, "bias_mu", None)
    bias_sigma = getattr(layer, "bias_sigma", None)
    if isinstance(bias, torch.Tensor):
        torch.nn.init.constant_(bias, bias_const)
    elif isinstance(bias_mu, torch.Tensor) and isinstance(bias_sigma, torch.Tensor):
        torch.nn.init.constant_(bias_mu, bias_const)
        torch.nn.init.constant_(bias_sigma, bias_const)

    return layer


def init_weights_gaussian(m: nn.Module, mean: float, std: float) -> None:
    """Initialize weights of a module using Gaussian distribution.

    :param m: Module to initialize
    :type m: nn.Module
    :param mean: Mean of the Gaussian distribution
    :type mean: float
    :param std: Standard deviation of the Gaussian distribution
    :type std: float
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=mean, std=std)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def create_cnn(
    block_type: ConvBlockType,
    in_channels: int,
    channel_size: Sequence[int],
    kernel_size: Sequence[KernelSizeType],
    stride_size: Sequence[KernelSizeType],
    name: str = "cnn",
    init_layers: bool = True,
    layer_norm: bool = False,
    activation_fn: str = "ReLU",
    device: DeviceType = "cpu",
) -> OrderedDict[str, nn.Module]:
    """Build a convolutional block.

    :param block_type: Type of convolutional block.
    :type block_type: Literal["Conv1d", "Conv2d", "Conv3d"]
    :param in_channels: Number of input channels.
    :type in_channels: int
    :param channel_size: Channel sizes for each layer.
    :type channel_size: Sequence[int]
    :param kernel_size: Kernel sizes for each layer.
    :type kernel_size: Sequence[int | tuple[int, ...]]
    :param stride_size: Stride sizes for each layer.
    :type stride_size: Sequence[int | tuple[int, ...]]
    :param name: Name of the block.
    :type name: str
    :param init_layers: Whether to initialize the layers. Defaults to True.
    :type init_layers: bool, optional
    :param layer_norm: Whether to use layer normalization. Defaults to False.
    :type layer_norm: bool, optional
    :param activation_fn: Activation function to use. Defaults to "ReLU".
    :type activation_fn: str, optional
    :param device: Device to use. Defaults to "cpu".
    :type device: DeviceType, optional

    :return: Convolutional block, as named modules for ``nn.Sequential``.
    :rtype: OrderedDict[str, nn.Module]
    """
    net_dict: OrderedDict[str, nn.Module] = OrderedDict()
    channel_size = [in_channels, *channel_size]
    for l_no in range(1, len(channel_size)):
        net_dict[f"{name}_conv_layer_{l_no!s}"] = get_conv_layer(
            conv_layer_name=block_type,
            in_channels=channel_size[l_no - 1],
            out_channels=channel_size[l_no],
            kernel_size=kernel_size[l_no - 1],
            stride=stride_size[l_no - 1],
            device=device,
        )
        if init_layers:
            net_dict[f"{name}_conv_layer_{l_no!s}"] = layer_init(
                net_dict[f"{name}_conv_layer_{l_no!s}"],
            )
        if layer_norm:
            net_dict[f"{name}_layer_norm_{l_no!s}"] = get_batch_norm_layer(
                block_type.replace("Conv", ""),
                num_features=channel_size[l_no],
                device=device,
            )
        net_dict[f"{name}_activation_{l_no!s}"] = get_activation(activation_fn)

    return net_dict


def create_mlp(
    input_size: int,
    output_size: int,
    hidden_size: list[int],
    output_vanish: bool,
    output_activation: str | None = None,
    noisy: bool = False,
    init_layers: bool = True,
    layer_norm: bool = False,
    output_layernorm: bool = False,
    activation: str = "ReLU",
    noise_std: float = 0.1,
    device: DeviceType = "cpu",
    new_gelu: bool = False,
    name: str = "mlp",
) -> nn.Sequential:
    """Create and returns multi-layer perceptron.

    :param input_size: Number of input features.
    :type input_size: int
    :param output_size: Number of output features.
    :type output_size: int
    :param hidden_size: List of hidden layer sizes.
    :type hidden_size: list[int]
    :param output_vanish: Whether to initialize output layer weights to a small value.
    :type output_vanish: bool
    :param output_activation: Activation function for output layer.
    :type output_activation: str | None
    :param noisy: Whether to use noisy layers.
    :type noisy: bool, optional
    :param init_layers: Whether to initialize the layers.
    :type init_layers: bool, optional
    :param layer_norm: Whether to use layer normalization.
    :type layer_norm: bool, optional
    :param output_layernorm: Whether to use layer normalization for the output layer.
    :type output_layernorm: bool, optional
    :param activation: Activation function for hidden layers.
    :type activation: str, optional
    :param noise_std: Standard deviation of noise for noisy layers.
    :type noise_std: float, optional
    :param name: Name of the network.
    :type name: str, default "mlp"

    :return: Multi-layer perceptron.
    :rtype: nn.Sequential
    """
    net_dict: OrderedDict[str, nn.Module] = OrderedDict()
    hidden_size = [input_size, *hidden_size]
    for l_no in range(1, len(hidden_size)):
        if noisy:  # Add linear layer
            net_dict[f"{name}_linear_layer_{l_no!s}"] = NoisyLinear(
                hidden_size[l_no - 1],
                hidden_size[l_no],
                noise_std,
                device=device,
            )
        else:
            net_dict[f"{name}_linear_layer_{l_no!s}"] = nn.Linear(
                hidden_size[l_no - 1],
                hidden_size[l_no],
                device=device,
            )

        if init_layers:  # Initialize layer weights
            net_dict[f"{name}_linear_layer_{l_no!s}"] = layer_init(
                net_dict[f"{name}_linear_layer_{l_no!s}"],
            )

        if layer_norm:  # Add layer normalization
            net_dict[f"{name}_layer_norm_{l_no!s}"] = nn.LayerNorm(
                hidden_size[l_no],
                device=device,
            )

        # Add activation function
        net_dict[f"{name}_activation_{l_no!s}"] = get_activation(
            activation,
            new_gelu,
        )

    # Output layer
    output_layer: NoisyLinear | nn.Linear
    if noisy:
        noisy_output_layer = NoisyLinear(
            hidden_size[-1],
            output_size,
            noise_std,
            device=device,
        )
        if init_layers:
            noisy_output_layer = layer_init(noisy_output_layer)
        if output_vanish:
            noisy_output_layer.weight_mu.data.mul_(0.1)
            noisy_output_layer.bias_mu.data.mul_(0.1)
            noisy_output_layer.weight_sigma.data.mul_(0.1)
            noisy_output_layer.bias_sigma.data.mul_(0.1)
        output_layer = noisy_output_layer
    else:
        linear_output_layer = nn.Linear(hidden_size[-1], output_size, device=device)
        if init_layers:
            linear_output_layer = layer_init(linear_output_layer)
        if output_vanish:
            linear_output_layer.weight.data.mul_(0.1)
            linear_output_layer.bias.data.mul_(0.1)
        output_layer = linear_output_layer

    net_dict[f"{name}_linear_layer_output"] = output_layer

    if output_layernorm:
        net_dict[f"{name}_layer_norm_output"] = nn.LayerNorm(
            output_size,
            device=device,
            elementwise_affine=False,
        )

    net_dict[f"{name}_activation_output"] = get_activation(
        activation_name=output_activation,
        new_gelu=new_gelu,
    )
    return nn.Sequential(net_dict)


def create_simba(
    input_size: int,
    output_size: int,
    hidden_size: int,
    num_blocks: int,
    output_activation: str | None = None,
    scale_factor: int = 4,
    device: DeviceType = "cpu",
    name: str = "simba",
) -> nn.Sequential:
    """Create a number of SimBa residual blocks.

    Paper: https://arxiv.org/abs/2410.09754.

    :param input_size: Number of input features.
    :type input_size: int
    :param output_size: Number of output features.
    :type output_size: int
    :param hidden_size: Number of hidden units.
    :type hidden_size: int
    :param num_blocks: Number of residual blocks.
    :type num_blocks: int
    :param output_activation: Activation function for output layer.
    :type output_activation: str | None
    :param scale_factor: Scale factor for the hidden layer.
    :type scale_factor: int, optional
    :param device: Device to use. Defaults to "cpu".
    :type device: DeviceType, optional
    :param name: Name of the network.
    :type name: str, default "simba"

    :return: Residual block.
    :rtype: nn.Sequential
    """
    net_dict: OrderedDict[str, nn.Module] = OrderedDict()

    # Initial dense layer
    input_layer = nn.Linear(input_size, hidden_size, device=device)
    nn.init.orthogonal_(input_layer.weight)
    net_dict[f"{name}_linear_layer_input"] = input_layer
    for l_no in range(1, num_blocks + 1):
        net_dict[f"{name}_residual_block_{l_no!s}"] = SimbaResidualBlock(
            hidden_size,
            scale_factor=scale_factor,
            device=device,
        )

    # Final layer norm and output dense
    net_dict[f"{name}_layer_norm_output"] = nn.LayerNorm(hidden_size, device=device)
    output_layer = nn.Linear(hidden_size, output_size, device=device)
    nn.init.orthogonal_(output_layer.weight)
    net_dict[f"{name}_linear_layer_output"] = output_layer

    net_dict[f"{name}_activation_output"] = get_activation(
        activation_name=output_activation,
    )

    return nn.Sequential(net_dict)


def create_resnet(
    input_channels: int,
    channel_size: int,
    kernel_size: int,
    stride_size: int,
    num_blocks: int,
    scale_factor: int = 4,
    device: DeviceType = "cpu",
    name: str = "resnet",
) -> OrderedDict[str, nn.Module]:
    """Create a number of residual blocks for image-based inputs.

    Paper: https://arxiv.org/abs/1512.03385.

    :param input_channels: Number of input channels.
    :type input_channels: int
    :param channel_size: Number of output channels.
    :type channel_size: int
    :param kernel_size: Kernel size of the convolutional layer.
    :type kernel_size: int
    :param stride_size: Stride size of the convolutional layer.
    :type stride_size: int
    :param num_blocks: Number of residual blocks.
    :type num_blocks: int
    :param scale_factor: Scale factor for the hidden layer.
    :type scale_factor: int, optional
    :param device: Device to use. Defaults to "cpu".
    :type device: DeviceType, optional
    :param name: Name of the network.
    :type name: str, default "resnet"

    :return: Residual block, as named modules for ``nn.Sequential``.
    :rtype: OrderedDict[str, nn.Module]
    """
    net_dict: OrderedDict[str, nn.Module] = OrderedDict()

    # Initial convolutional layer
    input_layer = nn.Conv2d(
        input_channels,
        channel_size,
        kernel_size=kernel_size,
        stride=stride_size,
        padding=(kernel_size - 1) // 2,
        bias=False,
        device=device,
    )
    nn.init.kaiming_uniform_(input_layer.weight)
    net_dict[f"{name}_conv_input"] = input_layer

    for l_no in range(1, num_blocks + 1):
        net_dict[f"{name}_residual_block_{l_no}"] = ResidualBlock(
            in_channels=channel_size,
            kernel_size=kernel_size,
            scale_factor=scale_factor,
            device=device,
        )

    return net_dict
