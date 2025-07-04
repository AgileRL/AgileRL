from collections import OrderedDict
from dataclasses import asdict
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from accelerate.optimizer import AcceleratedOptimizer
from gymnasium import spaces
from torch._dynamo.eval_frame import OptimizedModule
from torch.optim import Optimizer

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
from agilerl.typing import DeviceType, GymSpaceType, NetConfigType

TupleorInt = Union[Tuple[int, ...], int]


def get_input_size_from_space(
    observation_space: GymSpaceType,
) -> Union[int, Dict[str, int], Tuple[int, ...]]:
    """Returns the dimension of the state space as it pertains to the underlying
    networks (i.e. the input size of the networks).

    :param observation_space: The observation space of the environment.
    :type observation_space: spaces.Space or List[spaces.Space] or Dict[str, spaces.Space].

    :return: The dimension of the state space.
    :rtype: Union[int, Dict[str, int], Tuple[int, ...]]
    """
    if isinstance(observation_space, (list, tuple, spaces.Tuple)):
        return tuple(get_input_size_from_space(space) for space in observation_space)
    elif isinstance(observation_space, (spaces.Dict, dict)):
        return {
            key: get_input_size_from_space(subspace)
            for key, subspace in observation_space.items()
        }
    elif isinstance(observation_space, spaces.Discrete):
        return (observation_space.n,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        return (sum(observation_space.nvec),)
    elif isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.MultiBinary):
        return (observation_space.n,)
    else:
        raise AttributeError(
            f"Can't access state dimensions for {type(observation_space)} spaces."
        )


def get_output_size_from_space(
    action_space: GymSpaceType,
) -> Union[int, Dict[str, int], Tuple[int, ...]]:
    """Returns the dimension of the action space as it pertains to the underlying
    networks (i.e. the output size of the networks).

    :param action_space: The action space of the environment.
    :type action_space: spaces.Space or list[spaces.Space] or Dict[str, spaces.Space].

    :return: The dimension of the action space.
    :rtype: Union[int, Dict[str, int], Tuple[int, ...]]
    """
    if isinstance(action_space, (list, tuple)):
        return tuple(get_output_size_from_space(space) for space in action_space)
    elif isinstance(action_space, (spaces.Dict, dict)):
        return {
            key: get_output_size_from_space(subspace)
            for key, subspace in action_space.items()
        }
    elif isinstance(action_space, spaces.MultiBinary):
        return action_space.n
    elif isinstance(action_space, spaces.Discrete):
        return action_space.n
    elif isinstance(action_space, spaces.MultiDiscrete):
        return sum(action_space.nvec)
    elif isinstance(action_space, spaces.Box):
        # NOTE: Assume continuous actions are always one-dimensional
        return action_space.shape[0]
    else:
        raise AttributeError(
            f"Can't access action dimensions for {type(action_space)} spaces."
        )


def compile_model(
    model: Union[nn.Module, ModuleDict[EvolvableModule]],
    mode: Optional[str] = "default",
) -> Union[OptimizedModule, ModuleDict[EvolvableModule]]:
    """Compiles torch model if not already compiled

    :param model: torch model
    :type model: nn.Module | ModuleDict[EvolvableModule]
    :param mode: torch compile mode, defaults to "default"
    :type mode: str, optional
    :return: compiled model
    :rtype: OptimizedModule | ModuleDict[OptimizedModule]
    """
    if isinstance(model, ModuleDict):
        return ModuleDict(
            {
                agent_id: compile_model(module, mode)
                for agent_id, module in model.items()
            }
        )

    if not isinstance(model, OptimizedModule) and mode is not None:
        compiled_model = torch.compile(model, mode=mode, dynamic=True)
    else:
        compiled_model = model

    return compiled_model


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
    return (
        (isinstance(space, spaces.Box) and len(space.shape) in [0, 1])
        or isinstance(space, spaces.Discrete)
        or isinstance(space, spaces.MultiDiscrete)
    )


def config_from_dict(config_dict: NetConfigType) -> NetConfig:
    """Get the class of the net config from the dictionary.

    :param config_dict: The dictionary to get the class from.
    :type config_dict: NetConfigType
    :return: The net config class.
    :rtype: NetConfig
    """
    config_keys = config_dict.keys()
    if "hidden_size" in config_keys:
        if "num_layers" in config_keys:
            config_cls = LstmNetConfig
        elif "num_blocks" in config_keys:
            config_cls = SimBaNetConfig
        else:
            config_cls = MlpNetConfig
    elif "channel_size" in config_keys:
        config_cls = CnnNetConfig
    elif any(
        key in MultiInputNetConfig.__dataclass_fields__.keys() for key in config_keys
    ):
        config_cls = MultiInputNetConfig
    else:
        raise ValueError(
            f"Unable to determine net config class from: {config_dict}. "
            "Please verify that the keys correspond to the arguments of the net config class."
        )

    return config_cls.from_dict(config_dict)


def tuple_to_dict_space(tuple_space: spaces.Tuple) -> spaces.Dict:
    """Converts a Tuple observation space to a Dict observation space.

    :param tuple_space: Tuple observation space
    :type tuple_space: spaces.Tuple
    :return: Dictionary observation space
    :rtype: spaces.Dict
    """
    return spaces.Dict({str(i): space for i, space in enumerate(tuple_space.spaces)})


def tuple_to_dict_obs(tuple_obs: tuple) -> dict:
    """Converts a tuple observation to a Python dictionary

    :param tuple_obs: Tuple observation
    :type tuple_obs: tuple
    :return: Dictionary observation
    :rtype: dict
    """
    return {str(i): obs for i, obs in enumerate(tuple_obs)}


def get_default_encoder_config(
    observation_space: spaces.Space, simba: bool = False, recurrent: bool = False
) -> NetConfigType:
    """Get the default configuration for the encoder network based on the observation space.

    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
    :param simba: Whether to use SimBA encoder.
    :type simba: bool
    :param recurrent: Whether to use recurrent encoder.
    :type recurrent: bool

    :return: Default configuration for the encoder network.
    :rtype: Dict[str, Any]
    """
    if isinstance(observation_space, (spaces.Dict, spaces.Tuple)):
        config = MultiInputNetConfig(output_activation=None)
    elif is_image_space(observation_space):
        config = CnnNetConfig(
            channel_size=[32, 32],
            kernel_size=[3, 3],
            stride_size=[1, 1],
            output_activation=None,
        )
    elif simba:
        config = SimBaNetConfig(hidden_size=128, num_blocks=2, output_activation=None)
    elif recurrent:
        config = LstmNetConfig(
            hidden_state_size=128, num_layers=2, output_activation=None
        )
    else:
        config = MlpNetConfig(
            hidden_size=[64, 64], output_activation=None, output_vanish=False
        )

    return asdict(config)


def unwrap_optimizer(
    optimizer: Union[Optimizer, AcceleratedOptimizer],
    network: Union[nn.Module, Iterable[nn.Module]],
    lr: int,
):
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
            unwrapped_optimizer: Optimizer = type(optimizer.optimizer)(
                network.parameters(), lr=lr
            )

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


def get_batch_norm_layer(
    name: str, num_features: int, device: DeviceType = "cpu"
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
    conv_layer_name: Literal["Conv2d", "Conv3d"],
    in_channels: int,
    out_channels: int,
    kernel_size: TupleorInt,
    stride: TupleorInt = 1,
    padding: TupleorInt = 0,
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
    :type kernel_size: int or Tuple[int]
    :param stride: Stride size of convolutional layer
    :type stride: int or Tuple[int]
    :param padding: Convolutional layer padding
    :type padding: int or Tuple[int]

    :return: Convolutional layer
    :rtype: nn.Module
    """
    if conv_layer_name not in ["Conv1d", "Conv2d", "Conv3d"]:
        raise ValueError(
            f"Invalid convolutional layer {conv_layer_name}. Must be one of 'Conv1d', 'Conv2d', 'Conv3d'."
        )

    convolutional_layers = {
        "1d": nn.Conv1d,
        "2d": nn.Conv2d,
        "3d": nn.Conv3d,
    }

    # remove 'Conv' from the name if it is present
    conv_layer_name = conv_layer_name.replace("Conv", "")
    return convolutional_layers[conv_layer_name](
        in_channels, out_channels, kernel_size, stride, padding, device=device
    )


def get_normalization(
    normalization_name: str, layer_size: int, device: DeviceType = "cpu"
) -> nn.Module:
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

    return normalization_functions[normalization_name](layer_size, device=device)


def get_activation(activation_name: Optional[str], new_gelu: bool = False) -> nn.Module:
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
        "GELU": nn.GELU if not new_gelu else NewGELU,
        "Identity": nn.Identity,
    }

    activation_name = activation_name if activation_name is not None else "Identity"
    return (
        activation_functions[activation_name](dim=-1)
        if activation_name == "Softmax"
        else activation_functions[activation_name]()
    )


def get_pooling(
    pooling_name: str,
    kernel_size: Union[Tuple[int, ...], int],
    stride: Union[Tuple[int, ...], int],
    padding: Union[Tuple[int, ...], int],
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


def layer_init(
    layer: LayerType, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
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
    block_type: Literal["Conv2d", "Conv3d"],
    in_channels: int,
    channel_size: List[int],
    kernel_size: List[TupleorInt],
    stride_size: List[TupleorInt],
    name: str = "cnn",
    init_layers: bool = True,
    layer_norm: bool = False,
    activation_fn: str = "ReLU",
    device: DeviceType = "cpu",
) -> Dict[str, nn.Module]:
    """
    Build a convolutional block.

    :param block_type: Type of convolutional block.
    :type block_type: Literal["Conv2d", "Conv3d"]
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
    :param init_layers: Whether to initialize the layers. Defaults to True.
    :type init_layers: bool, optional
    :param layer_norm: Whether to use layer normalization. Defaults to False.
    :type layer_norm: bool, optional
    :param activation_fn: Activation function to use. Defaults to "ReLU".
    :type activation_fn: str, optional
    :param device: Device to use. Defaults to "cpu".
    :type device: DeviceType, optional

    :return: Convolutional block.
    :rtype: Dict[str, nn.Module]
    """
    net_dict = OrderedDict()
    channel_size = [in_channels] + channel_size
    for l_no in range(1, len(channel_size)):
        net_dict[f"{name}_conv_layer_{str(l_no)}"] = get_conv_layer(
            conv_layer_name=block_type,
            in_channels=channel_size[l_no - 1],
            out_channels=channel_size[l_no],
            kernel_size=kernel_size[l_no - 1],
            stride=stride_size[l_no - 1],
            device=device,
        )
        if init_layers:
            net_dict[f"{name}_conv_layer_{str(l_no)}"] = layer_init(
                net_dict[f"{name}_conv_layer_{str(l_no)}"]
            )
        if layer_norm:
            net_dict[f"{name}_layer_norm_{str(l_no)}"] = get_batch_norm_layer(
                block_type.replace("Conv", ""),
                num_features=channel_size[l_no],
                device=device,
            )
        net_dict[f"{name}_activation_{str(l_no)}"] = get_activation(activation_fn)

    return net_dict


MlpLayer = Union[nn.Linear, NoisyLinear, nn.LayerNorm]


def create_mlp(
    input_size: int,
    output_size: int,
    hidden_size: List[int],
    output_vanish: bool,
    output_activation: Optional[str] = None,
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
    net_dict: Dict[str, MlpLayer] = OrderedDict()
    hidden_size = [input_size] + hidden_size
    for l_no in range(1, len(hidden_size)):
        if noisy:  # Add linear layer
            net_dict[f"{name}_linear_layer_{str(l_no)}"] = NoisyLinear(
                hidden_size[l_no - 1], hidden_size[l_no], noise_std, device=device
            )
        else:
            net_dict[f"{name}_linear_layer_{str(l_no)}"] = nn.Linear(
                hidden_size[l_no - 1], hidden_size[l_no], device=device
            )

        if init_layers:  # Initialize layer weights
            net_dict[f"{name}_linear_layer_{str(l_no)}"] = layer_init(
                net_dict[f"{name}_linear_layer_{str(l_no)}"]
            )

        if layer_norm:  # Add layer normalization
            net_dict[f"{name}_layer_norm_{str(l_no)}"] = nn.LayerNorm(
                hidden_size[l_no], device=device
            )

        # Add activation function
        net_dict[f"{name}_activation_{str(l_no)}"] = get_activation(
            activation, new_gelu
        )

    # Output layer
    if noisy:
        output_layer = NoisyLinear(
            hidden_size[-1], output_size, noise_std, device=device
        )
    else:
        output_layer = nn.Linear(hidden_size[-1], output_size, device=device)

    if init_layers:
        output_layer = layer_init(output_layer)

    if output_vanish:
        if noisy:
            output_layer.weight_mu.data.mul_(0.1)
            output_layer.bias_mu.data.mul_(0.1)
            output_layer.weight_sigma.data.mul_(0.1)
            output_layer.bias_sigma.data.mul_(0.1)
        else:
            output_layer.weight.data.mul_(0.1)
            output_layer.bias.data.mul_(0.1)

    net_dict[f"{name}_linear_layer_output"] = output_layer

    if output_layernorm:
        net_dict[f"{name}_layer_norm_output"] = nn.LayerNorm(
            output_size, device=device, elementwise_affine=False
        )

    net_dict[f"{name}_activation_output"] = get_activation(
        activation_name=output_activation, new_gelu=new_gelu
    )
    return nn.Sequential(net_dict)


def create_simba(
    input_size: int,
    output_size: int,
    hidden_size: int,
    num_blocks: int,
    output_activation: Optional[str] = None,
    scale_factor: float = 4.0,
    device: DeviceType = "cpu",
    name: str = "simba",
) -> nn.Sequential:
    """Creates a number of SimBa residual blocks.

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
    :type output_activation: Optional[str]
    :param scale_factor: Scale factor for the hidden layer.
    :type scale_factor: float, optional
    :param device: Device to use. Defaults to "cpu".
    :type device: DeviceType, optional
    :param name: Name of the network.
    :type name: str, default "simba"

    :return: Residual block.
    :rtype: nn.Sequential
    """
    net_dict: Dict[str, nn.Module] = OrderedDict()

    # Initial dense layer
    net_dict[f"{name}_linear_layer_input"] = nn.Linear(
        input_size, hidden_size, device=device
    )
    nn.init.orthogonal_(net_dict[f"{name}_linear_layer_input"].weight)
    for l_no in range(1, num_blocks + 1):
        net_dict[f"{name}_residual_block_{str(l_no)}"] = SimbaResidualBlock(
            hidden_size, scale_factor=scale_factor, device=device
        )

    # Final layer norm and output dense
    net_dict[f"{name}_layer_norm_output"] = nn.LayerNorm(hidden_size, device=device)
    net_dict[f"{name}_linear_layer_output"] = nn.Linear(
        hidden_size, output_size, device=device
    )
    nn.init.orthogonal_(net_dict[f"{name}_linear_layer_output"].weight)

    net_dict[f"{name}_activation_output"] = get_activation(
        activation_name=output_activation
    )

    return nn.Sequential(net_dict)


def create_resnet(
    input_channels: int,
    channel_size: int,
    kernel_size: int,
    stride_size: int,
    num_blocks: int,
    scale_factor: int = 4,
    device: str = "cpu",
    name: str = "resnet",
):
    """Creates a number of residual blocks for image-based inputs."""
    net_dict = OrderedDict()

    # Initial convolutional layer
    net_dict[f"{name}_conv_input"] = nn.Conv2d(
        input_channels,
        channel_size,
        kernel_size=kernel_size,
        stride=stride_size,
        padding=(kernel_size - 1) // 2,
        bias=False,
        device=device,
    )
    nn.init.kaiming_uniform_(net_dict[f"{name}_conv_input"].weight)

    for l_no in range(1, num_blocks + 1):
        net_dict[f"{name}_residual_block_{l_no}"] = ResidualBlock(
            in_channels=channel_size,
            kernel_size=kernel_size,
            scale_factor=scale_factor,
            device=device,
        )

    return net_dict
