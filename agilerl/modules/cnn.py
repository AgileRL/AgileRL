from typing import List, Optional, Tuple, Literal, Union, Dict, Any
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn

from agilerl.typing import KernelSizeType
from agilerl.modules.base import EvolvableModule, MutationType, register_mutation_fn
from agilerl.utils.evolvable_networks import (
    get_activation,
    calc_max_kernel_sizes,
    create_cnn
)

BlockType = Literal["Conv2d", "Conv3d"]

def _assert_correct_kernel_sizes(sizes: List[Tuple[int, ...]], block_type: BlockType) -> None:
    """Check that kernel sizes correspond to either 2d or 3d convolutions. We enforce 
    CNN kernels to have the same value for width and height.
    
    :param sizes: Kernel sizes.
    :type sizes:  Union[int, Tuple[int, ...]]
    """
    for k_size in sizes:
        if len(k_size) == 2 and block_type == "Conv2d":
            wh_unique = set(k_size)
        elif len(k_size) == 3 and block_type == "Conv3d":
            wh_unique = set(k_size[-2:])
        else:
            msg = (
                "2 (width x height)" 
                if block_type == "Conv2d" 
                else "3 (depth x width x height)"
            )
            raise ValueError(
                f"Found CNN kernel with length {len(k_size)}. These should ", \
                f"have a length of {msg} for {block_type}."
                )

        assert (
            len(wh_unique) == 1, 
            "AgileRL currently doesnt support having different ", \
            f"values for width and height in a single CNN kernel: {wh_unique}"
        )

@dataclass
class MutableKernelSizes:
    sizes: List[KernelSizeType]
    cnn_block_type: Literal["Conv2d", "Conv3d"]

    def __post_init__(self) -> None:
        tuple_sizes = False
        if isinstance(self.sizes[0], (tuple, list)):
            tuple_sizes = True
            _assert_correct_kernel_sizes(self.sizes, self.cnn_block_type)

        self.tuple_sizes = tuple_sizes

    @property
    def int_sizes(self) -> List[int]:
        if self.tuple_sizes:
            return [k_size[-1] for k_size in self.sizes]
        return self.sizes
    
    def add_layer(self, other: int) -> None:
        if self.tuple_sizes:
            if self.cnn_block_type == "Conv2d":
                other = (other, other)
            else:
                # NOTE: Do we always want to use a depth of one when adding a layer
                # in multi-agent settings?
                other = (1, other, other)

        self.sizes += [other]
    
    def remove_layer(self) -> None:
        self.sizes = self.sizes[:-1]
    
    def calc_max_kernel_sizes(
            self,
            channel_size: List[int],
            stride_size: List[int],
            input_shape: List[int]
            ) -> List[int]:
        """Calculate the maximum kernel sizes for each convolutional layer.

        :param channel_size: Number of channels in each convolutional layer.
        :type channel_size: List[int]
        :param stride_size: Stride size of each convolutional layer.
        :type stride_size: List[int]
        :param input_shape: Input shape.
        :type input_shape: List[int]

        :return: Maximum kernel sizes for each convolutional layer.
        :rtype: List[int]
        """
        kernel_size = self.int_sizes if self.tuple_sizes else self.sizes
        return calc_max_kernel_sizes(channel_size, kernel_size, stride_size, input_shape)
    
    def change_kernel_size(
            self,
            hidden_layer: int,
            channel_size: List[int],
            stride_size: List[int], 
            input_shape: Tuple[int]
            ) -> None:
        """Randomly alters convolution kernel of random CNN layer.

        :param hidden_layer: Depth of hidden layer to change kernel size of.
        :type hidden_layer: int
        """
        max_kernels = self.calc_max_kernel_sizes(
            channel_size, stride_size, input_shape
        )
        new_kernel_size = np.random.randint(1, max_kernels[hidden_layer] + 1)
        if self.tuple_sizes:
            if self.cnn_block_type == "Conv2d":
                self.sizes[hidden_layer] = (new_kernel_size, new_kernel_size)
            else:
                depth = self.sizes[hidden_layer][0]
                self.sizes[hidden_layer] = (depth, new_kernel_size, new_kernel_size)
        else:
            self.sizes[hidden_layer] = np.random.randint(1, max_kernels[hidden_layer] + 1)


class EvolvableCNN(EvolvableModule):
    """
    The Evolvable Convolutional Neural Network class. It supports the evolution of the CNN architecture
    by adding or removing convolutional layers, changing the number of channels in each layer, changing the
    kernel size and stride size of each layer, and changing the number of nodes in the fully connected layer.

    :param input_shape: Input shape
    :type input_shape: List[int]
    :param num_outputs: Action dimension
    :type num_outputs: int
    :param channel_size: CNN channel size
    :type channel_size: List[int]
    :param kernel_size: Convolution kernel size
    :type kernel_size: List[KernelSizeType]
    :param stride_size: Convolution stride size
    :type stride_size: List[int]
    :param sample_input: Sample input tensor, defaults to None
    :type sample_input: Optional[torch.Tensor], optional
    :param block_type: Type of convolutional block, either 'Conv2d' or 'Conv3d', defaults to 'Conv2d'
    :type block_type: Literal["Conv2d", "Conv3d"], optional
    :param activation: CNN activation layer, defaults to 'ReLU'
    :type activation: str, optional
    :param output_activation: MLP output activation layer, defaults to None
    :type output_activation: Optional[str], optional
    :param min_hidden_layers: Minimum number of hidden layers the fully connected layer will shrink down to, defaults to 1
    :type min_hidden_layers: int, optional
    :param max_hidden_layers: Maximum number of hidden layers the fully connected layer will expand to, defaults to 6
    :type max_hidden_layers: int, optional
    :param min_channel_size: Minimum number of channels a convolutional layer can have, defaults to 32
    :type min_channel_size: int, optional
    :param max_channel_size: Maximum number of channels a convolutional layer can have, defaults to 256
    :type max_channel_size: int, optional
    :param layer_norm: Normalization between layers, defaults to False
    :type layer_norm: bool, optional
    :param init_layers: Initialise network layers, defaults to True
    :type init_layers: bool, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param name: Name of the CNN, defaults to 'cnn'
    :type name: str, optional
    """

    def __init__(
        self,
        input_shape: List[int],
        num_outputs: int,
        channel_size: List[int],
        kernel_size: List[KernelSizeType],
        stride_size: List[int],
        sample_input: Optional[torch.Tensor] = None,
        block_type: Literal["Conv2d", "Conv3d"] = "Conv2d",
        activation: str = "ReLU",
        output_activation: Optional[str] = None,
        min_hidden_layers: int = 1,
        max_hidden_layers: int = 6,
        min_channel_size: int = 32,
        max_channel_size: int = 256,
        layer_norm: bool = False,
        init_layers: bool = True,
        device: str = "cpu",
        name: str = "cnn"
        ) -> None:
        super().__init__(device)

        assert len(kernel_size) == len(
            channel_size
        ), "Length of kernel size list must be the same length as channel size list."
        assert len(stride_size) == len(
            channel_size
        ), "Length of stride size list must be the same length as channel size list."
        assert len(input_shape) >= 3, "Input shape must have at least 3 dimensions."
        assert (
            num_outputs > 0
        ), "'num_outputs' cannot be less than or equal to zero, please enter a valid integer."
        assert (
            min_hidden_layers < max_hidden_layers
        ), "'min_hidden_layers' must be less than 'max_hidden_layers."
        assert (
            min_channel_size < max_channel_size
        ), "'min_channel_size' must be less than 'max_channel_size'."

        if block_type == "Conv3d":
            assert (
                sample_input is not None, 
                "Sample input must be provided for 3D convolutional networks."
            )

        self.input_shape = input_shape
        self.channel_size = channel_size
        self.raw_kernel_size = kernel_size
        self.kernel_size = MutableKernelSizes(kernel_size, block_type)
        self.stride_size = stride_size
        self.block_type = block_type
        self.num_outputs = num_outputs
        self.output_activation = output_activation
        self._activation = activation
        self.min_hidden_layers = min_hidden_layers
        self.max_hidden_layers = max_hidden_layers
        self.min_channel_size = min_channel_size
        self.max_channel_size = max_channel_size
        self.layer_norm = layer_norm
        self.init_layers = init_layers
        self.sample_input = sample_input
        self.name = name

        if block_type == "Conv2d":
            sample_input = (
                torch.zeros(1, *input_shape, device=device) 
                if sample_input is None else sample_input
            )
        else:
            assert (
                sample_input is not None,
                "Sample input must be provided for 3D convolutional networks."
            )

        self.model = self.create_cnn(
            in_channels=input_shape[0],
            channel_size=channel_size,
            kernel_size=kernel_size,
            stride_size=stride_size,
            sample_input=sample_input,
            name=self.name,
        )

    @property
    def net_config(self) -> Dict[str, Any]:
        net_config = self.init_dict.copy()
        for attr in ["input_shape", "num_outputs", "device", "name"]:
            net_config.pop(attr, None)
        return net_config

    @property
    def init_dict(self):
        """Returns model information in dictionary."""
        init_dict = {
            "input_shape": self.input_shape,
            "num_outputs": self.num_outputs,
            "channel_size": self.channel_size,
            "kernel_size": self.raw_kernel_size,
            "stride_size": self.stride_size,
            "activation": self.activation,
            "block_type": self.block_type,
            "sample_input": self.sample_input,
            "output_activation": self.output_activation,
            "min_hidden_layers": self.min_hidden_layers,
            "max_hidden_layers": self.max_hidden_layers,
            "min_channel_size": self.min_channel_size,
            "max_channel_size": self.max_channel_size,
            "layer_norm": self.layer_norm,
            "device": self.device,
            "name": self.name
        }
        return init_dict
    
    @property
    def activation(self) -> str:
        """Returns the activation function of the network."""
        return self._activation
    
    @activation.setter
    def activation(self, activation: str) -> None:
        """Sets the activation function of the network."""
        self._activation = activation

    def init_weights_gaussian(self, std_coeff: float = 4) -> None:
        """Initialise weights of linear layer using Gaussian distribution."""
        # Output layer is initialised with std_coeff=2
        output_layer = self.get_output_dense()
        EvolvableModule.init_weights_gaussian(output_layer, std_coeff=std_coeff)
    
    def get_output_dense(self) -> torch.nn.Module:
        """Returns output layer of neural network."""
        return getattr(self.model, f"{self.name}_linear_output")
    
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
            self.output_activation = activation

        self.activation = activation
        self.recreate_network()

    def create_cnn(
        self,
        in_channels: int,
        channel_size: List[int],
        kernel_size: List[KernelSizeType],
        stride_size: List[int],
        sample_input: torch.Tensor,
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
        net_dict = create_cnn(
            block_type=self.block_type,
            in_channels=in_channels,
            channel_size=channel_size,
            kernel_size=kernel_size,
            stride_size=stride_size,
            name=name,
            init_layers=self.init_layers,
            layer_norm=self.layer_norm,
            activation_fn=self.activation,
            device=self.device
        )

        # Flatten image encodings and pass through a linear layer
        pre_flatten_output = nn.Sequential(net_dict)(sample_input)
        net_dict[f"{name}_flatten"] = nn.Flatten()
        with torch.no_grad():
            cnn_output = nn.Sequential(net_dict)(sample_input)
            flattened_size = cnn_output.shape[1]

        net_dict[f"{name}_linear_output"] = nn.Linear(
            flattened_size, self.num_outputs, device=self.device
            )
        net_dict[f"{name}_output_activation"] = get_activation(
            self.output_activation
        )

        self.cnn_output_size = pre_flatten_output.shape

        return nn.Sequential(net_dict)

    def reset_noise(self) -> None:
        """Resets noise of the model layers."""
        EvolvableModule.reset_noise(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns output of neural network.

        :param x: Neural network input
        :type x: torch.Tensor()

        :return: Output of the neural network
        :rtype: torch.Tensor
        """
        # Handle single-image input for 3D convolutions
        if self.block_type == "Conv3d" and len(x.shape) == 4:
            x = x.unsqueeze(2)

        return self.model(x)

    @register_mutation_fn(MutationType.LAYER)
    def add_layer(self) -> None:
        """Adds a hidden layer to convolutional neural network."""
        max_kernels = self.kernel_size.calc_max_kernel_sizes(
            self.channel_size, self.stride_size, self.input_shape
        )
        if (
            len(self.channel_size) < self.max_hidden_layers
            and not any(i <= 2 for i in self.cnn_output_size[-2:])
            and max_kernels[-1] > 2
        ):  # HARD LIMIT
            self.channel_size += [self.channel_size[-1]]
            k_size = np.random.randint(2, max_kernels[-1] + 1)
            self.kernel_size.add_layer(k_size)
            self.stride_size = self.stride_size + [
                np.random.randint(1, self.stride_size[-1] + 1)
            ]
            self.recreate_network()
        else:
            self.add_channel()

    @register_mutation_fn(MutationType.LAYER)
    def remove_layer(self) -> None:
        """Removes a hidden layer from convolutional neural network."""
        if len(self.channel_size) > self.min_hidden_layers:
            self.channel_size = self.channel_size[:-1]
            self.kernel_size.remove_layer()
            self.stride_size = self.stride_size[:-1]
            self.recreate_network(shrink_params=True)
        else:
            self.add_channel()

    @register_mutation_fn(MutationType.NODE)
    def change_kernel(self) -> None:
        """Randomly alters convolution kernel of random CNN layer."""
        if len(self.channel_size) > 1:
            hidden_layer = np.random.randint(1, min(4, len(self.channel_size)), 1)[0]
            self.kernel_size.change_kernel_size(
                hidden_layer, self.channel_size, self.stride_size, self.input_shape
                )
            self.recreate_network()
        else:
            self.add_layer()

    @register_mutation_fn(MutationType.NODE)
    def add_channel(
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
        
        # Randomly choose number of channels to add
        if numb_new_channels is None:
            numb_new_channels = np.random.choice([8, 16, 32], 1)[0]

        # HARD LIMIT
        if self.channel_size[hidden_layer] + numb_new_channels <= self.max_channel_size:
            self.channel_size[hidden_layer] += numb_new_channels
            self.recreate_network()

        return {"hidden_layer": hidden_layer, "numb_new_channels": numb_new_channels}

    @register_mutation_fn(MutationType.NODE)
    def remove_channel(
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
            self.recreate_network(shrink_params=True)

        return {"hidden_layer": hidden_layer, "numb_new_channels": numb_new_channels}

    def recreate_network(self, shrink_params: bool = False) -> None:
        """Recreates neural networks.

        :param shrink_params: Flag indicating whether to shrink the parameters, defaults to False
        :type shrink_params: bool, optional
        """
        sample_input = (
            torch.zeros(1, *self.input_shape, device=self.device) 
            if self.sample_input is None else self.sample_input
        )

        # Create model with new architecture
        model = self.create_cnn(
            in_channels=self.input_shape[0],
            channel_size=self.channel_size,
            kernel_size=self.kernel_size.sizes,
            stride_size=self.stride_size,
            sample_input=sample_input,
            name=self.name
        )

        # Copy parameters from old model to new model
        preserve_params_fn = (
            self.shrink_preserve_parameters if shrink_params else self.preserve_parameters
        )
        self.model = preserve_params_fn(old_net=self.model, new_net=model)
