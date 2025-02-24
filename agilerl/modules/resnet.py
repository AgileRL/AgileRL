from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from agilerl.modules import EvolvableCNN
from agilerl.modules.base import EvolvableModule, MutationType, mutation
from agilerl.typing import DeviceType, ObservationType
from agilerl.utils.evolvable_networks import create_resnet, get_activation


class EvolvableResNet(EvolvableModule):
    """Evolvable module that implements the architecture presented in 'Deep Residual Learning
    for Image Recognition'. Designed to train CNN's more successfully by introducing residual
    connections that skip one or more layers.

    Paper: https://arxiv.org/abs/1512.03385

    :param input_shape: Input shape of the neural network
    :type input_shape: List[int]
    :param num_outputs: Output layer dimension
    :type num_outputs: int
    :param channel_size: A list of integers representing the number of channels in each convolutional layer
    :type channel_size: int
    :param kernel_size: A list of integers representing the kernel size of each convolutional layer
    :type kernel_size: int
    :param stride_size: A list of integers representing the stride size of each convolutional layer
    :type stride_size: int
    :param num_blocks: Number of residual blocks that compose the network
    :type num_blocks: int
    :param output_activation: Output activation layer, defaults to None
    :type output_activation: str, optional
    :param scale_factor: Scale factor for the network, defaults to 4
    :type scale_factor: int
    :param min_blocks: Minimum number of residual blocks that compose the network, defaults to 1
    :type min_blocks: int
    :param max_blocks: Maximum number of residual blocks that compose the network, defaults to 4
    :type max_blocks: int
    :param min_channel_size: Minimum number of channels in each convolutional layer, defaults to 32
    :type min_channel_size: int
    :param max_channel_size: Maximum number of channels in each convolutional layer, defaults to 256
    :type max_channel_size: int
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param name: Name of the network, defaults to 'resnet'
    :type name: str, optional
    """

    def __init__(
        self,
        input_shape: List[int],
        num_outputs: int,
        channel_size: int,
        kernel_size: int,
        stride_size: int,
        num_blocks: int,
        output_activation: str = None,
        scale_factor: int = 4,
        min_blocks: int = 1,
        max_blocks: int = 4,
        min_channel_size: int = 32,
        max_channel_size: int = 256,
        device: DeviceType = "cpu",
        name: str = "resnet",
    ) -> None:
        super().__init__(device=device)

        assert isinstance(scale_factor, int), "Scale factor must be an integer."
        assert isinstance(num_blocks, int), "Number of blocks must be an integer."
        assert isinstance(channel_size, int), "Channel size must be an integer."
        assert num_outputs >= 1, "Number of blocks must be greater than or equal to 1."
        assert num_blocks >= 1, "Number of blocks must be greater than or equal to 1."

        self.input_shape = input_shape
        self.num_outputs = num_outputs
        self.channel_size = channel_size
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.num_blocks = num_blocks
        self.output_activation = output_activation
        self.scale_factor = scale_factor
        self.min_blocks = min_blocks
        self.max_blocks = max_blocks
        self.min_channel_size = min_channel_size
        self.max_channel_size = max_channel_size
        self.name = name

        self.model = self.create_resnet(
            input_shape=self.input_shape,
            channel_size=channel_size,
            kernel_size=kernel_size,
            stride_size=stride_size,
            num_blocks=num_blocks,
            scale_factor=self.scale_factor,
        )

    @property
    def net_config(self) -> Dict[str, Any]:
        """Returns model configuration in dictionary."""
        net_config = self.init_dict.copy()
        for attr in ["num_inputs", "num_outputs", "device", "name"]:
            if attr in net_config:
                net_config.pop(attr)

        return net_config

    def change_activation(self, activation: str, output: bool = False) -> None:
        """We currently do not support changing the activation function of the ResNet.

        :param activation: Activation function to use.
        :type activation: str
        :param output: Flag indicating whether to set the output activation function, defaults to False
        :type output: bool, optional
        """
        return

    def create_resnet(
        self,
        input_shape: List[int],
        channel_size: int,
        kernel_size: int,
        stride_size: int,
        num_blocks: int,
        scale_factor: int,
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
        :param num_blocks: The number of residual blocks in the CNN.
        :type num_blocks: int
        :param scale_factor: The scale factor for the CNN.
        :type scale_factor: int

        :return: The created convolutional neural network.
        :rtype: nn.Sequential
        """
        # Build the main convolutional block
        net_dict = create_resnet(
            input_channels=input_shape[0],
            channel_size=channel_size,
            kernel_size=kernel_size,
            stride_size=stride_size,
            num_blocks=num_blocks,
            scale_factor=scale_factor,
            device=self.device,
            name=self.name,
        )

        # Flatten image encodings and pass through a final linear layer
        sample_input = torch.zeros(1, *input_shape, device=self.device)
        pre_flatten_output = nn.Sequential(net_dict)(sample_input)
        net_dict[f"{self.name}_flatten"] = nn.Flatten()
        with torch.no_grad():
            cnn_output = nn.Sequential(net_dict)(sample_input)
            flattened_size = cnn_output.shape[1]

        net_dict[f"{self.name}_linear_output"] = nn.Linear(
            flattened_size, self.num_outputs, device=self.device
        )
        net_dict[f"{self.name}_output_activation"] = get_activation(
            self.output_activation
        )

        self.cnn_output_size = pre_flatten_output.shape

        return nn.Sequential(net_dict)

    def forward(self, x: ObservationType) -> torch.Tensor:
        """Returns output of neural network.

        :param x: Neural network input
        :type x: torch.Tensor
        :return: Neural network output
        :rtype: torch.Tensor
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        return self.model(x)

    @mutation(MutationType.LAYER)
    def add_block(self) -> None:
        """Adds a hidden layer to neural network. Falls back on ``add_channel()`` if
        max hidden layers reached."""
        # add layer to hyper params
        if self.num_blocks < self.max_blocks:  # HARD LIMIT
            self.num_blocks += 1
        else:
            return self.add_channel()

    @mutation(MutationType.LAYER, shrink_params=True)
    def remove_block(self) -> None:
        """Removes a hidden layer from neural network. Falls back on ``add_channel()`` if
        min hidden layers reached."""
        if self.num_blocks > self.min_blocks:  # HARD LIMIT
            self.num_blocks -= 1
        else:
            return self.add_channel()

    @mutation(MutationType.NODE)
    def add_channel(
        self,
        numb_new_channels: Optional[int] = None,
    ) -> Dict[str, int]:
        """Remove channel from hidden layer of convolutional neural network.

        :param numb_new_channels: Number of channels to add to hidden layer, defaults to None
        :type numb_new_channels: int, optional
        :return: Dictionary containing the hidden layer and number of new channels
        :rtype: Dict[str, Union[int, None]]
        """
        if numb_new_channels is None:
            numb_new_channels = np.random.choice([8, 16, 32], 1)[0]

        # HARD LIMIT
        if self.channel_size + numb_new_channels < self.max_channel_size:
            self.channel_size += numb_new_channels

        return {"numb_new_channels": numb_new_channels}

    @mutation(MutationType.NODE, shrink_params=True)
    def remove_channel(
        self,
        numb_new_channels: Optional[int] = None,
    ) -> Dict[str, int]:
        """Remove channel from hidden layer of convolutional neural network.

        :param numb_new_channels: Number of channels to add to hidden layer, defaults to None
        :type numb_new_channels: int, optional
        :return: Dictionary containing the hidden layer and number of new channels
        :rtype: Dict[str, Union[int, None]]
        """
        if numb_new_channels is None:
            numb_new_channels = np.random.choice([8, 16, 32], 1)[0]

        # HARD LIMIT
        if self.channel_size - numb_new_channels > self.min_channel_size:
            self.channel_size -= numb_new_channels

        return {"numb_new_channels": numb_new_channels}

    def recreate_network(self, shrink_params: bool = False) -> None:
        """Recreates neural networks.

        :param shrink_params: Flag indicating whether to shrink the parameters, defaults to False
        :type shrink_params: bool, optional
        """
        # Create model with new architecture
        model = self.create_resnet(
            input_shape=self.input_shape,
            channel_size=self.channel_size,
            kernel_size=self.kernel_size,
            stride_size=self.stride_size,
            num_blocks=self.num_blocks,
            scale_factor=self.scale_factor,
        )

        # Copy parameters from old model to new model
        preserve_params_fn = (
            EvolvableCNN.shrink_preserve_parameters
            if shrink_params
            else EvolvableModule.preserve_parameters
        )
        self.model = preserve_params_fn(old_net=self.model, new_net=model)
