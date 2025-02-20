from typing import Any, Dict, List

import torch
import torch.nn as nn

from agilerl.modules.base import EvolvableModule
from agilerl.typing import ObservationType
from agilerl.utils.evolvable_networks import create_resnet, get_activation


class EvolvableResNet(EvolvableModule):
    """Evolvable module that implements the architecture presented in 'Deep Residual Learning
    for Image Recognition'. Designed to train very deep neural networks with tens of layers
    successfully by introducing residual connections that skip one or more layers.

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
    :type scale_factor: int, optional
    :param min_blocks: Minimum number of residual blocks that compose the network, defaults to 1
    :type min_blocks: int, optional
    :param max_blocks: Maximum number of residual blocks that compose the network, defaults to 4
    :type max_blocks: int, optional
    :param min_mlp_nodes: Minimum number of nodes a layer can have within the network, defaults to 16
    :type min_mlp_nodes: int, optional
    :param max_mlp_nodes: Maximum number of nodes a layer can have within the network, defaults to 500
    :type max_mlp_nodes: int, optional
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
        device: str = "cpu",
        name: str = "simba",
    ) -> None:
        super().__init__(device=device)

        assert isinstance(scale_factor, int), "Scale factor must be an integer."

        self.input_shape = input_shape
        self.num_outputs = num_outputs
        self.channel_size = channel_size
        self.num_blocks = num_blocks
        self.output_activation = output_activation
        self.scale_factor = scale_factor
        self.min_blocks = min_blocks
        self.max_blocks = max_blocks
        self.min_channel_size = min_channel_size
        self.max_channel_size = max_channel_size
        self.name = name

        self.model = self.create_resnet(
            input_channels=self.input_shape[0],
            channel_size=channel_size,
            kernel_size=kernel_size,
            stride_size=stride_size,
            num_blocks=num_blocks,
            scale_factor=self.scale_factor,
            device=device,
            name=name,
        )

    @property
    def net_config(self) -> Dict[str, Any]:
        """Returns model configuration in dictionary."""
        net_config = self.init_dict.copy()
        for attr in ["num_inputs", "num_outputs", "device", "name"]:
            if attr in net_config:
                net_config.pop(attr)

        return net_config

    def create_resnet(
        self,
        input_shape: List[int],
        channel_size: int,
        kernel_size: int,
        stride_size: int,
        num_blocks: int,
        scale_factor: int,
        sample_input: torch.Tensor,
        name: str,
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
        :param sample_input: A sample input tensor.
        :type sample_input: torch.Tensor
        :param name: The name of the CNN.
        :type name: str

        :return: The created convolutional neural network.
        :rtype: nn.Sequential
        """
        # Build the main convolutional block
        net_dict = create_resnet(
            in_channels=input_shape[0],
            channel_size=channel_size,
            kernel_size=kernel_size,
            stride_size=stride_size,
            num_blocks=num_blocks,
            scale_factor=scale_factor,
            name=name,
            device=self.device,
        )

        # Flatten image encodings and pass through a final linear layer
        sample_input = torch.zeros(1, *input_shape, device=self.device)
        pre_flatten_output = nn.Sequential(net_dict)(sample_input)
        net_dict[f"{name}_flatten"] = nn.Flatten()
        with torch.no_grad():
            cnn_output = nn.Sequential(net_dict)(sample_input)
            flattened_size = cnn_output.shape[1]

        net_dict[f"{name}_linear_output"] = nn.Linear(
            flattened_size, self.num_outputs, device=self.device
        )
        net_dict[f"{name}_output_activation"] = get_activation(self.output_activation)

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
