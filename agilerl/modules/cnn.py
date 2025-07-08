from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from agilerl.modules.base import EvolvableModule, MutationType, mutation
from agilerl.typing import ArrayOrTensor, KernelSizeType
from agilerl.utils.evolvable_networks import create_cnn, get_activation

BlockType = Literal["Conv1d", "Conv2d", "Conv3d"]


def _assert_correct_kernel_sizes(
    sizes: List[Tuple[int, ...]], block_type: BlockType
) -> None:
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
        elif len(k_size) == 1 and block_type == "Conv1d":
            # For Conv1D, kernel_size is (length,), so no width/height to compare.
            # The assertion len(set(k_size)) == 1 will hold.
            wh_unique = set(k_size)
        else:
            msg = (
                "1 (length) for Conv1d, "
                "2 (width x height) for Conv2d, or "
                "3 (depth x width x height) for Conv3d"
            )
            raise ValueError(
                f"Found CNN kernel with length {len(k_size)}. These should "
                f"have a length of {msg} for {block_type}."
            )

        assert len(wh_unique) == 1, (
            "AgileRL currently doesn't support having different "
            f"values for width and height in a single CNN kernel: {wh_unique}"
        )


@dataclass
class MutableKernelSizes:
    sizes: List[KernelSizeType]
    cnn_block_type: Literal["Conv2d", "Conv3d"]
    sample_input: Optional[torch.Tensor]
    rng: np.random.Generator

    def __post_init__(self) -> None:
        tuple_sizes = False
        if isinstance(self.sizes[0], (tuple, list)):
            tuple_sizes = True
            _assert_correct_kernel_sizes(self.sizes, self.cnn_block_type)
        elif self.cnn_block_type == "Conv3d":
            # NOTE: If kernel sizes are passed as integers in multi-agent settings, we
            # add a depth dimension of 1 for all layers. Note that for e.g. value functions
            # or Q networks it is common for the first layer to have a depth corresponding
            # to the number of agents (since we stack the observations from all agents to
            # obtain the value)
            sizes = [(1, k_size, k_size) for k_size in self.sizes]

            # We infer the depth of the first kernel from the shape of the sample input tensor
            sizes[0] = (self.sample_input.size(2), self.sizes[0], self.sizes[0])
            self.sizes = sizes
            tuple_sizes = True
        elif self.cnn_block_type == "Conv1d":
            # If kernel sizes are passed as integers for Conv1d, convert to (length,) tuples
            self.sizes = [(k_size,) for k_size in self.sizes]
            tuple_sizes = True

        self.tuple_sizes = tuple_sizes

    @property
    def int_sizes(self) -> List[int]:
        if self.tuple_sizes:
            return [k_size[-1] for k_size in self.sizes]
        return self.sizes

    def __len__(self) -> int:
        return len(self.sizes)

    def add_layer(self, other: int) -> None:
        if self.tuple_sizes:
            if self.cnn_block_type == "Conv2d":
                other = (other, other)
            elif self.cnn_block_type == "Conv3d":
                # NOTE: Do we always want to use a depth of one when adding a layer
                # in multi-agent settings?
                other = (1, other, other)
            elif self.cnn_block_type == "Conv1d":
                other = (other,)

        self.sizes += [other]

    def remove_layer(self) -> None:
        self.sizes = self.sizes[:-1]

    def calc_max_kernel_sizes(
        self, channel_size: List[int], stride_size: List[int], input_shape: List[int]
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

    def change_kernel_size(
        self,
        hidden_layer: int,
        channel_size: List[int],
        stride_size: List[int],
        input_shape: Tuple[int],
        kernel_size: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> int:
        """Randomly alters convolution kernel of random CNN layer.

        :param hidden_layer: Depth of hidden layer to change kernel size of.
        :type hidden_layer: int
        :param channel_size: Number of channels in each convolutional layer.
        :type channel_size: List[int]
        :param stride_size: Stride size of each convolutional layer.
        :type stride_size: List[int]
        :param input_shape: Input shape.
        :type input_shape: Tuple[int]
        :param kernel_size: Kernel size to change to, defaults to None
        :type kernel_size: int, optional

        :return: New kernel size
        :rtype: int
        """
        if kernel_size is not None:
            if self.tuple_sizes:
                assert isinstance(kernel_size, tuple), "Kernel size must be a tuple."
            else:
                assert isinstance(kernel_size, int), "Kernel size must be an integer."

            new_kernel_size = kernel_size
        else:
            max_kernels = self.calc_max_kernel_sizes(
                channel_size, stride_size, input_shape
            )

            # Get current kernel size to avoid generating the same value
            current_kernel_size = self.sizes[hidden_layer]
            if self.tuple_sizes:
                current_kernel_size = current_kernel_size[-1]

            max_kernel = max_kernels[hidden_layer]
            if max_kernel == 1:
                new_kernel_size = 1
            else:
                candidates = [
                    k for k in range(1, max_kernel + 1) if k != current_kernel_size
                ]
                if candidates:
                    new_kernel_size = self.rng.choice(candidates)
                else:
                    new_kernel_size = self.rng.integers(1, max_kernel + 1)

        if self.tuple_sizes:
            if self.cnn_block_type == "Conv2d":
                self.sizes[hidden_layer] = (new_kernel_size, new_kernel_size)
            elif self.cnn_block_type == "Conv3d":
                depth = self.sizes[hidden_layer][0]
                self.sizes[hidden_layer] = (depth, new_kernel_size, new_kernel_size)
            elif self.cnn_block_type == "Conv1d":
                self.sizes[hidden_layer] = (new_kernel_size,)
        else:
            self.sizes[hidden_layer] = new_kernel_size

        return new_kernel_size


class EvolvableCNN(EvolvableModule):
    """The Evolvable Convolutional Neural Network class. Consists of a sequence of convolutional layers
    with an optional activation function between each layer. Supports using layer normalization. Allows for
    the following types of architecture mutations during training:

    * Adding or removing convolutional layers
    * Adding or removing channels from convolutional layers
    * Changing the kernel size and stride size of convolutional layers
    * Changing the activation function between layers (e.g. ReLU to GELU)
    * Changing the activation function for the output layer (e.g. ReLU to GELU)

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
    :param block_type: Type of convolutional block, either 'Conv1d', 'Conv2d' or 'Conv3d', defaults to 'Conv2d'.
    :type block_type: Literal["Conv1d", "Conv2d", "Conv3d"], optional
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
    :param random_seed: Random seed to use for the network. Defaults to None.
    :type random_seed: Optional[int]
    """

    def __init__(
        self,
        input_shape: List[int],
        num_outputs: int,
        channel_size: List[int],
        kernel_size: List[KernelSizeType],
        stride_size: List[int],
        sample_input: Optional[torch.Tensor] = None,
        block_type: Literal["Conv1d", "Conv2d", "Conv3d"] = "Conv2d",
        activation: str = "ReLU",
        output_activation: Optional[str] = None,
        min_hidden_layers: int = 1,
        max_hidden_layers: int = 6,
        min_channel_size: int = 16,
        max_channel_size: int = 256,
        layer_norm: bool = False,
        init_layers: bool = True,
        device: str = "cpu",
        name: str = "cnn",
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(device, random_seed)

        assert len(kernel_size) == len(
            channel_size
        ), "Length of kernel size list must be the same length as channel size list."
        assert len(stride_size) == len(
            channel_size
        ), "Length of stride size list must be the same length as channel size list."
        # assert len(input_shape) >= 3, "Input shape must have at least 3 dimensions." # Adjusted below
        assert (
            num_outputs > 0
        ), "'num_outputs' cannot be less than or equal to zero, please enter a valid integer."
        assert (
            min_hidden_layers < max_hidden_layers
        ), "'min_hidden_layers' must be less than 'max_hidden_layers."
        assert (
            min_channel_size < max_channel_size
        ), "'min_channel_size' must be less than 'max_channel_size'."

        if block_type == "Conv1d":
            assert (
                len(input_shape) == 2
            ), f"For Conv1d, input_shape should be (channels, length), got {input_shape}"
            _sample_input = (
                torch.zeros(1, *input_shape, device=device)  # (1, C, L)
                if sample_input is None
                else sample_input
            )
            assert (
                len(_sample_input.shape) == 3
            ), f"Sample input for Conv1d must be (B, C, L), got shape {_sample_input.shape}"
        elif block_type == "Conv2d":
            assert (
                len(input_shape) == 3
            ), f"For Conv2d, input_shape should be (channels, height, width), got {input_shape}"
            _sample_input = (
                torch.zeros(1, *input_shape, device=device)
                if sample_input is None
                else sample_input
            )
            assert (
                len(_sample_input.shape) == 4
            ), f"Sample input for Conv2d must be (B, C, H, W), got shape {_sample_input.shape}"
        elif block_type == "Conv3d":
            assert (
                len(input_shape) == 3
            ), f"For Conv3d, input_shape should be (channels, height, width), got {input_shape}"
            assert (
                sample_input is not None and len(sample_input.shape) == 5
            ), f"Sample input with shape format (B, C, D, H, W) must be provided for 3D convolutional networks, got {sample_input.shape if sample_input is not None else None}."
            _sample_input = sample_input
        else:
            raise ValueError(
                f"Invalid block type: {block_type}. Must be 'Conv1d', 'Conv2d' or 'Conv3d'."
            )

        self.input_shape = input_shape
        self.channel_size = channel_size
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
        self.sample_input = _sample_input.to(device)
        self.name = name
        self.mut_kernel_size = MutableKernelSizes(
            sizes=kernel_size,
            cnn_block_type=block_type,
            sample_input=self.sample_input,
            rng=self.rng,
        )

        self.model = self.create_cnn(
            in_channels=input_shape[0],
            channel_size=channel_size,
            kernel_size=self.mut_kernel_size.sizes,
            stride_size=stride_size,
            sample_input=self.sample_input,
        )

    @property
    def net_config(self) -> Dict[str, Any]:
        net_config = self.init_dict.copy()
        for attr in ["input_shape", "num_outputs", "device", "name"]:
            net_config.pop(attr, None)

        return net_config

    @property
    def kernel_size(self) -> List[KernelSizeType]:
        """Returns the kernel size of the network.

        :return: Kernel size
        :rtype: List[KernelSizeType]
        """
        return self.mut_kernel_size.int_sizes

    @property
    def activation(self) -> str:
        """Returns the activation function of the network.

        :return: Activation function
        :rtype: str
        """
        return self._activation

    @activation.setter
    def activation(self, activation: str) -> None:
        """Sets the activation function of the network.

        :param activation: Activation function to use.
        :type activation: str
        """
        self._activation = activation

    @staticmethod
    def shrink_preserve_parameters(old_net: nn.Module, new_net: nn.Module) -> nn.Module:
        """Returns shrunk new neural network with copied parameters from old network.

        :param old_net: Old neural network
        :type old_net: nn.Module
        :param new_net: New neural network
        :type new_net: nn.Module
        :return: Shrunk new neural network with copied parameters
        :rtype: nn.Module
        """
        old_net_dict = dict(old_net.named_parameters())

        for key, param in new_net.named_parameters():
            if key in old_net_dict.keys():
                old_param = old_net_dict[key]
                old_size = old_param.data.size()
                new_size = param.data.size()

                if old_size == new_size:
                    param.data = old_param.data
                elif "norm" not in key:
                    min_0 = min(old_size[0], new_size[0])
                    if len(param.data.size()) == 1:
                        param.data[:min_0] = old_param.data[:min_0]

                    # NOTE: We specifically implement this method to only maintain spatial
                    # information in convolutional layers when reducing kernel / channel
                    # sizes within a layer.
                    else:
                        min_1 = min(old_size[1], new_size[1])
                        param.data[:min_0, :min_1] = old_net_dict[key].data[
                            :min_0, :min_1
                        ]

        return new_net

    def init_weights_gaussian(self, std_coeff: float = 4) -> None:
        """Initialise weights of linear layer using Gaussian distribution.

        :param std_coeff: Standard deviation coefficient, defaults to 4
        :type std_coeff: float, optional
        """
        # Output layer is initialised with std_coeff=2
        output_layer = self.get_output_dense()
        EvolvableModule.init_weights_gaussian(output_layer, std_coeff=std_coeff)

    def get_output_dense(self) -> torch.nn.Module:
        """Returns output layer of neural network.

        :return: Output layer of neural network
        :rtype: torch.nn.Module
        """
        return getattr(self.model, f"{self.name}_linear_output")

    def change_activation(self, activation: str, output: bool = False) -> None:
        """Set the activation function for the network.

        :param activation: Activation function to use.
        :type activation: str
        :param output: Flag indicating whether to set the output activation function, defaults to False
        :type output: bool, optional
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
        :param sample_input: A sample input tensor.
        :type sample_input: torch.Tensor
        :param name: The name of the CNN.
        :type name: str

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
            name=self.name,
            init_layers=self.init_layers,
            layer_norm=self.layer_norm,
            activation_fn=self.activation,
            device=self.device,
        )

        # Flatten image encodings and pass through a final linear layer
        pre_flatten_model = nn.Sequential(net_dict)
        pre_flatten_model.eval()
        with torch.no_grad():
            pre_flatten_output = pre_flatten_model(sample_input)
            net_dict[f"{self.name}_flatten"] = nn.Flatten()
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

    def reset_noise(self) -> None:
        """Resets noise of the model layers."""
        EvolvableModule.reset_noise(self.model)

    def forward(self, x: ArrayOrTensor) -> torch.Tensor:
        """Returns output of neural network.

        :param x: Neural network input
        :type x: torch.Tensor or np.ndarray

        :return: Output of the neural network
        :rtype: torch.Tensor
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        expected_dims = 0
        if self.block_type == "Conv1d":
            expected_dims = 3  # (N, C, L)
        elif self.block_type == "Conv2d":
            expected_dims = 4  # (N, C, H, W)
        elif self.block_type == "Conv3d":
            expected_dims = 5  # (N, C, D, H, W)

            # Specific handling for Conv3d if it expects depth but receives data without it
            # (e.g. (N, C, H, W) -> (N, C, 1, H, W))
            if len(x.shape) == expected_dims - 1:
                x = x.unsqueeze(2)

        if len(x.shape) == expected_dims - 1:  # Missing batch dimension
            x = x.unsqueeze(0)

        return self.model(x)

    @mutation(MutationType.LAYER)
    def add_layer(self) -> None:
        """Adds a hidden layer to convolutional neural network.

        :return: If maximum number of hidden layers is reached, returns a dictionary containing
        the hidden layer and number of new channels.
        :rtype: Optional[Dict[str, int]]
        """
        dims_to_check = (
            self.cnn_output_size[-1:]
            if self.block_type == "Conv1d"
            else self.cnn_output_size[-2:]
        )

        condition_max_k_gt_2 = False
        if self.block_type == "Conv1d":
            # For Conv1D, if the current output length is greater than 2,
            # it implies that meaningful kernels (>2) were possible or are possible.
            if self.cnn_output_size[-1] > 2:
                condition_max_k_gt_2 = True
        else:
            # Original logic for Conv2D/Conv3D
            # This relies on calc_max_kernel_sizes which might need review for 3D if issues arise.
            max_kernels_from_util = self.mut_kernel_size.calc_max_kernel_sizes(
                self.channel_size, self.stride_size, self.input_shape
            )
            if max_kernels_from_util and max_kernels_from_util[-1] > 2:
                condition_max_k_gt_2 = True

        if (
            len(self.channel_size) < self.max_hidden_layers
            and not any(
                i <= 2 for i in dims_to_check
            )  # Current output dim is large enough
            and condition_max_k_gt_2
        ):
            # Try to add a new layer
            self.channel_size += [self.channel_size[-1]]  # Provisional: add channel

            l_in_for_new_layer = self.cnn_output_size[
                -1
            ]  # Output length of current last layer is input to new one

            if (
                l_in_for_new_layer < 2
            ):  # Not enough input dimension for a kernel of min size 2
                self.channel_size = self.channel_size[:-1]  # Revert channel addition
                return self.add_channel()

            # Determine kernel size for the new layer
            # Kernel size k_new: 2 <= k_new <= l_in_for_new_layer

            k_new = self.rng.integers(2, l_in_for_new_layer + 1)
            self.mut_kernel_size.add_layer(k_new)  # Provisional: add kernel

            # Determine stride for the new layer
            # Stride s_new: 1 <= s_new <= (l_in_for_new_layer - k_new + 1)
            max_s_new = l_in_for_new_layer - k_new + 1
            if (
                max_s_new < 1
            ):  # Not possible to make a valid stride (e.g. k_new > l_in_for_new_layer)
                self.channel_size = self.channel_size[:-1]  # Revert channel
                self.mut_kernel_size.remove_layer()  # Revert kernel
                return self.add_channel()

            s_new = self.rng.integers(1, max_s_new + 1)
            self.stride_size = self.stride_size + [s_new]

            # If all successful, the provisional additions are kept.
            # Network will be recreated by the @mutation decorator.
        else:
            # Conditions not met, or provisional addition failed and reverted.
            return self.add_channel()

    @mutation(MutationType.LAYER, shrink_params=True)
    def remove_layer(self) -> Optional[Dict[str, int]]:
        """Removes a hidden layer from convolutional neural network.

        :return: If minimum number of hidden layers is reached, returns a dictionary containing
        the hidden layer and number of new channels.
        :rtype: Optional[Dict[str, int]]
        """
        if len(self.channel_size) > self.min_hidden_layers:
            self.channel_size = self.channel_size[:-1]
            self.mut_kernel_size.remove_layer()
            self.stride_size = self.stride_size[:-1]
        else:
            return self.add_channel()

    @mutation(MutationType.NODE)
    def change_kernel(
        self, kernel_size: Optional[int] = None, hidden_layer: Optional[int] = None
    ) -> Dict[str, Union[int, None]]:
        """Randomly alters convolution kernel of random CNN layer.

        :param kernel_size: Kernel size to change to, defaults to None
        :type kernel_size: int, optional
        :param hidden_layer: Depth of hidden layer to change kernel size of, defaults to None
        :type hidden_layer: int, optional

        :return: Dictionary containing the hidden layer and kernel size
        :rtype: Dict[str, Union[int, None]]
        """
        if len(self.channel_size) > 1:
            if hidden_layer is None:
                hidden_layer = self.rng.integers(1, min(4, len(self.channel_size)))

            new_kernel_size = self.mut_kernel_size.change_kernel_size(
                hidden_layer,
                self.channel_size,
                self.stride_size,
                self.input_shape,
                kernel_size,
            )
        else:
            return self.add_layer()

        return {"hidden_layer": hidden_layer, "kernel_size": new_kernel_size}

    @mutation(MutationType.NODE)
    def add_channel(
        self,
        hidden_layer: Optional[int] = None,
        numb_new_channels: Optional[int] = None,
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
            hidden_layer = self.rng.integers(0, len(self.channel_size))
        else:
            hidden_layer = min(hidden_layer, len(self.channel_size) - 1)

        # Randomly choose number of channels to add
        if numb_new_channels is None:
            numb_new_channels = self.rng.choice([8, 16, 32])

        # HARD LIMIT
        if self.channel_size[hidden_layer] + numb_new_channels <= self.max_channel_size:
            self.channel_size[hidden_layer] += numb_new_channels

        return {"hidden_layer": hidden_layer, "numb_new_channels": numb_new_channels}

    @mutation(MutationType.NODE, shrink_params=True)
    def remove_channel(
        self,
        hidden_layer: Optional[int] = None,
        numb_new_channels: Optional[int] = None,
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
            hidden_layer = self.rng.integers(0, len(self.channel_size))
        else:
            hidden_layer = min(hidden_layer, len(self.channel_size) - 1)

        if numb_new_channels is None:
            numb_new_channels = self.rng.choice([8, 16, 32])

        # HARD LIMIT
        if self.channel_size[hidden_layer] - numb_new_channels >= self.min_channel_size:
            self.channel_size[hidden_layer] -= numb_new_channels
        else:
            numb_new_channels = 0

        return {"hidden_layer": hidden_layer, "numb_new_channels": numb_new_channels}

    def recreate_network(self, shrink_params: bool = False) -> None:
        """Recreates the neural network while preserving the parameters of the old network.

        :param shrink_params: Flag indicating whether to shrink the parameters, defaults to False
        :type shrink_params: bool, optional
        """

        # Create model with new architecture
        model = self.create_cnn(
            in_channels=self.input_shape[0],
            channel_size=self.channel_size,
            kernel_size=self.mut_kernel_size.sizes,
            stride_size=self.stride_size,
            sample_input=self.sample_input,  # Use the stored sample_input
        )

        # Copy parameters from old model to new model
        preserve_params_fn = (
            self.shrink_preserve_parameters
            if shrink_params
            else EvolvableModule.preserve_parameters
        )
        self.model = preserve_params_fn(old_net=self.model, new_net=model)
