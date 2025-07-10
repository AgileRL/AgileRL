from typing import Any, Dict, Optional

import torch

from agilerl.modules.base import EvolvableModule, MutationType, mutation
from agilerl.typing import ObservationType
from agilerl.utils.evolvable_networks import create_simba


class EvolvableSimBa(EvolvableModule):
    """Evolvable module that implements the architecture presented in 'SimBa: Simplicity
    Bias for Scaling Up Parameters in Deep Reinforcement Learning'. Designed to avoid
    overfitting by integrating components that induce a simplicity bias, guiding models toward
    simple and generalizable solutions. Supports the following types of architecture mutations during training:

    * Adding or removing residual blocks
    * Adding or removing nodes from residual blocks
    * Changing the activation function between layers
    * Changing the activation function for the output layer

    Paper: https://arxiv.org/abs/2410.09754

    :param num_inputs: Input layer dimension
    :type num_inputs: int
    :param num_outputs: Output layer dimension
    :type num_outputs: int
    :param hidden_size: Hidden layer(s) size
    :type hidden_size: int
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
    :param name: Name of the network, defaults to 'mlp'
    :type name: str, optional
    :param random_seed: Random seed to use for the network. Defaults to None.
    :type random_seed: Optional[int]
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        hidden_size: int,
        num_blocks: int,
        output_activation: str = None,
        scale_factor: int = 4,
        min_blocks: int = 1,
        max_blocks: int = 4,
        min_mlp_nodes: int = 16,
        max_mlp_nodes: int = 500,
        device: str = "cpu",
        name: str = "simba",
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(device, random_seed)

        assert isinstance(scale_factor, int), "Scale factor must be an integer."

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.output_activation = output_activation
        self.scale_factor = scale_factor
        self.min_blocks = min_blocks
        self.max_blocks = max_blocks
        self.min_mlp_nodes = min_mlp_nodes
        self.max_mlp_nodes = max_mlp_nodes
        self.name = name

        self.model = create_simba(
            input_size=num_inputs,
            output_size=num_outputs,
            hidden_size=hidden_size,
            num_blocks=num_blocks,
            output_activation=output_activation,
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

    def forward(self, x: ObservationType) -> torch.Tensor:
        """Returns output of neural network.

        :param x: Neural network input
        :type x: torch.Tensor
        :return: Neural network output
        :rtype: torch.Tensor
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        return self.model(x)

    def get_output_dense(self) -> torch.nn.Module:
        """Returns output layer of neural network."""
        return getattr(self.model, f"{self.name}_linear_layer_output")

    def init_weights_gaussian(
        self, std_coeff: float = 4, output_coeff: float = 4
    ) -> None:
        """Initialise weights of neural network using Gaussian distribution."""
        EvolvableModule.init_weights_gaussian(self.model, std_coeff=std_coeff)

        # Output layer is initialised with std_coeff=2
        output_layer = self.get_output_dense()
        EvolvableModule.init_weights_gaussian(output_layer, std_coeff=output_coeff)

    @mutation(MutationType.ACTIVATION)
    def change_activation(self, activation: str, output: bool = False) -> None:
        """The SimBa architecture uses ReLU activations by default and this
        shouldn't be changed during training.

        :return: Activation function
        :rtype: str
        """
        return

    @mutation(MutationType.LAYER)
    def add_block(self) -> None:
        """Adds a hidden layer to neural network. Falls back on add_node if
        max hidden layers reached."""
        # add layer to hyper params
        if self.num_blocks < self.max_blocks:  # HARD LIMIT
            self.num_blocks += 1
        else:
            return self.add_node()

    @mutation(MutationType.LAYER)
    def remove_block(self) -> None:
        """Removes a hidden layer from neural network. Falls back on remove_node if
        min hidden layers reached."""
        if self.num_blocks > self.min_blocks:  # HARD LIMIT
            self.num_blocks -= 1
        else:
            return self.add_node()

    @mutation(MutationType.NODE)
    def add_node(self, numb_new_nodes: Optional[int] = None) -> Dict[str, int]:
        """Adds nodes to residual blocks of the neural network.

        :param numb_new_nodes: Number of nodes to add, defaults to None
        :type numb_new_nodes: int, optional
        """
        if numb_new_nodes is None:
            numb_new_nodes = self.rng.choice([16, 32, 64])

        if self.hidden_size + numb_new_nodes <= self.max_mlp_nodes:  # HARD LIMIT
            self.hidden_size += numb_new_nodes

        return {"numb_new_nodes": numb_new_nodes}

    @mutation(MutationType.NODE)
    def remove_node(self, numb_new_nodes: Optional[int] = None) -> Dict[str, int]:
        """Removes nodes from hidden layer of neural network.

        :param hidden_layer: Depth of hidden layer to remove nodes from, defaults to None
        :type hidden_layer: int, optional
        :param numb_new_nodes: Number of nodes to remove from hidden layer, defaults to None
        :type numb_new_nodes: int, optional
        """
        if numb_new_nodes is None:
            numb_new_nodes = self.rng.choice([16, 32, 64])

        # HARD LIMIT
        if self.hidden_size - numb_new_nodes > self.min_mlp_nodes:
            self.hidden_size -= numb_new_nodes

        return {"numb_new_nodes": numb_new_nodes}

    def recreate_network(self) -> None:
        """Recreates neural networks.

        :param shrink_params: Shrink parameters of neural networks, defaults to False
        :type shrink_params: bool, optional
        """
        model = create_simba(
            input_size=self.num_inputs,
            output_size=self.num_outputs,
            hidden_size=self.hidden_size,
            num_blocks=self.num_blocks,
            output_activation=self.output_activation,
            scale_factor=self.scale_factor,
            device=self.device,
            name=self.name,
        )

        self.model = EvolvableModule.preserve_parameters(
            old_net=self.model, new_net=model
        )
