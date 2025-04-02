from typing import Any, Dict, List, Optional

import numpy as np
import torch

from agilerl.modules.base import EvolvableModule, MutationType, mutation
from agilerl.typing import ArrayOrTensor
from agilerl.utils.evolvable_networks import create_mlp


class EvolvableMLP(EvolvableModule):
    """The Evolvable Multi-layer Perceptron class.

    :param num_inputs: Input layer dimension
    :type num_inputs: int
    :param num_outputs: Output layer dimension
    :type num_outputs: int
    :param hidden_size: Hidden layer(s) size
    :type hidden_size: List[int]
    :param activation: Activation layer, defaults to 'ReLU'
    :type activation: str, optional
    :param output_activation: Output activation layer, defaults to None
    :type output_activation: str, optional
    :param min_hidden_layers: Minimum number of hidden layers the network will shrink down to, defaults to 1
    :type min_hidden_layers: int, optional
    :param max_hidden_layers: Maximum number of hidden layers the network will expand to, defaults to 3
    :type max_hidden_layers: int, optional
    :param min_mlp_nodes: Minimum number of nodes a layer can have within the network, defaults to 64
    :type min_mlp_nodes: int, optional
    :param max_mlp_nodes: Maximum number of nodes a layer can have within the network, defaults to 500
    :type max_mlp_nodes: int, optional
    :param layer_norm: Normalization between layers, defaults to True
    :type layer_norm: bool, optional
    :param output_layernorm: Normalization for the output layer, defaults to False
    :type output_layernorm: bool, optional
    :param output_vanish: Vanish output by multiplying by 0.1, defaults to True
    :type output_vanish: bool, optional
    :param init_layers: Initialise network layers, defaults to True
    :type init_layers: bool, optional
    :param noise_std: Noise standard deviation, defaults to 0.5
    :type noise_std: float, optional
    :param noisy: Add noise to network, defaults to False
    :type noisy: bool, optional
    :param new_gelu: Use new GELU activation function, defaults to False
    :type new_gelu: bool, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param name: Name of the network, defaults to 'mlp'
    :type name: str, optional
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        hidden_size: List[int],
        activation: str = "ReLU",
        output_activation: str = None,
        min_hidden_layers: int = 1,
        max_hidden_layers: int = 3,
        min_mlp_nodes: int = 64,
        max_mlp_nodes: int = 500,
        layer_norm: bool = True,
        output_layernorm: bool = False,
        output_vanish: bool = True,
        init_layers: bool = True,
        noisy: bool = False,
        noise_std: float = 0.5,
        new_gelu: bool = False,
        device: str = "cpu",
        name: str = "mlp",
    ):
        super().__init__(device)

        assert (
            num_inputs > 0
        ), "'num_inputs' cannot be less than or equal to zero, please enter a valid integer."
        assert (
            num_outputs > 0
        ), "'num_outputs' cannot be less than or equal to zero, please enter a valid integer."
        for num in hidden_size:
            assert (
                num > 0
            ), "'hidden_size' cannot contain zero, please enter a valid integer."
        assert len(hidden_size) != 0, "MLP must contain at least one hidden layer."
        assert (
            min_hidden_layers < max_hidden_layers
        ), "'min_hidden_layers' must be less than 'max_hidden_layers."
        assert (
            min_mlp_nodes < max_mlp_nodes
        ), "'min_mlp_nodes' must be less than 'max_mlp_nodes."

        self.name = name
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self._activation = activation
        self.new_gelu = new_gelu
        self.output_activation = output_activation
        self.min_hidden_layers = min_hidden_layers
        self.max_hidden_layers = max_hidden_layers
        self.min_mlp_nodes = min_mlp_nodes
        self.max_mlp_nodes = max_mlp_nodes
        self.layer_norm = layer_norm
        self.output_vanish = output_vanish
        self.output_layernorm = output_layernorm
        self.init_layers = init_layers
        self.hidden_size = hidden_size
        self.noisy = noisy
        self.noise_std = noise_std

        self.model = create_mlp(
            input_size=self.num_inputs,
            output_size=self.num_outputs,
            hidden_size=self.hidden_size,
            output_vanish=self.output_vanish,
            output_activation=self.output_activation,
            noisy=self.noisy,
            init_layers=self.init_layers,
            layer_norm=self.layer_norm,
            output_layernorm=self.output_layernorm,
            activation=self.activation,
            noise_std=self.noise_std,
            device=self.device,
            new_gelu=self.new_gelu,
            name=self.name,
        )

    @property
    def net_config(self) -> Dict[str, Any]:
        """Returns model configuration in dictionary."""
        net_config = self.init_dict.copy()
        for attr in ["num_inputs", "num_outputs", "device", "name"]:
            if attr in net_config:
                net_config.pop(attr)

        return net_config

    @property
    def activation(self) -> str:
        """Returns activation function."""
        return self._activation

    @activation.setter
    def activation(self, activation: str) -> None:
        """Set activation function."""
        self._activation = activation

    def init_weights_gaussian(
        self, std_coeff: float = 4, output_coeff: float = 4
    ) -> None:
        """Initialise weights of neural network using Gaussian distribution."""
        EvolvableModule.init_weights_gaussian(self.model, std_coeff=std_coeff)

        # Output layer is initialised with std_coeff=2
        output_layer = self.get_output_dense()
        EvolvableModule.init_weights_gaussian(output_layer, std_coeff=output_coeff)

    def forward(self, x: ArrayOrTensor) -> torch.Tensor:
        """Returns output of neural network.

        :param x: Neural network input
        :type x: torch.Tensor or np.ndarray

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

    @mutation(MutationType.LAYER)
    def add_layer(self) -> None:
        """Adds a hidden layer to neural network. Falls back on ``add_node()`` if
        max hidden layers reached."""
        # add layer to hyper params
        if len(self.hidden_size) < self.max_hidden_layers:  # HARD LIMIT
            self.hidden_size += [self.hidden_size[-1]]
        else:
            return self.add_node()

    @mutation(MutationType.LAYER)
    def remove_layer(self) -> None:
        """Removes a hidden layer from neural network. Falls back on ``add_node()`` if
        min hidden layers reached."""
        if len(self.hidden_size) > self.min_hidden_layers:  # HARD LIMIT
            self.hidden_size = self.hidden_size[:-1]
        else:
            return self.add_node()

    @mutation(MutationType.NODE)
    def add_node(
        self, hidden_layer: Optional[int] = None, numb_new_nodes: Optional[int] = None
    ) -> Dict[str, int]:
        """Adds nodes to hidden layer of neural network.

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
    def remove_node(
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

        # HARD LIMIT
        if self.hidden_size[hidden_layer] - numb_new_nodes > self.min_mlp_nodes:
            self.hidden_size[hidden_layer] -= numb_new_nodes

        return {"hidden_layer": hidden_layer, "numb_new_nodes": numb_new_nodes}

    def recreate_network(self) -> None:
        """Recreates neural networks.

        :param shrink_params: Shrink parameters of neural networks, defaults to False
        :type shrink_params: bool, optional
        """
        model = create_mlp(
            input_size=self.num_inputs,
            output_size=self.num_outputs,
            hidden_size=self.hidden_size,
            output_vanish=self.output_vanish,
            output_activation=self.output_activation,
            noisy=self.noisy,
            init_layers=self.init_layers,
            layer_norm=self.layer_norm,
            output_layernorm=self.output_layernorm,
            activation=self.activation,
            noise_std=self.noise_std,
            new_gelu=self.new_gelu,
            device=self.device,
            name=self.name,
        )

        self.model = EvolvableModule.preserve_parameters(
            old_net=self.model, new_net=model
        )
