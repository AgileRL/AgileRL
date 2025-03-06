from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from agilerl.modules.base import EvolvableModule, MutationType, mutation
from agilerl.typing import ArrayOrTensor
from agilerl.utils.evolvable_networks import create_rnn


class EvolvableRNN(EvolvableModule):
    """
    The Evolvable Recurrent Neural Network (RNN) class for processing data in POMDPs.

    :param input_size: Size of input features
    :type input_size: int
    :param hidden_size: Size of hidden layers
    :type hidden_size: List[int]
    :param num_outputs: Output dimension
    :type num_outputs: int
    :param num_layers: Number of RNN layers
    :type num_layers: int
    :param activation: Activation function for hidden layers, defaults to 'ReLU'
    :type activation: str, optional
    :param output_activation: Output activation function, defaults to None
    :type output_activation: Optional[str], optional
    :param min_hidden_layers: Minimum number of hidden layers, defaults to 1
    :type min_hidden_layers: int, optional
    :param max_hidden_layers: Maximum number of hidden layers, defaults to 3
    :type max_hidden_layers: int, optional
    :param min_nodes: Minimum number of nodes, defaults to 32
    :type min_nodes: int, optional
    :param max_nodes: Maximum number of nodes, defaults to 256
    :type max_nodes: int, optional
    :param bidirectional: Whether to use bidirectional RNN, defaults to False
    :type bidirectional: bool, optional
    :param dropout: Dropout probability, defaults to 0.0
    :type dropout: float, optional
    :param layer_norm: Whether to use layer normalization, defaults to True
    :type layer_norm: bool, optional
    :param output_vanish: Whether to initialize output layer weights to a small value, defaults to True
    :type output_vanish: bool, optional
    :param init_layers: Whether to initialize layers, defaults to True
    :type init_layers: bool, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param name: Name of the network, defaults to 'rnn'
    :type name: str, optional
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: List[int],
        num_outputs: int,
        num_layers: int = 1,
        activation: str = "ReLU",
        output_activation: Optional[str] = None,
        min_hidden_layers: int = 1,
        max_hidden_layers: int = 3,
        min_nodes: int = 32,
        max_nodes: int = 256,
        bidirectional: bool = False,
        dropout: float = 0.0,
        layer_norm: bool = True,
        output_vanish: bool = True,
        init_layers: bool = True,
        device: str = "cpu",
        name: str = "   ",
    ):
        super().__init__(device)

        assert (
            input_size > 0
        ), "'input_size' cannot be less than or equal to zero, please enter a valid integer."
        assert (
            num_outputs > 0
        ), "'num_outputs' cannot be less than or equal to zero, please enter a valid integer."
        assert (
            num_layers > 0
        ), "'num_layers' cannot be less than or equal to zero, please enter a valid integer."
        for size in hidden_size:
            assert (
                size > 0
            ), "'hidden_size' cannot contain zero, please enter a valid integer."
        assert len(hidden_size) != 0, "LSTM must contain at least one hidden layer."
        assert (
            min_hidden_layers < max_hidden_layers
        ), "'min_hidden_layers' must be less than 'max_hidden_layers'."
        assert min_nodes < max_nodes, "'min_nodes' must be less than 'max_nodes'."
        assert 0 <= dropout < 1, "'dropout' must be between 0 and 1."

        self.name = name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self._activation = activation
        self.output_activation = output_activation
        self.min_hidden_layers = min_hidden_layers
        self.max_hidden_layers = max_hidden_layers
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.output_vanish = output_vanish
        self.init_layers = init_layers

        # Create the LSTM network
        self.model = self.create_lstm()

    def create_lstm(self) -> nn.Sequential:
        """Creates and returns LSTM network.

        :return: LSTM network
        :rtype: nn.Sequential
        """
        return create_rnn(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_outputs=self.num_outputs,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            output_activation=self.output_activation,
            output_vanish=self.output_vanish,
            init_layers=self.init_layers,
            layer_norm=self.layer_norm,
            activation=self.activation,
            device=self.device,
            name=self.name,
        )

    @property
    def activation(self) -> str:
        """Returns the activation function.

        :return: Activation function
        :rtype: str
        """
        return self._activation

    @activation.setter
    def activation(self, activation: str) -> None:
        """Sets the activation function.

        :param activation: Activation function
        :type activation: str
        """
        self._activation = activation

    def init_weights_gaussian(self, std_coeff: float = 4) -> None:
        """Initialize weights with Gaussian distribution.

        :param std_coeff: Standard deviation coefficient, defaults to 4
        :type std_coeff: float, optional
        """
        EvolvableModule.init_weights_gaussian(self.model, std_coeff)

    def forward(self, x: ArrayOrTensor) -> torch.Tensor:
        """Forward pass through the network.

        :param x: Input tensor of shape (batch_size, sequence_length, input_size)
        :type x: ArrayOrTensor

        :return: Output tensor
        :rtype: torch.Tensor
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        # Ensure input has the right shape (batch_size, sequence_length, input_size)
        if len(x.shape) == 2:
            # If input is (batch_size, input_size), add sequence dimension
            x = x.unsqueeze(1)

        # Process through LSTM and get the last output
        lstm_output = None
        hidden_states = None
        for name, module in self.model.named_children():
            if "lstm" in name:
                # LSTM layer returns (output, (h_n, c_n))
                lstm_output, hidden_states = module(x)
                # Use the output of the last time step
                x = lstm_output[:, -1, :]
            else:
                # Process through other layers
                x = module(x)

        return x

    def get_output_dense(self) -> torch.nn.Module:
        """Returns the output layer.

        :return: Output layer
        :rtype: torch.nn.Module
        """
        return self.model._modules[f"{self.name}_linear_layer_output"]

    def change_activation(self, activation: str, output: bool = False) -> None:
        """Changes the activation function.

        :param activation: New activation function
        :type activation: str
        :param output: Whether to change the output activation, defaults to False
        :type output: bool, optional
        """
        if output:
            self.output_activation = activation
        else:
            self._activation = activation
        self.recreate_network()

    @mutation(MutationType.LAYER)
    def add_layer(self) -> None:
        """Adds a hidden layer to the network."""
        if len(self.hidden_size) < self.max_hidden_layers:
            # Add a new hidden layer with size between min and max
            new_size = np.random.randint(
                self.min_hidden_layers, self.max_hidden_layers + 1
            )
            self.hidden_size.append(new_size)
            self.recreate_network()

    @mutation(MutationType.LAYER)
    def remove_layer(self) -> None:
        """Removes a hidden layer from the network."""
        if len(self.hidden_size) > self.min_hidden_layers:
            # Remove the last hidden layer
            self.hidden_size.pop()
            self.recreate_network()

    @mutation(MutationType.NODE)
    def add_node(
        self, hidden_layer: Optional[int] = None, numb_new_nodes: Optional[int] = None
    ) -> Dict[str, int]:
        """Adds nodes to a hidden layer.

        :param hidden_layer: Index of the hidden layer to add nodes to, defaults to None (random)
        :type hidden_layer: Optional[int], optional
        :param numb_new_nodes: Number of nodes to add, defaults to None (random)
        :type numb_new_nodes: Optional[int], optional

        :return: Dictionary with hidden layer index and number of nodes added
        :rtype: Dict[str, int]
        """
        if hidden_layer is None:
            # Choose a random hidden layer
            hidden_layer = np.random.randint(0, len(self.hidden_size))

        if numb_new_nodes is None:
            # Choose a random number of nodes to add
            max_new_nodes = self.max_lstm_size - self.hidden_size[hidden_layer]
            if max_new_nodes <= 0:
                return {"hidden_layer": hidden_layer, "numb_new_nodes": 0}
            numb_new_nodes = np.random.randint(1, min(max_new_nodes, 20) + 1)

        # Add nodes to the selected hidden layer
        self.hidden_size[hidden_layer] += numb_new_nodes
        self.recreate_network()

        return {"hidden_layer": hidden_layer, "numb_new_nodes": numb_new_nodes}

    @mutation(MutationType.NODE)
    def remove_node(
        self, hidden_layer: Optional[int] = None, numb_new_nodes: Optional[int] = None
    ) -> Dict[str, int]:
        """Removes nodes from a hidden layer.

        :param hidden_layer: Index of the hidden layer to remove nodes from, defaults to None (random)
        :type hidden_layer: Optional[int], optional
        :param numb_new_nodes: Number of nodes to remove, defaults to None (random)
        :type numb_new_nodes: Optional[int], optional

        :return: Dictionary with hidden layer index and number of nodes removed
        :rtype: Dict[str, int]
        """
        if hidden_layer is None:
            # Choose a random hidden layer
            hidden_layer = np.random.randint(0, len(self.hidden_size))

        if numb_new_nodes is None:
            # Choose a random number of nodes to remove
            max_remove_nodes = self.hidden_size[hidden_layer] - self.min_lstm_size
            if max_remove_nodes <= 0:
                return {"hidden_layer": hidden_layer, "numb_new_nodes": 0}
            numb_new_nodes = np.random.randint(1, min(max_remove_nodes, 20) + 1)

        # Remove nodes from the selected hidden layer
        self.hidden_size[hidden_layer] -= numb_new_nodes
        self.recreate_network()

        return {"hidden_layer": hidden_layer, "numb_new_nodes": numb_new_nodes}

    @mutation(MutationType.NODE)
    def change_lstm_layers(self, num_layers: Optional[int] = None) -> Dict[str, int]:
        """Changes the number of stacked LSTM layers.

        :param num_layers: New number of LSTM layers, defaults to None (random)
        :type num_layers: Optional[int], optional

        :return: Dictionary with the new number of LSTM layers
        :rtype: Dict[str, int]
        """
        if num_layers is None:
            # Choose a random number of layers between 1 and 3
            num_layers = np.random.randint(1, 4)

        self.num_layers = num_layers
        self.recreate_network()

        return {"num_layers": num_layers}

    @mutation(MutationType.NODE)
    def toggle_bidirectional(self) -> Dict[str, bool]:
        """Toggles the bidirectional property of the LSTM.

        :return: Dictionary with the new bidirectional state
        :rtype: Dict[str, bool]
        """
        self.bidirectional = not self.bidirectional
        self.recreate_network()

        return {"bidirectional": self.bidirectional}

    @mutation(MutationType.NODE)
    def adjust_dropout(self, dropout: Optional[float] = None) -> Dict[str, float]:
        """Adjusts the dropout rate of the LSTM.

        :param dropout: New dropout rate, defaults to None (random)
        :type dropout: Optional[float], optional

        :return: Dictionary with the new dropout rate
        :rtype: Dict[str, float]
        """
        if dropout is None:
            # Choose a random dropout rate between 0 and 0.5
            dropout = np.random.uniform(0, 0.5)

        self.dropout = dropout
        self.recreate_network()

        return {"dropout": dropout}

    @staticmethod
    def preserve_parameters(old_net: nn.Module, new_net: nn.Module) -> nn.Module:
        """Returns new neural network with copied parameters from old network. Specifically, it
        handles tensors with different sizes by copying the minimum number of elements.

        :param old_net: Old neural network
        :type old_net: nn.Module
        :param new_net: New neural network
        :type new_net: nn.Module
        :return: New neural network with copied parameters
        :rtype: nn.Module
        """

    def recreate_network(self) -> None:
        """Recreates the network after a mutation."""
        old_model = self.model
        self.model = self.create_lstm()

        # Preserve parameters from the old model using LSTM-specific preservation
        self.model = EvolvableRNN.preserve_parameters(old_model, self.model)

    @staticmethod
    def from_observation_space(
        observation_space: spaces.Box,
        num_outputs: int,
        hidden_size: Optional[List[int]] = None,
        num_layers: int = 1,
        activation: str = "ReLU",
        output_activation: Optional[str] = None,
        min_hidden_layers: int = 1,
        max_hidden_layers: int = 3,
        min_lstm_size: int = 32,
        max_lstm_size: int = 256,
        bidirectional: bool = False,
        dropout: float = 0.0,
        layer_norm: bool = True,
        output_vanish: bool = True,
        init_layers: bool = True,
        device: str = "cpu",
        name: str = "rnn",
    ) -> "EvolvableRNN":
        """Creates an EvolvableRNN from a 2D Box observation space.

        :param observation_space: 2D Box observation space
        :type observation_space: spaces.Box
        :param num_outputs: Output dimension
        :type num_outputs: int
        :param hidden_size: Size of hidden layers, defaults to None (auto-configured)
        :type hidden_size: Optional[List[int]], optional
        :param num_layers: Number of LSTM layers, defaults to 1
        :type num_layers: int, optional
        :param activation: Activation function for hidden layers, defaults to 'ReLU'
        :type activation: str, optional
        :param output_activation: Output activation function, defaults to None
        :type output_activation: Optional[str], optional
        :param min_hidden_layers: Minimum number of hidden layers, defaults to 1
        :type min_hidden_layers: int, optional
        :param max_hidden_layers: Maximum number of hidden layers, defaults to 3
        :type max_hidden_layers: int, optional
        :param min_lstm_size: Minimum size of LSTM hidden state, defaults to 32
        :type min_lstm_size: int, optional
        :param max_lstm_size: Maximum size of LSTM hidden state, defaults to 256
        :type max_lstm_size: int, optional
        :param bidirectional: Whether to use bidirectional LSTM, defaults to False
        :type bidirectional: bool, optional
        :param dropout: Dropout probability, defaults to 0.0
        :type dropout: float, optional
        :param layer_norm: Whether to use layer normalization, defaults to True
        :type layer_norm: bool, optional
        :param output_vanish: Whether to initialize output layer weights to a small value, defaults to True
        :type output_vanish: bool, optional
        :param init_layers: Whether to initialize layers, defaults to True
        :type init_layers: bool, optional
        :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
        :type device: str, optional
        :param name: Name of the network, defaults to 'lstm'
        :type name: str, optional

        :return: EvolvableLSTM instance
        :rtype: EvolvableLSTM
        """
        assert isinstance(
            observation_space, spaces.Box
        ), "Observation space must be a Box space"
        assert (
            len(observation_space.shape) == 2
        ), "Observation space must be 2D for time-series data"

        # For time-series data, the shape is typically (sequence_length, feature_dim)
        sequence_length, feature_dim = observation_space.shape

        # If hidden_size is not provided, create a default configuration
        if hidden_size is None:
            # A common pattern is to use hidden sizes that are powers of 2
            hidden_size = [64, 32]

        return EvolvableRNN(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_outputs=num_outputs,
            num_layers=num_layers,
            activation=activation,
            output_activation=output_activation,
            min_hidden_layers=min_hidden_layers,
            max_hidden_layers=max_hidden_layers,
            min_lstm_size=min_lstm_size,
            max_lstm_size=max_lstm_size,
            bidirectional=bidirectional,
            dropout=dropout,
            layer_norm=layer_norm,
            output_vanish=output_vanish,
            init_layers=init_layers,
            device=device,
            name=name,
        )
