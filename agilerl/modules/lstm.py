from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from agilerl.modules.base import EvolvableModule, MutationType, mutation
from agilerl.typing import ArrayOrTensor, BatchDimension
from agilerl.utils.evolvable_networks import get_activation


class EvolvableLSTM(EvolvableModule):
    """The Evolvable Long Short-Term Memory (LSTM) class.

    :param input_size: Size of input features
    :type input_size: int
    :param hidden_state_size: Size of hidden state
    :type hidden_state_size: int
    :param num_outputs: Output dimension
    :type num_outputs: int
    :param num_layers: Number of LSTM layers stacked together
    :type num_layers: int
    :param output_activation: Output activation layer, defaults to None
    :type output_activation: str, optional
    :param min_hidden_state_size: Minimum hidden state size, defaults to 32
    :type min_hidden_state_size: int, optional
    :param max_hidden_state_size: Maximum hidden state size, defaults to 512
    :type max_hidden_state_size: int, optional
    :param min_layers: Minimum number of LSTM layers, defaults to 1
    :type min_layers: int, optional
    :param max_layers: Maximum number of LSTM layers, defaults to 3
    :type max_layers: int, optional
    :param dropout: Dropout probability between LSTM layers, defaults to 0.0
    :type dropout: float, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param name: Name of the network, defaults to 'lstm'
    :type name: str, optional
    :param random_seed: Random seed to use for the network. Defaults to None.
    :type random_seed: Optional[int]
    """

    def __init__(
        self,
        input_size: int,
        hidden_state_size: int,
        num_outputs: int,
        num_layers: int = 1,
        output_activation: str = None,
        min_hidden_state_size: int = 32,
        max_hidden_state_size: int = 512,
        max_seq_len: int = None,
        min_layers: int = 1,
        max_layers: int = 3,
        dropout: float = 0.0,
        device: str = "cpu",
        name: str = "lstm",
        random_seed: Optional[int] = None,
    ):
        super().__init__(device, random_seed)

        assert (
            input_size > 0
        ), "'input_size' cannot be less than or equal to zero, please enter a valid integer."
        assert (
            hidden_state_size > 0
        ), "'hidden_state_size' cannot be less than or equal to zero, please enter a valid integer."
        assert (
            num_outputs > 0
        ), "'num_outputs' cannot be less than or equal to zero, please enter a valid integer."
        assert (
            num_layers > 0
        ), "'num_layers' cannot be less than or equal to zero, please enter a valid integer."
        assert (
            min_hidden_state_size < max_hidden_state_size
        ), "'min_hidden_state_size' must be less than 'max_hidden_state_size'."
        assert min_layers < max_layers, "'min_layers' must be less than 'max_layers'."
        assert 0 <= dropout < 1, "'dropout' must be between 0 and 1."

        self.name = name
        self.input_size = input_size
        self.hidden_state_size = hidden_state_size
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.output_activation = output_activation
        self.min_hidden_state_size = min_hidden_state_size
        self.max_hidden_state_size = max_hidden_state_size
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        # For LSTM, hidden state and cell state have shape (num_layers * num_directions, batch_size, hidden_size)
        # Assuming unidirectional LSTM (num_directions=1) !TODO: SHOULD WE HAVE A DIRECTIONAL LSTM IN THE FUTURE?
        self.hidden_state_architecture = {
            "h": (self.num_layers, BatchDimension, self.hidden_state_size),
            "c": (self.num_layers, BatchDimension, self.hidden_state_size),
        }

        # Create the network
        self.model = self.create_lstm()

    def create_lstm(self) -> nn.ModuleDict:
        """Creates and returns an LSTM network with the current configuration.

        :return: LSTM network
        :rtype: nn.ModuleDict
        """
        # Define network components
        model_dict = nn.ModuleDict()
        model_dict[f"{self.name}_lstm"] = nn.LSTM(
            input_size=self.input_size,
            hidden_size=int(self.hidden_state_size),
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            device=self.device,
        )

        # Add activation if specified
        model_dict[f"{self.name}_lstm_output"] = nn.Linear(
            self.hidden_state_size, self.num_outputs, device=self.device
        )
        model_dict[f"{self.name}_output_activation"] = get_activation(
            self.output_activation
        )

        return model_dict

    @property
    def net_config(self) -> Dict[str, Any]:
        """Returns model configuration in dictionary format."""
        net_config = self.init_dict.copy()
        for attr in ["input_size", "num_outputs", "device", "name"]:
            if attr in net_config:
                net_config.pop(attr)

        return net_config

    @property
    def activation(self) -> str:
        """Returns activation function."""
        return

    @activation.setter
    def activation(self, activation: str) -> None:
        """Set activation function."""
        pass

    @mutation(MutationType.ACTIVATION)
    def change_activation(self, activation: str, output: bool = False) -> None:
        """Set the output activation function for the network.

        :param activation: Activation function to use.
        :type activation: str
        :param output: Flag indicating whether to set the output activation function, defaults to False
        :type output: bool, optional
        """
        self.output_activation = activation
        self.recreate_network()

    def forward(
        self,
        x: ArrayOrTensor,
        hidden_state: Optional[Dict[str, ArrayOrTensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass of the network.

        :param x: Input tensor
        :type x: ArrayOrTensor
        :param hidden_state: Dict containing hidden and cell states, defaults to None
        :type hidden_state: Optional[Dict[str, torch.Tensor]]
        :return: Output tensor and next hidden state
        :rtype: Tuple[torch.Tensor, Dict[str, torch.Tensor]]
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        # If input is 2D (batch_size, features), add sequence length dimension
        if x.dim() == 2:
            # Reshape to (batch_size, seq_len=1, features)
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(
                f"Expected 2D (batch_size, features) or 3D (batch_size, seq_len, features) input, but got {x.dim()}D"
            )

        # Use provided hidden state if available
        if hidden_state is not None:
            h0 = hidden_state.get(f"{self.name}_h", None)
            c0 = hidden_state.get(f"{self.name}_c", None)

            lstm_output, (h_n, c_n) = self.model[f"{self.name}_lstm"](x, (h0, c0))
        else:
            raise ValueError("Hidden state is required for LSTM forward pass.")

        # Process output
        lstm_output = self.model[f"{self.name}_lstm_output"](lstm_output[:, -1, :])
        lstm_output = self.model[f"{self.name}_output_activation"](lstm_output)

        # Return output and new hidden state
        next_hidden = {
            **hidden_state,
            f"{self.name}_h": h_n,
            f"{self.name}_c": c_n,
        }
        return lstm_output, next_hidden

    def get_output_dense(self) -> torch.nn.Module:
        """Returns output layer of neural network."""
        return self.model[f"{self.name}_linear_output"]

    @mutation(MutationType.LAYER)
    def add_layer(self) -> None:
        """Adds an LSTM layer to the network. Falls back on `add_node()` if
        max layers reached."""
        if self.num_layers < self.max_layers:  # HARD LIMIT
            self.num_layers += 1
        else:
            return self.add_node()

    @mutation(MutationType.LAYER)
    def remove_layer(self) -> None:
        """Removes an LSTM layer from the network. Falls back on `add_node()` if
        min layers reached."""
        if self.num_layers > self.min_layers:  # HARD LIMIT
            self.num_layers -= 1
        else:
            return self.add_node()

    @mutation(MutationType.NODE)
    def add_node(self, numb_new_nodes: Optional[int] = None) -> Dict[str, int]:
        """Increases hidden size of the LSTM.

        :param numb_new_nodes: Number of nodes to add to hidden size, defaults to None
        :type numb_new_nodes: int, optional
        :return: Dictionary with number of new nodes
        :rtype: Dict[str, int]
        """
        if numb_new_nodes is None:
            numb_new_nodes = self.rng.choice([16, 32, 64])

        if (
            self.hidden_state_size + numb_new_nodes <= self.max_hidden_state_size
        ):  # HARD LIMIT
            self.hidden_state_size += numb_new_nodes

        return {"numb_new_nodes": numb_new_nodes}

    @mutation(MutationType.NODE)
    def remove_node(self, numb_new_nodes: Optional[int] = None) -> Dict[str, int]:
        """Decreases hidden size of the LSTM.

        :param numb_new_nodes: Number of nodes to remove from hidden size, defaults to None
        :type numb_new_nodes: int, optional
        :return: Dictionary with number of new nodes
        :rtype: Dict[str, int]
        """
        if numb_new_nodes is None:
            numb_new_nodes = self.rng.choice([16, 32, 64])

        if (
            self.hidden_state_size - numb_new_nodes >= self.min_hidden_state_size
        ):  # HARD LIMIT
            self.hidden_state_size -= numb_new_nodes

        return {"numb_new_nodes": numb_new_nodes}

    def recreate_network(self) -> None:
        """Recreates the LSTM network with current parameters."""
        model = self.create_lstm()

        # Preserve parameters where possible
        self.model = EvolvableModule.preserve_parameters(
            old_net=self.model, new_net=model
        )
