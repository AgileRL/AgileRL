import copy
from dataclasses import asdict
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from agilerl.modules import EvolvableCNN, EvolvableLSTM, EvolvableMLP
from agilerl.modules.base import EvolvableModule, ModuleDict, MutationType, mutation
from agilerl.modules.configs import CnnNetConfig, LstmNetConfig, MlpNetConfig
from agilerl.typing import ArrayOrTensor, ConfigType
from agilerl.utils.evolvable_networks import (
    get_activation,
    is_image_space,
    is_vector_space,
    tuple_to_dict_space,
)

ModuleType = Union[EvolvableModule, nn.Module]
SupportedEvolvableTypes = Union[EvolvableCNN, EvolvableMLP, EvolvableLSTM]
MultiInputConfigType = Union[ConfigType, Dict[str, ConfigType]]
TupleOrDictSpace = Union[spaces.Tuple, spaces.Dict]
TupleOrDictObservation = Union[Dict[str, ArrayOrTensor], Tuple[ArrayOrTensor]]

# Default configurations for the feature extractors
DefaultCnnConfig = CnnNetConfig(
    channel_size=[16, 16],
    kernel_size=[3, 3],
    stride_size=[1, 1],
    output_activation="ReLU",
)
DefaultMlpConfig = MlpNetConfig(
    hidden_size=[64, 64],
    output_activation="ReLU",
)
DefaultLstmConfig = LstmNetConfig(
    hidden_size=64,
    num_layers=2,
    output_activation="ReLU",
)


def get_total_flatdim(observation_spaces: spaces.Dict) -> int:
    """Get the total flat dimension of the observation space.

    :param observation_space: Dictionary or Tuple space of observations.
    :type observation_space: spaces.Dict or spaces.Tuple
    :return: Total flat dimension of the observation space.
    """
    return sum([spaces.flatdim(space) for space in observation_spaces.spaces.values()])


class EvolvableMultiInput(EvolvableModule):
    """Evolvable multi-input network for `Tuple` or `Dict` observation spaces. It inspects the
    observation space to determine the type of network to build for each key. It builds an
    `EvolvableCNN` for image subspaces, an `EvolvableLSTM` for time-series subspaces, and a
    `nn.Flatten()` for vector subspaces with more than 3 dimensions. Vector observations are
    concatenated with the extracted features before passing through an `EvolvableMLP` to produce
    the output tensor. Optionally, users may specify an additional `EvolvableMLP` to be applied to
    the concatenated vector observations before concatenation with the extracted features.

    :param observation_space: Dictionary or Tuple space of observations.
    :type observation_space: spaces.Dict or spaces.Tuple
    :param num_outputs: Dimension of the output tensor.
    :type num_outputs: int
    :param latent_dim: Dimension of the latent space representation. Default is 16.
    :type latent_dim: int, optional
    :param vector_space_mlp: Whether to use an MLP for the vector spaces. This is done by concatenating the
        flattened observations and passing them through an `EvolvableMLP`. Default is False, whereby the
        observations are concatenated directly to the feature encodings before the final MLP.
    :type vector_space_mlp: bool, optional
    :param cnn_config: Configuration for the CNN feature extractor. Default is None.
    :type cnn_config: MultiInputConfigType, optional
    :param mlp_config: Configuration for the MLP feature extractor. Default is None.
    :type mlp_config: MultiInputConfigType, optional
    :param lstm_config: Configuration for the LSTM feature extractor. Default is None.
    :type lstm_config: MultiInputConfigType, optional
    :param init_dicts: Dictionary of initialization dictionaries for the feature extractors. Default is {}.
    :type init_dicts: Dict[str, Dict[str, Any]], optional
    :param output_activation: Activation function for the output layer. Default is None.
    :type output_activation: Optional[str], optional
    :param device: Device to use for the network. Default is "cpu".
    :type device: str, optional
    :param name: Name of the network. Default is "multi_input".
    :type name: str, optional
    """

    feature_net: ModuleDict
    _SupportedSpaces = (spaces.Box, spaces.Discrete, spaces.MultiDiscrete)

    def __init__(
        self,
        observation_space: TupleOrDictSpace,
        num_outputs: int,
        latent_dim: int = 16,
        vector_space_mlp: bool = False,
        cnn_config: Optional[MultiInputConfigType] = None,
        mlp_config: Optional[MultiInputConfigType] = None,
        lstm_config: Optional[MultiInputConfigType] = None,
        init_dicts: Optional[MultiInputConfigType] = None,
        output_activation: Optional[str] = None,
        min_latent_dim: int = 8,
        max_latent_dim: int = 128,
        device: str = "cpu",
        name: str = "multi_input",
    ):
        super().__init__(device)

        assert num_outputs > 0, "Number of outputs must be greater than 0."
        assert latent_dim > 0, "Latent dimension must be greater than 0."
        assert isinstance(
            observation_space, (spaces.Dict, spaces.Tuple)
        ), "Observation space must be a Dict or Tuple space."

        subspaces = (
            observation_space.spaces.values()
            if isinstance(observation_space, spaces.Dict)
            else observation_space.spaces
        )
        assert all(
            [isinstance(space, self._SupportedSpaces) for space in subspaces]
        ), "Observation space must contain only Box, Discrete, or MultiDiscrete spaces."

        # Convert Tuple space to Dict space for consistency
        self.is_tuple_space = False
        if isinstance(observation_space, spaces.Tuple):
            observation_space = tuple_to_dict_space(observation_space)
            self.is_tuple_space = True

        self.observation_space = observation_space
        self.num_outputs = num_outputs
        self.cnn_config = cnn_config or copy.deepcopy(asdict(DefaultCnnConfig))
        self.mlp_config = mlp_config or copy.deepcopy(asdict(DefaultMlpConfig))
        self.lstm_config = lstm_config or copy.deepcopy(asdict(DefaultLstmConfig))
        self._init_dicts = init_dicts or {}
        self._activation = None

        self.vector_space_mlp = vector_space_mlp
        self.latent_dim = latent_dim
        self.output_activation = output_activation
        self.min_latent_dim = min_latent_dim
        self.max_latent_dim = max_latent_dim
        self.name = name
        self.device = device

        self.vector_spaces = spaces.Dict(
            {
                key: space
                for key, space in observation_space.spaces.items()
                if is_vector_space(space)
            }
        )
        self.total_vector_dims = get_total_flatdim(self.vector_spaces)

        # Build feature extractor
        self.feature_net = self.build_feature_extractor()

        # Calculate total extracted features from non-vector spaces
        # (i.e. Box spaces with more than one dimension)
        self.extracted_features_dim = self.calc_extracted_features_dim()

        # Vector observations (i.e. 1D Box, Discrete, MultiDiscrete) are either
        # passed through an MLP or concatenated directly to the extracted features
        features_dim = self.extracted_features_dim + self.total_vector_dims * (
            1 - self.vector_space_mlp
        )

        # Final dense layer to convert feature encodings to desired num_outputs
        self.final_dense = nn.Linear(features_dim, num_outputs, device=device)
        self.output = get_activation(output_activation)

    @property
    def net_config(self) -> Dict[str, Any]:
        """Returns the configuration of the network.

        :return: Network configuration
        :rtype: Dict[str, Any]
        """
        net_config = self.init_dict.copy()
        for attr in ["observation_space", "num_outputs", "device", "name"]:
            net_config.pop(attr)
        return net_config

    @property
    def activation(self) -> str:
        """Get the activation function for the network.

        :return: Activation function
        :rtype: str
        """
        return self._activation

    @activation.setter
    def activation(self, activation: str) -> None:
        """Set the activation function for the network.

        :param activation: Activation function to use.
        :type activation: str
        """
        self._activation = activation
        for module in self.feature_net.modules().values():
            module.change_activation(activation, output=True)

    @property
    def init_dicts(self) -> Dict[str, Dict[str, Any]]:
        """Returns the initialization dictionaries for the network.

        :return: Initialization dictionaries
        :rtype: Dict[str, Dict[str, Any]]
        """
        if not hasattr(self, "feature_net"):
            return self._init_dicts

        reformatted_dicts = {}
        for key, net in self.feature_net.modules().items():
            init_dict = net.init_dict
            init_dict.pop("input_size", None)  # LSTM
            init_dict.pop("input_shape", None)  # CNN
            init_dict.pop("num_inputs", None)  # MLP
            init_dict["num_outputs"] = self.latent_dim
            reformatted_dicts[key] = init_dict

        return reformatted_dicts

    @property
    def cnn_init_dict(self) -> Dict[str, Any]:
        """Returns the initialization dictionary for the CNN."""
        return copy.deepcopy(self.cnn_config)

    @property
    def mlp_init_dict(self) -> Dict[str, Any]:
        """Returns the initialization dictionary for the MLP."""
        return copy.deepcopy(self.mlp_config)

    @property
    def lstm_init_dict(self) -> Dict[str, Any]:
        """Returns the initialization dictionary for the LSTM."""
        return copy.deepcopy(self.lstm_config)

    def init_weights_gaussian(
        self, std_coeff: float = 4, output_coeff: float = 4
    ) -> None:
        """Initialise weights of linear layers using Gaussian distribution."""
        for module in self.feature_net.modules().values():
            module.init_weights_gaussian(std_coeff=std_coeff)

        # Initialise final dense layer
        EvolvableModule.init_weights_gaussian(self.final_dense, std_coeff=output_coeff)

    def calc_extracted_features_dim(self) -> int:
        """Calculates the dimension of the extracted features."""
        extracted_features_dim = 0
        for name in self.feature_net.keys():
            if name not in self.vector_spaces:
                subspace = self.observation_space.spaces.get(name)
                if subspace is None:
                    extracted_features_dim += self.latent_dim
                else:
                    extracted_features_dim += (
                        spaces.flatdim(subspace)
                        if len(subspace.shape) > 3
                        else self.latent_dim
                    )

        return extracted_features_dim

    def get_inner_init_dict(
        self, key: str, default: Literal["cnn", "mlp", "lstm"]
    ) -> Dict[str, Any]:
        """Returns the initialization dictionary for the specified key.

        Arguments:
            key (str): Key of the observation space.

        Returns:
            Dict[str, Any]: Initialization dictionary.
        """
        init_dicts = self.init_dicts
        if key in init_dicts:
            return init_dicts[key]

        assert default in [
            "cnn",
            "mlp",
            "lstm",
        ], "Invalid default value provided, must be 'cnn' or 'mlp' or 'lstm'."

        if default == "cnn":
            init_dict = self.cnn_init_dict
        elif default == "mlp":
            init_dict = self.mlp_init_dict
        elif default == "lstm":
            init_dict = self.lstm_init_dict

        init_dict["num_outputs"] = self.latent_dim
        init_dict["device"] = self.device
        return init_dict

    def build_feature_extractor(self) -> Dict[str, SupportedEvolvableTypes]:
        """Creates the feature extractor and final MLP networks.

        :return: Dictionary of feature extractors.
        :rtype: Dict[str, Union[EvolvableCNN, EvolvableMLP]]
        """
        # Automatically build feature extractors from subspaces
        feature_net = ModuleDict(device=self.device)
        for key, space in self.observation_space.spaces.items():
            if is_image_space(space):  # Use CNN if it's an image space
                init_dict = copy.deepcopy(self.get_inner_init_dict(key, default="cnn"))
                feature_extractor = EvolvableCNN(
                    input_shape=space.shape,
                    name=init_dict.pop("name", key),
                    **init_dict
                )
            elif isinstance(space, spaces.Box) and key not in self.vector_spaces.keys():
                # We assume 2D Box spaces correspond to time-series data.
                # For other types of data, it is probably adequate to flatten the space
                # and use an MLP.
                if len(space.shape) == 2:
                    init_dict = copy.deepcopy(
                        self.get_inner_init_dict(key, default="lstm")
                    )
                    feature_extractor = EvolvableLSTM(
                        input_size=space.shape[1],
                        name=init_dict.pop("name", key),
                        **init_dict
                    )
                elif len(space.shape) > 2:
                    feature_extractor = nn.Flatten()
            else:
                continue

            feature_net[key] = feature_extractor

        # Optionally, use an EvolvableMLP for all concatenated vector inputs
        if self.vector_space_mlp:
            init_dict = copy.deepcopy(
                self.get_inner_init_dict("vector_mlp", default="mlp")
            )
            vector_mlp = EvolvableMLP(
                num_inputs=self.total_vector_dims,
                name=init_dict.pop("name", "vector_mlp"),
                **init_dict
            )

            self.init_dicts["vector_mlp"] = vector_mlp.init_dict
            feature_net["vector_mlp"] = vector_mlp

        return feature_net

    def forward(self, x: TupleOrDictObservation) -> torch.Tensor:
        """Forward pass of the composed network. Extracts features from each observation key and concatenates
        them with the corresponding observation key if specified. The concatenated features are then passed
        through the final MLP to produce the output tensor.

        :param x: Dictionary of observations.
        :type x: Dict[str, ArrayOrTensor], Tuple[ArrayOrTensor]
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        if isinstance(x, tuple):
            x = dict(zip(self.observation_space.spaces.keys(), x))

        for key, obs in x.items():
            if not isinstance(obs, torch.Tensor):
                x[key] = torch.tensor(obs, device=self.device, dtype=torch.float32)

        # Extract features from non-vector subspaces
        if self.extracted_features_dim > 0:
            extracted_features = [
                self.feature_net[key](x[key])
                for key in x.keys()
                if key in self.feature_net.keys()
            ]
            extracted_features = torch.cat(extracted_features, dim=1)
        else:
            extracted_features = torch.tensor([], device=self.device)

        # Extract raw features from vector spaces
        vector_inputs = []
        for key in self.vector_spaces:
            _obs = x[key]
            if len(_obs.shape) == 1:
                _obs = _obs.unsqueeze(0)

            vector_inputs.append(_obs)

        # Concatenate vector inputs and, optionally, pass through additional EvolvableMLP
        vector_inputs = torch.cat(vector_inputs, dim=1)
        if self.vector_space_mlp:
            vector_features = self.feature_net["vector_mlp"](vector_inputs)
        else:
            vector_features = vector_inputs

        # Concatenate all features and pass through final MLP
        features = torch.cat([extracted_features, vector_features], dim=1)
        latent = self.final_dense(features)
        return self.output(latent)

    @mutation(MutationType.ACTIVATION)
    def change_activation(self, activation: str, output: bool = False) -> None:
        """Set the activation function for the network.

        :param activation: Activation function to use.
        :type activation: str
        :param output: Flag indicating whether to set the output activation function, defaults to False
        :type output: bool, optional

        :return: Activation function
        :rtype: str
        """
        self.activation = activation
        if output:
            self.output = get_activation(activation)

    @mutation(MutationType.NODE)
    def add_latent_node(self, numb_new_nodes: Optional[int] = None) -> Dict[str, Any]:
        """Add a latent node to the network.

        :param numb_new_nodes: Number of new nodes to add, defaults to None
        :type numb_new_nodes: int, optional

        :return: Configuration for adding a latent node.
        :rtype: Dict[str, Any]
        """
        if numb_new_nodes is None:
            numb_new_nodes = np.random.choice([8, 16, 32], 1)[0]

        if self.latent_dim + numb_new_nodes < self.max_latent_dim:
            self.latent_dim += numb_new_nodes

        return {"numb_new_nodes": numb_new_nodes}

    @mutation(MutationType.NODE)
    def remove_latent_node(
        self, numb_new_nodes: Optional[int] = None
    ) -> Dict[str, Any]:
        """Remove a latent node from the network.

        :param numb_new_nodes: Number of nodes to remove, defaults to None
        :type numb_new_nodes: int, optional

        :return: Configuration for removing a latent node.
        :rtype: Dict[str, Any]
        """
        if numb_new_nodes is None:
            numb_new_nodes = np.random.choice([8, 16, 32], 1)[0]

        if self.latent_dim - numb_new_nodes > self.min_latent_dim:
            self.latent_dim -= numb_new_nodes

        return {"numb_new_nodes": numb_new_nodes}

    def recreate_network(self) -> None:
        """Recreates the network with the new latent dimension."""
        feature_net = self.build_feature_extractor()
        self.feature_net = EvolvableModule.preserve_parameters(
            old_net=self.feature_net, new_net=feature_net
        )

        # Calculate total extracted features dimension
        extracted_features_dim = self.calc_extracted_features_dim()
        vector_features_dim = (
            self.latent_dim if self.vector_space_mlp else self.total_vector_dims
        )

        features_dim = extracted_features_dim + vector_features_dim
        final_dense = nn.Linear(features_dim, self.num_outputs, device=self.device)
        self.final_dense = EvolvableModule.preserve_parameters(
            old_net=self.final_dense, new_net=final_dense
        )
