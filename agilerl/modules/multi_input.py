from typing import List, Optional, Dict, Any, Union, Literal, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from agilerl.typing import ArrayOrTensor
from agilerl.utils.evolvable_networks import is_image_space, get_activation
from agilerl.modules.base import EvolvableModule, ModuleDict, register_mutation_fn, MutationType
from agilerl.modules.cnn import EvolvableCNN
from agilerl.modules.mlp import EvolvableMLP

ModuleType = Union[EvolvableModule, nn.Module]
SupportedEvolvableTypes = Union[EvolvableCNN, EvolvableMLP]
TupleOrDictSpace = Union[spaces.Tuple, spaces.Dict]
TupleOrDictObservation = Union[Dict[str, ArrayOrTensor], Tuple[ArrayOrTensor]]

def tuple_to_dict_space(observation_space: spaces.Tuple) -> spaces.Dict:
    """Converts a Tuple observation space to a Dict observation space.
    
    :param observation_space: Tuple observation space.
    :type observation_space: spaces.Tuple
    :return: Dictionary observation space.
    :rtype: spaces.Dict
    """
    return spaces.Dict({
        f"observation_{i}": space for i, space in enumerate(observation_space.spaces)
    })


class EvolvableMultiInput(EvolvableModule):
    """
    General object that allows for the composition of multiple evolvable networks given a dictionary 
    observation space. For each key in the space, either a `EvolvableCNN` or `EvolvableMLP` object is 
    created and used to process the corresponding observation. Each network is a feature extractor and 
    their output is concatenated and passed through a final `EvolvableMLP` to produce the final output 
    given the action space.

    .. note::
        The user can inherit from this class and override the `forward` method to define a custom 
        forward pass for the network, manipulating the observation dictionary as needed, and adding any 
        non-evolvable additional layers needed for their specific problem.

    .. note::
        The mutations are done on the basis of the allowed methods in the `EvolvableCNN` and `EvolvableMLP`
        classes. For any selected method, we choose a random module containing the method to apply it against.

    :param observation_space: Dictionary or Tuple space of observations.
    :type observation_space: spaces.Dict, spaces.Tuple
    :param num_outputs: Dimension of the output tensor.
    :type num_outputs: int
    :param channel_size: List of channel sizes for the convolutional layers.
    :type channel_size: List[int]
    :param kernel_size: List of kernel sizes for the convolutional layers.
    :type kernel_size: List[int]
    :param stride_size: List of stride sizes for the convolutional layers.
    :type stride_size: List[int]
    :param latent_dim: Dimension of the latent space representation. Default is 16.
    :type latent_dim: int, optional
    :param cnn_block_type: Type of convolutional block to use. Default is "Conv2d".
    :type cnn_block_type: Literal["Conv2d", "Conv3d"], optional
    :param sample_input: Sample input tensor for the CNN. Default is None.
    :type sample_input: Optional[torch.Tensor], optional
    :param vector_space_mlp: Whether to use an MLP for the vector spaces. This is done by concatenating the 
        flattened observations and passing them through an `EvolvableMLP`. Default is False, whereby the 
        observations are concatenated directly to the feature encodings before the final MLP.
    :type vector_space_mlp: bool, optional
    :param hidden_size: List of hidden sizes for the MLP. Default is None.
    :type hidden_size: List[int], optional
    :param init_dicts: Dictionary of initialization dictionaries for the feature extractors. Default is {}.
    :type init_dicts: Dict[str, Dict[str, Any]], optional
    :param output_activation: Activation function for the output layer. Default is None.
    :type output_activation: Optional[str], optional
    :param activation: Activation function for the module layers. Default is "ReLU".
    :type activation: str, optional
    :param min_hidden_layers: Minimum number of hidden layers for the MLP. Default is 1.
    :type min_hidden_layers: int, optional
    :param max_hidden_layers: Maximum number of hidden layers for the MLP. Default is 3.
    :type max_hidden_layers: int, optional
    :param min_mlp_nodes: Minimum number of nodes for the MLP. Default is 64.
    :type min_mlp_nodes: int, optional
    :param max_mlp_nodes: Maximum number of nodes for the MLP. Default is 1024.
    :type max_mlp_nodes: int, optional
    :param min_cnn_hidden_layers: Minimum number of hidden layers for the CNN. Default is 1.
    :type min_cnn_hidden_layers: int, optional
    :param max_cnn_hidden_layers: Maximum number of hidden layers for the CNN. Default is 6.
    :type max_cnn_hidden_layers: int, optional
    :param min_channel_size: Minimum channel size for the CNN. Default is 32.
    :type min_channel_size: int, optional
    :param max_channel_size: Maximum channel size for the CNN. Default is 256.
    :type max_channel_size: int, optional
    :param layer_norm: Whether to use layer normalization. Default is False.
    :type layer_norm: bool, optional
    :param noise_std: Standard deviation of the noise. Default is 0.5.
    :type noise_std: float, optional
    :param noisy: Whether to use noisy layers. Default is False.
    :type noisy: bool, optional
    :param init_layers: Whether to initialize the layers. Default is True.
    :type init_layers: bool, optional
    :param device: Device to use for the network. Default is "cpu".
    :type device: str, optional
    """
    feature_net: Dict[str, SupportedEvolvableTypes]

    arch: str = "multi_input"

    def __init__(
            self,
            observation_space: TupleOrDictSpace,
            num_outputs: int,
            channel_size: List[int],
            kernel_size: List[int],
            stride_size: List[int],
            latent_dim: int = 16,
            cnn_block_type: Literal["Conv2d", "Conv3d"] = "Conv2d",
            sample_input: Optional[torch.Tensor] = None,
            vector_space_mlp: bool = False,
            hidden_size: Optional[List[int]] = None,
            init_dicts: Dict[str, Dict[str, Any]] = {},
            output_activation: Optional[str] = None,
            activation: str = "ReLU",
            min_hidden_layers: int = 1,
            max_hidden_layers: int = 3,
            min_mlp_nodes: int = 64,
            max_mlp_nodes: int = 1024,
            min_cnn_hidden_layers: int = 1,
            max_cnn_hidden_layers: int = 6,
            min_channel_size: int = 32,
            max_channel_size: int = 256,
            min_latent_dim: int = 8,
            max_latent_dim: int = 128,
            layer_norm: bool = False,
            noisy: bool = False,
            noise_std: float = 0.5,
            init_layers: bool = True,
            device: str = "cpu",
            name: str = "multi_input"
        ):
        super().__init__(device)

        assert (
            isinstance(observation_space, (spaces.Dict, spaces.Tuple)),
            "Observation space must be a Dict or Tuple space."
        )

        if isinstance(observation_space, spaces.Tuple):
            observation_space = tuple_to_dict_space(observation_space)

        if vector_space_mlp:
            assert hidden_size is not None, "Hidden size must be specified for vector space MLP."
        
        self.observation_space = observation_space
        self.num_outputs = num_outputs
        self.name = name
        self.vector_space_mlp = vector_space_mlp
        self.channel_size = channel_size
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.hidden_size = hidden_size
        self.sample_input = sample_input
        self.cnn_block_type = cnn_block_type
        self.output_activation = output_activation
        self.min_hidden_layers = min_hidden_layers
        self.max_hidden_layers = max_hidden_layers
        self.min_mlp_nodes = min_mlp_nodes
        self.max_mlp_nodes = max_mlp_nodes
        self.min_latent_dim = min_latent_dim
        self.max_latent_dim = max_latent_dim
        self.min_cnn_hidden_layers = min_cnn_hidden_layers
        self.max_cnn_hidden_layers = max_cnn_hidden_layers
        self.min_channel_size = min_channel_size
        self.max_channel_size = max_channel_size
        self.layer_norm = layer_norm
        self.init_layers = init_layers
        self.latent_dim = latent_dim
        self.noisy = noisy
        self.noise_std = noise_std
        self._init_dicts = init_dicts
        self._activation = activation
        self.vector_spaces = [
            key for key, space in observation_space.spaces.items() 
            if not is_image_space(space)
            ]

        self.feature_net = self.build_feature_extractor()

        # Collect all vector space shapes for concatenation
        vector_input_dim = sum(
            [spaces.flatdim(self.observation_space.spaces[key]) for key in self.vector_spaces]
            )

        # Calculate total feature dimension for final MLP
        image_features_dim = sum(
            [self.latent_dim for subspace in self.observation_space.spaces.values() if is_image_space(subspace)]
            )

        vector_features_dim = self.latent_dim if self.vector_space_mlp else vector_input_dim
        features_dim = image_features_dim + vector_features_dim

        # Final dense layer to convert feature encodings to desired num_outputs
        self.final_dense = nn.Linear(features_dim, num_outputs, device=device)
        self.output = get_activation(output_activation)

        # If we dont define an EvolvableMLP for vector spaces, we should signal this for Mutations
        if not vector_space_mlp:
            self.filter_mutation_methods("mlp")
    
    @property
    def net_config(self) -> Dict[str, Any]:
        """Returns the configuration of the network.
        
        :return: Network configuration
        :rtype: Dict[str, Any]
        """
        net_config = self.init_dict.copy()
        for attr in ['observation_space', 'num_outputs', 'device', 'name']:
            net_config.pop(attr)
        return net_config
    
    @property
    def activation(self) -> str:
        """Returns the activation function for the network.
        
        :return: Activation function
        :rtype: str
        """
        return self._activation
    
    @activation.setter
    def activation(self, activation: str) -> None:
        """Sets the activation function for the network.
        
        :param activation: Activation function to use.
        :type activation: str
        """
        self._activation = activation

    @property
    def base_init_dict(self) -> Dict[str, Any]:
        """Returns dictionary of base information.
        
        :return: Base information
        :rtype: Dict[str, Any]
        """
        return {
            "num_outputs": self.latent_dim,
            "layer_norm": self.layer_norm,
            "init_layers": self.init_layers,
            "device": self.device
        }

    @property
    def mlp_init_dict(self) -> Dict[str, Any]:
        """Returns dictionary of MLP information.
        
        :return: MLP information
        :rtype: Dict[str, Any]
        """
        base = self.base_init_dict.copy()
        extra_kwargs = {
            "hidden_size": self.hidden_size,
            "activation": self.activation,
            "output_activation": self.activation,
            "min_hidden_layers": self.min_hidden_layers,
            "max_hidden_layers": self.max_hidden_layers,
            "min_mlp_nodes": self.min_mlp_nodes,
            "max_mlp_nodes": self.max_mlp_nodes,
            "noisy": self.noisy,
            "noise_std": self.noise_std
        }
        base.update(extra_kwargs)
        return base

    @property
    def cnn_init_dict(self) -> Dict[str, Any]:
        """Returns dictionary of CNN information.
        
        :return: CNN information
        :rtype: Dict[str, Any]
        """
        base = self.base_init_dict.copy()
        extra_kwargs = {
            "channel_size": self.channel_size,
            "kernel_size": self.kernel_size,
            "stride_size": self.stride_size,
            "activation": self.activation,
            "sample_input": self.sample_input,
            "block_type": self.cnn_block_type,
            "output_activation": self.activation,
            "min_hidden_layers": self.min_cnn_hidden_layers,
            "max_hidden_layers": self.max_cnn_hidden_layers,
            "min_channel_size": self.min_channel_size,
            "max_channel_size": self.max_channel_size,
        }
        base.update(extra_kwargs)
        return base

    @property
    def init_dict(self) -> Dict[str, Any]:
        """Returns model information in dictionary.
        
        :return: Model information
        :rtype: Dict[str, Any]
        """
        kwargs = self.base_init_dict.copy()
        extra_kwargs = {
            "observation_space": self.observation_space,
            "num_outputs": self.num_outputs,
            "latent_dim": self.latent_dim,
            "vector_space_mlp": self.vector_space_mlp,
            "init_dicts": self.init_dicts,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "name": self.name,
            # CNN kwargs
            "channel_size": self.channel_size,
            "kernel_size": self.kernel_size,
            "stride_size": self.stride_size,
            "sample_input": self.sample_input,
            "cnn_block_type": self.cnn_block_type,
            "min_cnn_hidden_layers": self.min_cnn_hidden_layers,
            "max_cnn_hidden_layers": self.max_cnn_hidden_layers,
            "min_channel_size": self.min_channel_size,
            "max_channel_size": self.max_channel_size,
            # MLP kwargs
            "hidden_size": self.hidden_size,
            "min_hidden_layers": self.min_hidden_layers,
            "max_hidden_layers": self.max_hidden_layers,
            "min_mlp_nodes": self.min_mlp_nodes,
            "max_mlp_nodes": self.max_mlp_nodes
        }
        kwargs.update(extra_kwargs)
        return kwargs
    
    @property
    def init_dicts(self) -> Dict[str, Dict[str, Any]]:
        """Returns the initialization dictionaries for the network.
        
        :return: Initialization dictionaries
        :rtype: Dict[str, Dict[str, Any]]
        """
        if not self._init_dicts:
            return {}

        reformatted_dicts = {}
        for key, d in self._init_dicts.items():
            d.pop("input_shape", None)
            d.pop("num_inputs", None)
            d['num_outputs'] = self.latent_dim
            reformatted_dicts[key] = d

        return reformatted_dicts
    
    def init_weights_gaussian(self, std_coeff: float = 4, output_coeff: float = 4) -> None:
        """Initialise weights of linear layers using Gaussian distribution."""
        for module in self.feature_net.values():
            module.init_weights_gaussian(std_coeff=std_coeff)

        # Initialise final dense layer
        EvolvableModule.init_weights_gaussian(self.final_dense, std_coeff=output_coeff)
    
    def get_init_dict(self, key: str, default: Literal['cnn', 'mlp']) -> Dict[str, Any]:
        """Returns the initialization dictionary for the specified key.
        
        Arguments:
            key (str): Key of the observation space.
        
        Returns:
            Dict[str, Any]: Initialization dictionary.
        """
        init_dicts = self.init_dicts
        if key in init_dicts:
            return init_dicts[key]
        
        assert default in ['cnn', 'mlp'], "Invalid default value provided, must be 'cnn' or 'mlp'."

        return self.cnn_init_dict if default == 'cnn' else self.mlp_init_dict

    def build_feature_extractor(self) -> Dict[str, SupportedEvolvableTypes]:
        """Creates the feature extractor and final MLP networks.
        
        Returns:
            Tuple[Dict[str, ModuleType], EvolvableMLP, EvolvableMLP]: Tuple containing the feature extractor,
                value network, and advantage network.
        """
        # Build feature extractors for image spaces only
        feature_net = ModuleDict(device=self.device)
        for key, space in self.observation_space.spaces.items():
            if is_image_space(space):  # Use CNN if it's an image space
                init_dict = copy.deepcopy(self.get_init_dict(key, default='cnn'))
                feature_extractor = EvolvableCNN(
                    input_shape=space.shape,
                    name=init_dict.pop("name", key),
                    **init_dict
                    )

                self._init_dicts[key] = feature_extractor.init_dict
                feature_net[key] = feature_extractor

        # Collect all vector space shapes for concatenation
        vector_input_dim = sum(
            [spaces.flatdim(self.observation_space.spaces[key]) for key in self.vector_spaces]
            )

        # Optional MLP for all concatenated vector inputs
        if self.vector_space_mlp:
            assert (
                self.hidden_size is not None,
                "Hidden size must be specified for vector space MLP."
            )
            init_dict = copy.deepcopy(self.get_init_dict("vector_mlp", default='mlp'))
            vector_mlp = EvolvableMLP(
                num_inputs=vector_input_dim,
                name=init_dict.pop("name", "vector_mlp"),
                **init_dict
                )

            self._init_dicts["vector_mlp"] = vector_mlp.init_dict
            feature_net["vector_mlp"] = vector_mlp
        
        return feature_net

    def forward(self, x: TupleOrDictObservation) -> torch.Tensor:
        """Forward pass of the composed network. Extracts features from each observation key and concatenates
        them with the corresponding observation key if specified. The concatenated features are then passed
        through the final MLP to produce the output tensor.
        
        :param x: Dictionary of observations.
        :type x: Dict[str, ArrayOrTensor], Tuple[ArrayOrTensor]
        :param xc: Optional additional input tensor for critic network, defaults to None.
        :type xc: Optional[ArrayOrTensor], optional
        :param q: Flag to indicate if Q-values should be computed, defaults to True.
        :type q: bool, optional
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        if isinstance(x, tuple):
            x = dict(zip(self.observation_space.spaces.keys(), x))
        
        # Extract features from image spaces
        image_features = [
            self.feature_net[key](x[key]) for key in x.keys() if key in self.feature_net.keys()
            ]
        image_features = torch.cat(image_features, dim=1)

        # Extract raw features from vector spaces
        vector_inputs = []
        for key in self.vector_spaces:
            # Flatten if necessary 
            if len(x[key].shape) > 2:
                x[key] = x[key].flatten(start_dim=1)
            
            vector_inputs.append(x[key])

        # Concatenate vector inputs
        vector_inputs = torch.cat(vector_inputs, dim=1)

        # Pass through optional MLP
        if self.vector_space_mlp:
            vector_features = self.feature_net['vector_mlp'](vector_inputs)
        else:
            vector_features = vector_inputs

        # Concatenate all features and pass through final MLP
        features = torch.cat([image_features, vector_features], dim=1)
        return self.output(self.final_dense(features))

    @register_mutation_fn(MutationType.ACTIVATION)
    def change_activation(self, activation: str, output: bool = False) -> None:
        """Set the activation function for the network.

        :param activation: Activation function to use.
        :type activation: str
        :param output: Flag indicating whether to set the output activation function, defaults to False
        :type output: bool, optional

        :return: Activation function
        :rtype: str
        """
        for key, module in self.feature_net.items():
            module.change_activation(activation, output=True)
            self.init_dicts[key] = self.extract_init_dict(key)
        
        if output:
            self.output = get_activation(activation)

    @register_mutation_fn(MutationType.NODE)
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
            self.recreate_network()

        return {"numb_new_nodes": numb_new_nodes}

    @register_mutation_fn(MutationType.NODE)
    def remove_latent_node(self, numb_new_nodes: Optional[int] = None) -> Dict[str, Any]:
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
            self.recreate_network(shrink_params=True)

        return {"numb_new_nodes": numb_new_nodes}
    
    def recreate_network(self, shrink_params: bool = False) -> None:
        """Recreates the network with the new latent dimension.
        
        :param shrink_params: Flag to indicate if the network should be recreated 
            with smaller parameters, defaults to False.
        :type shrink_params: bool, optional
        """
        feature_net = self.build_feature_extractor()

        # Collect all vector space shapes for concatenation
        vector_input_dim = sum(
            [spaces.flatdim(self.observation_space.spaces[key]) for key in self.vector_spaces]
            )

        # Calculate total feature dimension for final MLP
        image_features_dim = sum(
            [self.latent_dim for subspace in self.observation_space.spaces.values() if is_image_space(subspace)]
            )

        vector_features_dim = self.latent_dim if self.vector_space_mlp else vector_input_dim
        features_dim = image_features_dim + vector_features_dim

        final_dense = nn.Linear(features_dim, self.num_outputs, device=self.device)

        # Copy parameters from old model to new model
        preserve_params_fn = (
            EvolvableModule.shrink_preserve_parameters if shrink_params 
            else EvolvableModule.preserve_parameters
        )

        self.feature_net = preserve_params_fn(old_net=self.feature_net, new_net=feature_net)
        self.final_dense = preserve_params_fn(old_net=self.final_dense, new_net=final_dense)