from typing import List, Optional, Dict, Any, Union, Literal, Tuple
import copy

import torch
import torch.nn as nn
from gymnasium import spaces

from agilerl.typing import ArrayOrTensor
from agilerl.utils.evolvable_networks import is_image_space, get_activation
from agilerl.modules.base import EvolvableModule, register_mutation_fn, MutationType
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
    :param channel_size: List of channel sizes for the convolutional layers.
    :type channel_size: List[int]
    :param kernel_size: List of kernel sizes for the convolutional layers.
    :type kernel_size: List[int]
    :param stride_size: List of stride sizes for the convolutional layers.
    :type stride_size: List[int]
    :param hidden_size: List of hidden sizes for the MLP layers.
    :type hidden_size: List[int]
    :param num_outputs: Dimension of the output tensor.
    :type num_outputs: int
    :param vector_space_mlp: Whether to use an MLP for the vector spaces. This is done by concatenating the 
        flattened observations and passing them through an `EvolvableMLP`. Default is False, whereby the 
        observations are concatenated directly to the feature encodings before the final MLP.
    :type vector_space_mlp: bool, optional
    :param mlp_output_activation: Activation function for the output of the final MLP. Default is None.
    :type mlp_output_activation: Optional[str], optional
    :param mlp_activation: Activation function for the MLP layers. Default is "ReLU".
    :type mlp_activation: str, optional
    :param cnn_activation: Activation function for the CNN layers. Default is "ReLU".
    :type cnn_activation: str, optional
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
    :param n_agents: Number of agents. Default is None.
    :type n_agents: Optional[int], optional
    :param layer_norm: Whether to use layer normalization. Default is False.
    :type layer_norm: bool, optional
    :param noise_std: Standard deviation of the noise. Default is 0.5.
    :type noise_std: float, optional
    :type critic: bool, optional
    :param init_layers: Whether to initialize the layers. Default is True.
    :type init_layers: bool, optional
    :param output_vanish: Whether the output of the network vanishes. Default is False.
    :type output_vanish: bool, optional
    :param device: Device to use for the network. Default is "cpu".
    :type device: str, optional
    """
    feature_net: Dict[str, SupportedEvolvableTypes]

    arch: str = "composed"

    def __init__(
            self,
            observation_space: TupleOrDictSpace,
            channel_size: List[int],
            kernel_size: List[int],
            stride_size: List[int],
            hidden_size: List[int],
            num_outputs: int,
            latent_dim: int = 16,
            vector_space_mlp: bool = False,
            init_dicts: Dict[str, Dict[str, Any]] = {},
            mlp_output_activation: Optional[str] = None,
            mlp_activation: str = "ReLU",
            cnn_activation: str = "ReLU",
            min_hidden_layers: int = 1,
            max_hidden_layers: int = 3,
            min_mlp_nodes: int = 64,
            max_mlp_nodes: int = 1024,
            min_cnn_hidden_layers: int = 1,
            max_cnn_hidden_layers: int = 6,
            min_channel_size: int = 32,
            max_channel_size: int = 256,
            layer_norm: bool = False,
            noisy: bool = False,
            noise_std: float = 0.5,
            init_layers: bool = True,
            output_vanish: bool = False,
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
        
        self.observation_space = observation_space
        self.vector_space_mlp = vector_space_mlp
        self.channel_size = channel_size
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.hidden_size = hidden_size
        self.num_outputs = num_outputs
        self.mlp_activation = mlp_activation
        self.mlp_output_activation = mlp_output_activation
        self.cnn_activation = cnn_activation
        self.min_hidden_layers = min_hidden_layers
        self.max_hidden_layers = max_hidden_layers
        self.min_mlp_nodes = min_mlp_nodes
        self.max_mlp_nodes = max_mlp_nodes
        self.min_cnn_hidden_layers = min_cnn_hidden_layers
        self.max_cnn_hidden_layers = max_cnn_hidden_layers
        self.min_channel_size = min_channel_size
        self.max_channel_size = max_channel_size
        self.layer_norm = layer_norm
        self.init_layers = init_layers
        self.latent_dim = latent_dim
        self.name = name
        self.noisy = noisy
        self.noise_std = noise_std
        self.output_vanish = output_vanish
        self.init_dicts = init_dicts
        self.vector_spaces = [
            key for key, space in observation_space.spaces.items() if not is_image_space(space)
            ]

        self._net_config = {
            "arch": self.arch,
            "name": self.name,
            "channel_size": self.channel_size,
            "kernel_size": self.kernel_size,
            "stride_size": self.stride_size,
            "cnn_activation": self.cnn_activation,
            "hidden_size": self.hidden_size,
            "mlp_activation": self.mlp_activation,
            "mlp_output_activation": self.mlp_output_activation,
            "min_cnn_hidden_layers": self.min_cnn_hidden_layers,
            "max_cnn_hidden_layers": self.max_cnn_hidden_layers,
            "min_channel_size": self.min_channel_size,
            "max_channel_size": self.max_channel_size,
            "min_hidden_layers": self.min_hidden_layers,
            "max_hidden_layers": self.max_hidden_layers,
            "min_mlp_nodes": self.min_mlp_nodes,
            "max_mlp_nodes": self.max_mlp_nodes,
        }

        self.feature_net = self.build_feature_extractor()
    
        # Collect all vector space shapes for concatenation
        vector_input_dim = sum([spaces.flatdim(self.observation_space.spaces[key]) for key in self.vector_spaces])

        # Calculate total feature dimension for final MLP
        image_features_dim = sum([self.latent_dim for subspace in self.observation_space.spaces.values() if is_image_space(subspace)])
        vector_features_dim = self.latent_dim if self.vector_space_mlp else vector_input_dim
        features_dim = image_features_dim + vector_features_dim

        # Final dense layer to convert feature encodings to desired num_outputs
        self.final_dense = nn.Linear(features_dim, num_outputs)
        self.output_activation = get_activation(mlp_output_activation)

        # If we dont define an EvolvableMLP for vector spaces, we should signal this for Mutations
        if not vector_space_mlp:
            self.filter_mutation_methods("mlp")
    
    @property
    def net_config(self) -> Dict[str, Any]:
        """Returns the configuration of the network.
        
        :return: Network configuration
        :rtype: Dict[str, Any]
        """
        return self._net_config

    @property
    def base_init_dict(self) -> Dict[str, Any]:
        """Returns dictionary of base information.
        
        :return: Base information
        :rtype: Dict[str, Any]
        """
        return {
            "num_outputs": self.latent_dim,
            "layer_norm": self.layer_norm,
            "output_vanish": self.output_vanish,
            "init_layers": self.init_layers,
            "noise_std": self.noise_std,
            "noisy": self.noisy,
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
            "activation": self.mlp_activation,
            "output_activation": self.mlp_activation,
            "min_hidden_layers": self.min_hidden_layers,
            "max_hidden_layers": self.max_hidden_layers,
            "min_mlp_nodes": self.min_mlp_nodes,
            "max_mlp_nodes": self.max_mlp_nodes
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
            "activation": self.cnn_activation,
            "output_activation": self.cnn_activation,
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
        kwargs['num_outputs'] = self.num_outputs
        extra_kwargs = {
            "observation_space": self.observation_space,
            "latent_dim": self.latent_dim,
            "vector_space_mlp": self.vector_space_mlp,
            "init_dicts": self.init_dicts,
            # CNN kwargs
            "channel_size": self.channel_size,
            "kernel_size": self.kernel_size,
            "stride_size": self.stride_size,
            "cnn_activation": self.cnn_activation,
            "min_cnn_hidden_layers": self.min_cnn_hidden_layers,
            "max_cnn_hidden_layers": self.max_cnn_hidden_layers,
            "min_channel_size": self.min_channel_size,
            "max_channel_size": self.max_channel_size,
            # MLP kwargs
            "hidden_size": self.hidden_size,
            "mlp_activation": self.mlp_activation,
            "mlp_output_activation": self.mlp_activation,
            "min_hidden_layers": self.min_hidden_layers,
            "max_hidden_layers": self.max_hidden_layers,
            "min_mlp_nodes": self.min_mlp_nodes,
            "max_mlp_nodes": self.max_mlp_nodes
        }
        kwargs.update(extra_kwargs)
        return kwargs

    
    def get_init_dict(self, key: str, default: Literal['cnn', 'mlp']) -> Dict[str, Any]:
        """Returns the initialization dictionary for the specified key.
        
        Arguments:
            key (str): Key of the observation space.
        
        Returns:
            Dict[str, Any]: Initialization dictionary.
        """
        if key in self.init_dicts:
            return self.init_dicts[key]
        
        assert default in ['cnn', 'mlp'], "Invalid default value provided, must be 'cnn' or 'mlp'."

        return self.cnn_init_dict if default == 'cnn' else self.mlp_init_dict
    
    def extract_init_dict(self, key: str) -> Dict[str, Any]:
        """Extracts and reformats the initialization dictionary to include all keys.
        
        Arguments:
            init_dict (Dict[str, Any]): Initialization dictionary.
        
        Returns:
            Dict[str, Any]: Reformatted initialization dictionary.
        """
        if key in self.feature_net.keys():
            module = self.feature_net[key]
        else:
            raise ValueError(f"Invalid key {key} provided.")

        init_dict = module.init_dict.copy()
        del init_dict["num_outputs"]
        if isinstance(module, EvolvableCNN):
            del init_dict["input_shape"]
        elif isinstance(module, EvolvableMLP):
            del init_dict["num_inputs"]
        
        return init_dict
    
    def build_feature_extractor(self) -> Dict[str, SupportedEvolvableTypes]:
        """Creates the feature extractor and final MLP networks.
        
        Returns:
            Tuple[Dict[str, ModuleType], EvolvableMLP, EvolvableMLP]: Tuple containing the feature extractor,
                value network, and advantage network.
        """
        # Build feature extractors for image spaces only
        feature_net = nn.ModuleDict()
        for key, space in self.observation_space.spaces.items():
            if is_image_space(space):  # Use CNN if it's an image space
                feature_net[key] = EvolvableCNN(
                    input_shape=space.shape,
                    **copy.deepcopy(self.get_init_dict(key, default='cnn'))
                )

        # Collect all vector space shapes for concatenation
        vector_input_dim = sum([spaces.flatdim(self.observation_space.spaces[key]) for key in self.vector_spaces])

        # Optional MLP for all concatenated vector inputs
        if self.vector_space_mlp:
            feature_net['vector_mlp'] = EvolvableMLP(
                num_inputs=vector_input_dim,
                **copy.deepcopy(self.get_init_dict("vector_mlp", default='mlp'))
            )
        
        return feature_net

    def forward(self, x: TupleOrDictObservation, xc: Optional[ArrayOrTensor] = None, q: bool = True) -> torch.Tensor:
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

        # Convert observations to tensors
        for key, obs in x.items():
            x[key] = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        
        # Extract features from image spaces
        image_features = [self.feature_net[key](x[key]) for key in x.keys() if key in self.feature_net.keys()]
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
        return self.output_activation(self.final_dense(features))

    def choose_random_module(self, mtype: Optional[Literal["cnn", "mlp"]] = None) -> Tuple[str, SupportedEvolvableTypes]:
        """Randomly selects a module from the feature extractors or the final MLP.
        
        return: Tuple containing the key and the selected module.
        rtype: Tuple[str, SupportedEvolvableTypes]
        """
        if mtype is not None:
            modules = {}
            for key, module in self.feature_net.items():
                if isinstance(module, EvolvableCNN) and mtype == "cnn":
                    modules[key] = module
                elif isinstance(module, EvolvableMLP) and mtype == "mlp":
                    modules[key] = module
        else:
            modules = self.feature_net

        key = list(modules.keys())[torch.randint(len(modules.keys()), (1,)).item()]

        return key, self.feature_net[key]

    @register_mutation_fn(MutationType.LAYER)
    def add_mlp_layer(self, key: Optional[str] = None) -> Dict[str, str]:
        """Adds a hidden layer from the fully connected layer.
        
        param key: Key specifying the available evolvable module to add the layer to. Default is None.
        type key: str, optional
        rtype: Dict[str, str]
        """
        if key is not None:
            module = self.feature_net[key]
        else:
            key, module = self.choose_random_module("mlp")

        module.add_layer()
        self.init_dicts[key] = self.extract_init_dict(key)
        return {"key": key}

    @register_mutation_fn(MutationType.LAYER)
    def remove_mlp_layer(self, key: Optional[str] = None) -> Dict[str, str]:
        """Removes a hidden layer from the fully connected layer.
        
        param key: Key specifying the available evolvable module to remove the layer from. Default is None.
        type key: str, optional
        rtype: Dict[str, str]
        """
        if key is not None:
            module = self.feature_net[key]
        else:
            key, module = self.choose_random_module("mlp")
            
        module.remove_layer()
        self.init_dicts[key] = self.extract_init_dict(key)
        return {"key": key}

    @register_mutation_fn(MutationType.NODE)
    def add_mlp_node(
            self,
            key: Optional[str] = None,
            hidden_layer: Optional[int] = None,
            numb_new_nodes: Optional[int] = None
            ) -> Dict[str, Union[str, int]]:
        """Adds nodes to the hidden layer of the Multi-layer Perceptron.

        :param key: Key specifying the available evolvable module to add the nodes to, defaults to None
        :type key: str, optional
        :param hidden_layer: Depth of the hidden layer to add nodes to, defaults to None
        :type hidden_layer: int, optional
        :param numb_new_nodes: Number of nodes to add to the hidden layer, defaults to None
        :type numb_new_nodes: int, optional
        :return: Dictionary containing the hidden layer and number of new nodes added
        :rtype: dict
        """
        if key is not None:
            module = self.feature_net[key]
        else:
            key, module = self.choose_random_module("mlp")

        mut_dict = module.add_node(hidden_layer, numb_new_nodes)
        self.init_dicts[key] = self.extract_init_dict(key)
        mut_dict["key"] = key
        return mut_dict

    @register_mutation_fn(MutationType.NODE)
    def remove_mlp_node(
            self,
            key: Optional[str] = None,
            hidden_layer: Optional[int] = None,
            numb_new_nodes: Optional[int] = None
            ) -> Dict[str, Union[str, int]]:
        """Removes nodes from hidden layer of fully connected layer.

        :param key: Key specifying the available evolvable module to remove the nodes from, defaults to None
        :type key: str, optional
        :param hidden_layer: Depth of hidden layer to remove nodes from, defaults to None
        :type hidden_layer: int, optional
        :param numb_new_nodes: Number of nodes to remove from hidden layer, defaults to None
        :type numb_new_nodes: int, optional
        :return: Dictionary containing the hidden layer index and the number of nodes removed
        :rtype: Dict[str, Optional[Union[str, int]]]
        """
        if key is not None:
            module = self.feature_net[key]
        else:
            key, module = self.choose_random_module("mlp")

        mut_dict = module.remove_node(hidden_layer, numb_new_nodes)
        self.init_dicts[key] = self.extract_init_dict(key)
        mut_dict["key"] = key
        return mut_dict

    @register_mutation_fn(MutationType.LAYER)
    def add_cnn_layer(self, key: Optional[str] = None) -> Dict[str, str]:
        """Adds a hidden layer to convolutional neural network.
        
        param key: Key specifying the available evolvable module to add the layer to. Default is None.
        type key: str, optional
        rtype: Dict[str, str]
        """
        if key is not None:
            module = self.feature_net[key]
        else:
            key, module = self.choose_random_module("cnn")

        module.add_layer()
        self.init_dicts[key] = self.extract_init_dict(key)
        return {"key": key}
    
    @register_mutation_fn(MutationType.LAYER)
    def remove_cnn_layer(self, key: Optional[str] = None) -> Dict[str, str]:
        """Removes a hidden layer from convolutional neural network.
        
        param key: Key specifying the available evolvable module to remove the layer from. Default is None.
        type key: str, optional
        rtype: Dict[str, str]
        """
        if key is not None:
            module = self.feature_net[key]
        else:
            key, module = self.choose_random_module("cnn")

        module.remove_layer()
        self.init_dicts[key] = self.extract_init_dict(key)
        return {"key": key}
    
    @register_mutation_fn(MutationType.NODE)
    def change_cnn_kernel(self, key: Optional[str] = None) -> Dict[str, str]:
        """Randomly alters convolution kernel of random CNN layer.
        
        param key: Key specifying the available evolvable module to change the kernel of. Default is None.
        type key: str, optional
        rtype: Dict[str, str]
        """
        if key is not None:
            module = self.feature_net[key]
        else:
            key, module = self.choose_random_module("cnn")

        module.change_kernel()
        self.init_dicts[key] = self.extract_init_dict(key)
        return {"key": key}
    
    @register_mutation_fn(MutationType.NODE)
    def add_cnn_channel(
            self,
            key: Optional[str] = None,
            hidden_layer: Optional[int] = None,
            numb_new_channels: Optional[int] = None
            ) -> Dict[str, Union[str, int]]:
        """Adds channel to hidden layer of convolutional neural networks.

        :param hidden_layer: Depth of hidden layer to add channel to, defaults to None
        :type hidden_layer: int, optional
        :param numb_new_channels: Number of channels to add to hidden layer, defaults to None
        :type numb_new_channels: int, optional
        :return: Dictionary containing the hidden layer and number of new channels added
        :rtype: Dict[str, Union[str, int]]
        """
        if key is not None:
            module = self.feature_net[key]
        else:
            key, module = self.choose_random_module("cnn")

        mut_dict = module.add_channel(hidden_layer, numb_new_channels)
        self.init_dicts[key] = self.extract_init_dict(key)
        mut_dict["key"] = key
        return mut_dict

    @register_mutation_fn(MutationType.NODE)
    def remove_cnn_channel(
            self,
            key: Optional[str] = None,
            hidden_layer: Optional[int] = None,
            numb_new_channels: Optional[int] = None
            ) -> Dict[str, Union[str, int]]:
        """Remove channel from hidden layer of convolutional neural networks.

        :param hidden_layer: Depth of hidden layer to add channel to, defaults to None
        :type hidden_layer: int, optional
        :param numb_new_channels: Number of channels to add to hidden layer, defaults to None
        :type numb_new_channels: int, optional
        :return: Dictionary containing the hidden layer and number of new channels
        :rtype: Dict[str, Optional[Union[str, int]]]
        """
        if key is not None:
            module = self.feature_net[key]
        else:
            key, module = self.choose_random_module("cnn")

        mut_dict = module.remove_channel(hidden_layer, numb_new_channels)
        self.init_dicts[key] = self.extract_init_dict(key)
        mut_dict["key"] = key
        return mut_dict

