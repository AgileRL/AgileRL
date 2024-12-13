from typing import Optional, Union, Iterable, TypeVar, Dict, Any
from dataclasses import asdict
import copy
from abc import ABC, abstractmethod, ABCMeta
from gymnasium import spaces
import numpy as np 
import torch

from agilerl.protocols import MutationType, MutationMethod
from agilerl.typing import DeviceType, TorchObsType, ConfigType
from agilerl.modules.base import EvolvableModule
from agilerl.modules.cnn import EvolvableCNN
from agilerl.modules.multi_input import EvolvableMultiInput
from agilerl.modules.mlp import EvolvableMLP
from agilerl.utils.evolvable_networks import is_image_space
from agilerl.utils.algo_utils import recursive_check_module_attrs

SelfEvolvableNetwork = TypeVar("SelfEvolvableNetwork", bound="EvolvableNetwork")
SupportedEvolvable = Union[EvolvableMLP, EvolvableCNN, EvolvableMultiInput]

def assert_correct_mlp_net_config(net_config: Dict[str, Any]) -> None:
    """Asserts that the MLP network configuration is correct.
    
    :param net_config: Configuration of the MLP network.
    :type net_config: Dict[str, Any]
    """
    assert (
        "hidden_size" in net_config.keys()
    ), "Net config must contain hidden_size: int."
    assert isinstance(
        net_config["hidden_size"], list
    ), "Net config hidden_size must be a list."
    assert (
        len(net_config["hidden_size"]) > 0
    ), "Net config hidden_size must contain at least one element."

def assert_correct_cnn_net_config(net_config: Dict[str, Any]) -> None:
    """Asserts that the CNN network configuration is correct.
    
    :param net_config: Configuration of the CNN network.
    :type net_config: Dict[str, Any]
    """
    for key in [
        "channel_size",
        "kernel_size",
        "stride_size",
    ]:
        assert (
            key in net_config.keys()
        ), f"Net config must contain {key}: int."
        assert isinstance(
            net_config[key], list
        ), f"Net config {key} must be a list."
        assert (
            len(net_config[key]) > 0
        ), f"Net config {key} must contain at least one element."

        if key == "kernel_size":
            assert (
                isinstance(net_config[key], (int, tuple, list)), 
                "Kernel size must be of type int, list, or tuple."
            )
    
def assert_correct_multi_input_net_config(net_config: Dict[str, Any]) -> None:
    """Asserts that the MultiInput network configuration is correct.
    
    :param net_config: Configuration of the MultiInput network.
    :type net_config: Dict[str, Any]
    """
    # Multi-input networks contain at least one image space
    assert_correct_cnn_net_config(net_config)
    assert (
        "latent_dim" in net_config.keys()
    ), "Net config must contain latent_dim: int."


class _NetworkMeta(type):
    """Metaclass to wrap registry information after algorithm is done 
    intiializing with specified network groups and optimizers."""
    def __call__(cls, *args, **kwargs):
        # Create the instance
        instance: SelfEvolvableNetwork = super().__call__(*args, **kwargs)

        # Call the base class post_init_hook after all initialization
        if isinstance(instance, cls) and hasattr(instance, "_parse_mutation_methods"):
            instance._parse_mutation_methods()

        return instance

class NetworkMeta(_NetworkMeta, ABCMeta):
    ...

class EvolvableNetwork(EvolvableModule, ABC, metaclass=NetworkMeta):
    """Base class for evolvable networks i.e. evolvable modules that are configured in 
    a specific way for a reinforcement learning algorithm - analogously to how CNNs are used 
    as building blocks in ResNet, VGG, etc. Specifically, we automatically inspect the passed 
    observation space to determine the appropriate encoder to build through the AgileRL 
    evolvable modules.
    
    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
    :param action_space: Action space of the environment
    :type action_space: spaces.Space
    :param net_config: Configuration of the network.
    :type net_config: Dict[str, Any]
    :param n_agents: Number of agents in the environment. Defaults to None, which corresponds to 
        single-agent environments.
    :type n_agents: Optional[int]
    :param latent_dim: Dimension of the latent space representation.
    :type latent_dim: int
    :param device: Device to use for the network.
    :type device: DeviceType
    """
    def __init__(
            self,
            observation_space: spaces.Space,
            encoder_config: ConfigType,
            action_space: Optional[spaces.Space] = None,
            n_agents: Optional[int] = None,
            latent_dim: int = 32,
            device: DeviceType = "cpu"
            ) -> None:
        super().__init__(device)

        # For multi-agent settings, we use a depth corresponding to that of the 
        # sample input for the kernel of the first layer of CNN-based networks
        if n_agents is not None and "kernel_size" in encoder_config.keys():
            encoder_config = EvolvableNetwork.modify_multi_agent_config(
                net_config=encoder_config,
                observation_space=observation_space
                )

        self.observation_space = observation_space
        self.action_space = action_space
        self.n_agents = n_agents
        self.latent_dim = latent_dim
        self.encoder_config = (
            encoder_config if isinstance(encoder_config, dict) 
            else asdict(encoder_config)
        )

        # Encoder processes an observation into a latent vector representation
        output_activation = self.encoder_config.get("output_activation", None)
        if output_activation is None:
            activation = self.encoder_config.get("activation", "ReLU")
            encoder_config['output_activation'] = activation

        self.encoder = self._build_encoder(encoder_config)
    
    @property
    def init_dict(self) -> Dict[str, Any]:
        """Initial dictionary for the network.
        
        :return: Initial dictionary for the network.
        :rtype: Dict[str, Any]
        """
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "encoder_config": self.encoder.net_config,
            "n_agents": self.n_agents,
            "latent_dim": self.latent_dim,
            "device": self.device
        }
    
    @abstractmethod
    def forward(self, x: TorchObsType) -> torch.Tensor:
        """Forward pass of the network.
        
        :param x: Input to the network.
        :type x: TorchObsType

        :return: Output of the network.
        :rtype: torch.Tensor
        """
        raise NotImplementedError
    
    @abstractmethod
    def build_network_head(self, net_config: Dict[str, Any]) -> Union[EvolvableModule, Iterable[EvolvableModule]]:
        """Builds the head of the network based on the passed configuration.
        
        :param net_config: Configuration of the network head.
        :type net_config: Dict[str, Any]
        
        :return: Network head.
        :rtype: EvolvableModule
        """
        raise NotImplementedError
    
    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            mut_method = self.get_mutation_methods().get(name)
            if mut_method is not None:
                return mut_method
            raise e

    @staticmethod
    def modify_multi_agent_config(
            net_config: Dict[str, Any],
            observation_space: spaces.Space,
            ) -> Dict[str, Any]:
        """In multi-agent settings, it is not clear what the shape of the input to the 
        encoder is based on the passed observation space. If kernel sizes are passed as 
        integers, we add a depth dimension of 1 for all layers. Note that for e.g. value 
        functions the first layer should have a depth corresponding to the number of agents 
        to receive a single output rather than `self.n_agents`
        """
        kernel_sizes = net_config['kernel_size']
        if isinstance(observation_space, (spaces.Dict, spaces.Tuple)):
            net_config['cnn_block_type'] = "Conv3d"
        else:
            net_config['block_type'] = "Conv3d"

        assert (
            "sample_input" in net_config.keys(), 
            "A sample input must be passed for multi-agent CNN-based networks."
        )

        # NOTE: If kernel sizes are passed as integers, we add a depth dimension of 
        # 1 for all layers. Note that for e.g. value functions or Q networks 
        # it is common for the first layer to have a depth corresponding to 
        # the number of agents 
        if isinstance(kernel_sizes[0], int):
            net_config['kernel_size'] = [
                (1, k_size, k_size) for k_size in kernel_sizes
                ]
            # We infer the depth of the first kernel from the shape of the sample input tensor
            sample_input: torch.Tensor = net_config['sample_input']
            net_config['kernel_size'][0] = (
                sample_input.size(2), kernel_sizes[0], kernel_sizes[0]
            )
    
        return net_config
    
    def evolvable_modules(self) -> Dict[str, EvolvableModule]:
        """Returns the attributes related to the evolvable networks in the algorithm. Includes 
        attributes that are either evolvable networks or a list of evolvable networks, as well 
        as the optimizers associated with the networks.

        :param networks_only: If True, only include evolvable networks, defaults to False
        :type networks_only: bool, optionals
        
        :return: A dictionary of network attributes.
        :rtype: dict[str, Any]
        """
        def is_evolvable(attr: str, obj: Any):
            return (
                recursive_check_module_attrs(obj, networks_only=True)
                and not attr.startswith("_") and not attr.endswith("_")
            )
        # Inspect evolvable
        evolvable_attrs = {}
        for attr in dir(self):
            obj = getattr(self, attr)
            if is_evolvable(attr, obj):
                evolvable_attrs[attr] = obj

        return evolvable_attrs

    
    def _parse_mutation_methods(self) -> None:
        """Parse the mutation methods for the network. Here we form a mapping 
        between mutation methods and the evolvable modules they belong to."""
        self._mutation_methods = []
        self._layer_mutation_methods = []
        self._node_mutation_methods = []
        for attr, module in self.evolvable_modules().items():
            for name, method in module.get_mutation_methods().items():
                method_name = ".".join([attr, name])
                self._mutation_methods.append(method_name)

                method_type = method._mutation_type
                if method_type == MutationType.LAYER:
                    self._layer_mutation_methods.append(method_name)
                elif method_type == MutationType.NODE:
                    self._node_mutation_methods.append(method_name)
                else:
                    raise ValueError(f"Invalid mutation type: {method_type}")
    
    def _build_encoder(self, net_config: Dict[str, Any]) -> SupportedEvolvable:
        """Builds the encoder for the network based on the environment's observation space.
        
        :return: Encoder module.
        :rtype: EvolvableModule
        """
        if isinstance(self.observation_space, (spaces.Dict, spaces.Tuple)):
            assert_correct_multi_input_net_config(net_config)

            encoder = EvolvableMultiInput(
                observation_space=self.observation_space,
                num_outputs=self.latent_dim,
                device=self.device,
                name="encoder",
                **net_config
            )
        elif is_image_space(self.observation_space):
            assert_correct_cnn_net_config(net_config)

            encoder = EvolvableCNN(
                input_shape=self.observation_space.shape,
                num_outputs=self.latent_dim,
                device=self.device,
                name="encoder",
                **net_config
            )
        else:
            assert_correct_mlp_net_config(net_config)

            encoder = EvolvableMLP(
                num_inputs=spaces.flatdim(self.observation_space),
                num_outputs=self.latent_dim,
                device=self.device,
                name="encoder",
                **net_config
            )

        return encoder
    
    def get_mutation_methods(self) -> Dict[str, MutationMethod]:
        """Get all mutation methods for the network.

        :return: A dictionary of mutation methods.
        :rtype: Dict[str, MutationMethod]
        """
        def get_method_from_name(name: str) -> MutationMethod:
            attr, method = name.split(".")
            return getattr(getattr(self, attr), method)

        return {name: get_method_from_name(name) for name in self._mutation_methods}
    
    def clone(self) -> SelfEvolvableNetwork:
        """Clone the network.
        
        :return: Cloned network.
        :rtype: SelfEvolvableNetwork
        """
        clone = self.__class__(**copy.deepcopy(self.init_dict))

        # Load state dicts of underlying evolvable modules
        for attr, module in self.evolvable_modules().items():
            clone_module: EvolvableModule = getattr(clone, attr)
            if module.state_dict():
                clone_module.load_state_dict(module.state_dict())
            
        return clone
    
