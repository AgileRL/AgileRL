from typing import Optional, Union, TypeVar, Dict, Any
from dataclasses import asdict
import copy
from abc import ABC, abstractmethod
from gymnasium import spaces
import numpy as np 
import torch

from agilerl.protocols import MutationType
from agilerl.typing import DeviceType, TorchObsType, ConfigType
from agilerl.modules.base import EvolvableModule, ModuleMeta, register_mutation_fn
from agilerl.modules.cnn import EvolvableCNN
from agilerl.modules.multi_input import EvolvableMultiInput
from agilerl.modules.mlp import EvolvableMLP
from agilerl.utils.evolvable_networks import is_image_space, get_default_encoder_config

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

class NetworkError(Exception):
    """Exception raised for errors in the network."""
    pass

class NetworkMeta(ModuleMeta):
    """Metaclass for evolvable networks. Checks that the network has 
    an encoder and a head_net (named as such)."""

    def __call__(cls, *args, **kwargs):
        instance: SelfEvolvableNetwork = super().__call__(*args, **kwargs)

        # Check that the mutation methods of the network are correctly defined
        # i.e. only contain underlying methods corresponding to the encoder and head_net
        for mut_method in instance.mutation_methods:
            if "." in mut_method:
                attr =  mut_method.split(".")[0]
                if attr not in ['encoder', 'head_net']:
                    raise NetworkError(
                        "Mutation methods in EvolvableNetwork's should only correspond to encoder or head_net."
                        )

        return instance

class EvolvableNetwork(EvolvableModule, ABC, metaclass=NetworkMeta):
    """
    Base class for evolvable networks, i.e., evolvable modules that are configured in 
    a specific way for a reinforcement learning algorithm, similar to how CNNs are used 
    as building blocks in ResNet, VGG, etc. An evolvable network automatically inspects the passed 
    observation space to determine the appropriate encoder to build through the AgileRL 
    evolvable modules, inheriting the mutation methods of any underlying evolvable module.

    .. note::
        Currently, evolvable networks should only have the encoder (which is automatically 
        built from the observation space) and a 'head_net' attribute that processes the latent 
        encodings into the desired number of outputs as evolvable components. For example, in RainbowQNetwork, 
        we signal the advantage net as unevolvable and apply the same mutations to it as the 'value' 
        net, which is the network head in this case. Users should follow the same philosophy.

    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
    :param encoder_config: Configuration of the encoder. Defaults to None.
    :type encoder_config: Optional[ConfigType]
    :param action_space: Action space of the environment. Defaults to None.
    :type action_space: Optional[spaces.Space]
    :param min_latent_dim: Minimum dimension of the latent space representation. Defaults to 8.
    :type min_latent_dim: int
    :param max_latent_dim: Maximum dimension of the latent space representation. Defaults to 128.
    :type max_latent_dim: int
    :param n_agents: Number of agents in the environment. Defaults to None, which corresponds to 
        single-agent environments.
    :type n_agents: Optional[int]
    :param encoder_mutations: If True, allow mutations to the encoder. Defaults to False.
    :type encoder_mutations: bool
    :param latent_dim: Dimension of the latent space representation. Defaults to 32.
    :type latent_dim: int
    :param device: Device to use for the network. Defaults to "cpu".
    :type device: DeviceType
    """
    def __init__(
            self,
            observation_space: spaces.Space,
            encoder_config: Optional[ConfigType] = None,
            action_space: Optional[spaces.Space] = None,
            min_latent_dim: int = 8,
            max_latent_dim: int = 128,
            n_agents: Optional[int] = None,
            latent_dim: int = 32,
            device: DeviceType = "cpu"
            ) -> None:
        super().__init__(device)

        if encoder_config is None:
            encoder_config = get_default_encoder_config(observation_space)

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
        self.min_latent_dim = min_latent_dim
        self.max_latent_dim = max_latent_dim
        self.device = device
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

        # NOTE: We disable layer mutations for the encoder since this usually incurs 
        # a lot of variance in the optimization process and makes learning unstable
        self.encoder.disable_mutations(MutationType.LAYER)
    
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
    
    @property
    def activation(self) -> str:
        """Activation function of the network.
        
        :return: Activation function.
        :rtype: str
        """
        return self.encoder.activation

    @abstractmethod
    def forward(self, x: TorchObsType) -> torch.Tensor:
        """Forward pass of the network.
        
        :param x: Input to the network.
        :type x: TorchObsType

        :return: Output of the network.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

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
        if isinstance(observation_space, (spaces.Dict, spaces.Tuple)):
            net_config['cnn_block_type'] = "Conv3d"
        else:
            net_config['block_type'] = "Conv3d"

        return net_config

    def modules(self) -> Dict[str, SupportedEvolvable]:
        """Modules of the network.
        
        :return: Modules of the network.
        :rtype: Dict[str, EvolvableModule]
        """
        return super().modules()
    
    def init_weights_gaussian(self, std_coeff: float = 4.0, output_coeff: float = 2.0) -> None:
        """Initialize the weights of the network with a Gaussian distribution.
        
        :param std_coeff: Coefficient for the standard deviation of the Gaussian distribution, defaults to 4.0
        :type std_coeff: float, optional
        :param output_coeff: Coefficient for the standard deviation of the Gaussian distribution for the output layer, defaults to 2.0
        :type output_coeff: float, optional
        """
        # Initialize the weights of the encoder
        self.encoder.init_weights_gaussian(std_coeff=std_coeff)

        # Initialize weights of network heads
        # NOTE: We assume the head is an instance of EvolvableMLP
        for attr, module in self.modules().items():
            if attr != "encoder":
                module.init_weights_gaussian(std_coeff=std_coeff, output_coeff=output_coeff)
    
    def change_activation(self, activation: str, output: bool = False) -> None:
        """Change the activation function for the network.

        :param activation: Activation function to use.
        :type activation: str
        :param output: If True, change the output activation function, defaults to False
        :type output: bool, optional
        """
        for attr, module in self.modules().items():
            _output = False if attr == "encoder" else output
            module.change_activation(activation, output=_output)

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

    def clone(self) -> SelfEvolvableNetwork:
        """Clone the network.
        
        :return: Cloned network.
        :rtype: SelfEvolvableNetwork
        """
        clone = self.__class__(**copy.deepcopy(self.init_dict))

        # Load state dicts of underlying evolvable modules
        for attr, module in self.modules().items():
            clone_module: EvolvableModule = getattr(clone, attr)

            # NOTE: Sometimes e.g. target networks have empty state dicts (when detached)
            if module.state_dict():
                clone_module.load_state_dict(module.state_dict())
            
        return clone
    
    def recreate_network(self, shrink_params: bool = False) -> None:
        """Recreate the encoder network.
        
        :param shrink_params: If True, shrink the parameters of the network, defaults to False
        :type shrink_params: bool, optional
        """
        encoder = self._build_encoder(self.encoder_config)
        preserve_params_fn = (
            EvolvableModule.shrink_preserve_parameters if shrink_params 
            else EvolvableModule.preserve_parameters
        )
        self.encoder = preserve_params_fn(self.encoder, encoder)
