from typing import Optional, Type, Dict, Any
from abc import ABC, abstractmethod
from gymnasium import spaces
import torch

from agilerl.typing import DeviceType, TorchObsType
from agilerl.modules.base import EvolvableModule
from agilerl.modules.cnn import EvolvableCNN
from agilerl.modules.multi_input import EvolvableMultiInput
from agilerl.modules.mlp import EvolvableMLP
from agilerl.utils.evolvable_networks import is_image_space

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
        "hidden_size",
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


class EvolvableNetwork(EvolvableModule, ABC):
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
            action_space: spaces.Space,
            encoder_config: Dict[str, Any],
            n_agents: Optional[int] = None,
            latent_dim: int = 32,
            device: DeviceType = "cpu",
            ) -> None:
        super().__init__()
        assert isinstance(encoder_config, dict), "Net config must be a dictionary."

        self.observation_space = observation_space
        self.action_space = action_space
        self.n_agents = n_agents
        self.latent_dim = latent_dim
        self.device = device

        # Encoder processes an observation into a latent vector representation
        self.encoder = self._build_encoder(encoder_config)
    
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
    def build_network_head(self, net_config: Dict[str, Any]) -> EvolvableModule:
        """Builds the head of the network based on the passed configuration.
        
        :param net_config: Configuration of the network head.
        :type net_config: Dict[str, Any]
        
        :return: Network head.
        :rtype: EvolvableModule
        """
        raise NotImplementedError

    @staticmethod
    def modify_multi_agent_config(
            net_config: Dict[str, Any],
            n_agents: Optional[int] = None,
            n_agents_depth: bool = False
            ) -> Dict[str, Any]:
        """In multi-agent settings, it is not clear what the shape of the input to the 
        encoder is based on the passed observation space. If kernel sizes are passed as 
        integers, we add a depth dimension of 1 for all layers. Note that for e.g. value 
        functions the first layer should have a depth corresponding to the number of agents 
        to receive a single output rather than `self.n_agents`
        """
        kernel_sizes = net_config['kernel_size']
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
            # If n_agents_depth is True, we add a depth dimension corresponding 
            # to the number of agents for the first layer of the network
            if n_agents_depth:
                assert n_agents is not None, "Number of agents must be passed."
                net_config['kernel_size'][0] = (
                    n_agents, kernel_sizes[0], kernel_sizes[0]
                )
        
        return net_config

    def _build_encoder(self, net_config: Dict[str, Any]) -> EvolvableModule:
        """Builds the encoder for the network based on the environment's observation space.
        
        :return: Encoder module.
        :rtype: EvolvableModule
        """
        if isinstance(self.observation_space, (spaces.Dict, spaces.Tuple)):
            assert_correct_multi_input_net_config(net_config)

            if self.n_agents is not None:
                net_config = EvolvableNetwork.modify_multi_agent_config(net_config)

            encoder = EvolvableMultiInput(
                observation_space=self.observation_space,
                num_outputs=self.latent_dim,
                device=self.device,
                name="encoder",
                **net_config
            )
        elif is_image_space(self.observation_space):
            assert_correct_cnn_net_config(net_config)

            if self.n_agents is not None:
                net_config = EvolvableNetwork.modify_multi_agent_config(net_config)

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
                input_size=spaces.flatdim(self.observation_space),
                output_size=self.latent_dim,
                device=self.device,
                name="encoder",
                **net_config
            )

        return encoder
        
    
