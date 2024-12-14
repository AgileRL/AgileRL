from typing import Union, Optional, Tuple, Dict, Any
from dataclasses import asdict
import torch
import torch.nn.functional as F
from gymnasium import spaces

from agilerl.typing import TorchObsType, ConfigType
from agilerl.configs import MlpNetConfig, NetConfig
from agilerl.networks.base import EvolvableNetwork
from agilerl.modules.mlp import EvolvableMLP

DiscreteSpace = Union[spaces.Discrete, spaces.MultiDiscrete]

class QNetwork(EvolvableNetwork):
    """Q Networks correspond to state-action value functions in deep reinforcement learning. From any given 
    state, they predict the value of each action that can be taken from that state.
    
    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
    :param action_space: Action space of the environment
    :type action_space: DiscreteSpace
    :param encoder_config: Configuration of the encoder network.
    :type encoder_config: ConfigType
    :param head_config: Configuration of the network MLP head.
    :type head_config: Optional[ConfigType]
    :param n_agents: Number of agents in the environment. Defaults to None, which corresponds to 
        single-agent environments.
    :type n_agents: Optional[int]
    :param latent_dim: Dimension of the latent space representation.
    :type latent_dim: int
    :param device: Device to use for the network.
    :type device: str
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: DiscreteSpace,
            encoder_config: ConfigType,
            head_config: Optional[ConfigType] = None,
            n_agents: Optional[int] = None,
            latent_dim: int = 32,
            device: str = "cpu"
            ):
        
        super().__init__(
            observation_space, encoder_config, action_space, n_agents, latent_dim, device
            )

        if not isinstance(action_space, (spaces.Discrete, spaces.MultiDiscrete)):
            raise ValueError("Action space must be either Discrete or MultiDiscrete")

        if head_config is None:
            head_config = asdict(
                MlpNetConfig(
                    hidden_size=[64],
                    output_activation=None
                    )
                )

        self.num_actions = spaces.flatdim(action_space)
        self.value_net = self.build_network_head(head_config)

    @property
    def init_dict(self) -> Dict[str, Any]:
        """Initializes the configuration of the Q network.
        
        :return: Configuration of the Q network.
        :rtype: Dict[str, Any]
        """
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "encoder_config": self.encoder.net_config,
            "head_config": self.value_net.net_config,
            "n_agents": self.n_agents,
            "latent_dim": self.latent_dim,
            "device": self.device
            }
    
    def build_network_head(self, net_config: Dict[str, Any]) -> EvolvableMLP:
        """Builds the head of the network based on the passed configuration.
        
        :param net_config: Configuration of the network head.
        :type net_config: Dict[str, Any]
        
        :return: Network head.
        :rtype: EvolvableModule
        """
        return EvolvableMLP(
            num_inputs=self.latent_dim,
            num_outputs=self.num_actions,
            device=self.device,
            name="value",
            **net_config
            )

    def forward(self, x: TorchObsType) -> torch.Tensor:
        """Forward pass of the Q network.

        :param x: Input to the network.
        :type x: TorchObsType

        :return: Output of the network.
        :rtype: torch.Tensor
        """
        latent = self.encoder(x)
        return self.value_net(latent)


class RainbowQNetwork(EvolvableNetwork):
    """RainbowQNetwork is an extension of the QNetwork that incorporates the Rainbow DQN improvements 
    from "Rainbow: Combining Improvements in Deep Reinforcement Learning" (Hessel et al., 2017).
    
    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
    :param action_space: Action space of the environment
    :type action_space: DiscreteSpace
    :param encoder_config: Configuration of the encoder network.
    :type encoder_config: ConfigType
    :param support: Support for the distributional value function.
    :type support: torch.Tensor
    :param num_atoms: Number of atoms in the distributional value function. Defaults to 51.
    :type num_atoms: int
    :param head_config: Configuration of the network MLP head.
    :type head_config: Optional[ConfigType]
    :param n_agents: Number of agents in the environment. Defaults to None, which corresponds to 
        single-agent environments.
    :type n_agents: Optional[int]
    :param latent_dim: Dimension of the latent space representation.
    :type latent_dim: int
    :param device: Device to use for the network.
    :type device: str
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: DiscreteSpace,
            encoder_config: ConfigType,
            support: torch.Tensor,
            num_atoms: int = 51,
            head_config: Optional[ConfigType] = None,
            n_agents: Optional[int] = None,
            latent_dim: int = 32,
            device: str = "cpu",
            ):

        super().__init__(
            observation_space, encoder_config, action_space, n_agents, latent_dim, device
            )

        if not isinstance(action_space, (spaces.Discrete, spaces.MultiDiscrete)):
            raise ValueError("Action space must be either Discrete or MultiDiscrete")

        if head_config is None:
            head_config = asdict(
                MlpNetConfig(
                    hidden_size=[64, 64],
                    noisy=True,
                    output_activation=None
                    )
                )
        elif isinstance(head_config, NetConfig):
            head_config = asdict(head_config)

        self.num_actions = spaces.flatdim(action_space)
        self.num_atoms = num_atoms
        self.support = support

        self.value_net, self.advantage_net = self.build_network_head(head_config)

    @property
    def init_dict(self) -> Dict[str, Any]:
        """Initializes the configuration of the Rainbow Q network.
        
        :return: Configuration of the Rainbow Q network.
        :rtype: Dict[str, Any]
        """
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "support": self.support,
            "num_atoms": self.num_atoms,
            "encoder_config": self.encoder.net_config,
            "head_config": self.value_net.net_config,
            "n_agents": self.n_agents,
            "latent_dim": self.latent_dim,
            "device": self.device
            }
    
    def build_network_head(self, net_config: Dict[str, Any]) -> Tuple[EvolvableMLP, EvolvableMLP]:
        """Builds the head of the network based on the passed configuration.
        
        :param net_config: Configuration of the network head.
        :type net_config: Dict[str, Any]

        :return: Network head.
        :rtype: Tuple[EvolvableModule, EvolvableModule]
        """
        value_net = EvolvableMLP(
            num_inputs=self.latent_dim,
            num_outputs=self.num_atoms,
            device=self.device,
            name="value",
            **net_config
            )
        
        advantage_net = EvolvableMLP(
            num_inputs=self.latent_dim,
            num_outputs=self.num_actions * self.num_atoms,
            device=self.device,
            name="advantage",
            **net_config
            )
        
        return value_net, advantage_net
    
    def forward(self, x: TorchObsType, q: bool = True, log: bool = False) -> torch.Tensor:
        """Forward pass of the Rainbow Q network.

        :param x: Input to the network.
        :type x: TorchObsType
        :param q: Whether to return Q values. Defaults to True.
        :type q: bool
        :param log: Whether to return log probabilities. Defaults to False.
        :type log: bool

        :return: Output of the network.
        :rtype: torch.Tensor
        """
        latent = self.encoder(x)
        value = self.value_net(latent)
        advantage = self.advantage_net(latent)
        
        batch_size = value.size(0)
        value = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)

        x = value + advantage - advantage.mean(1, keepdim=True)
        if log:
            x = F.log_softmax(x.view(-1, self.num_atoms), dim=-1).view(
                -1, self.num_actions, self.num_atoms
            )
            return x
        else:
            x = F.softmax(x.view(-1, self.num_atoms), dim=-1).view(
                -1, self.num_actions, self.num_atoms
            )
            x = x.clamp(min=1e-3)

        if q:
            x = torch.sum(x * self.support, dim=2)

        return x
        


