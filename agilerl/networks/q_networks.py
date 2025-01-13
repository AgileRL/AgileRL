from typing import Union, Optional, Dict, Any
from dataclasses import asdict
import torch
import torch.nn.functional as F
from gymnasium import spaces

from agilerl.typing import TorchObsType, ConfigType
from agilerl.modules.configs import MlpNetConfig, NetConfig
from agilerl.networks.base import EvolvableNetwork
from agilerl.modules.base import EvolvableModule

DiscreteSpace = Union[spaces.Discrete, spaces.MultiDiscrete]

class QNetwork(EvolvableNetwork):
    """Q Networks correspond to state-action value functions in deep reinforcement learning. From any given 
    state, they predict the value of each action that can be taken from that state. By default, we build an 
    encoder that extracts features from an input corresponding to the passed observation space using the 
    AgileRL evolvable modules. The QNetwork then uses an EvolvableMLP as head to predict a value for each 
    possible discrete action for the given state.
    
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
            encoder_config: Optional[ConfigType] = None,
            head_config: Optional[ConfigType] = None,
            min_latent_dim: int = 8,
            max_latent_dim: int = 128,
            n_agents: Optional[int] = None,
            latent_dim: int = 32,
            device: str = "cpu"
            ):

        super().__init__(
            observation_space, 
            encoder_config=encoder_config,
            action_space=action_space,
            min_latent_dim=min_latent_dim, 
            max_latent_dim=max_latent_dim,
            n_agents=n_agents,
            latent_dim=latent_dim,
            device=device
            )

        if not isinstance(action_space, (spaces.Discrete, spaces.MultiDiscrete)):
            raise ValueError("Action space must be either Discrete or MultiDiscrete")

        if head_config is None:
            head_config = asdict(
                MlpNetConfig(
                    hidden_size=[16],
                    output_activation=None
                    )
                )

        self.num_actions = spaces.flatdim(action_space)

        # Build value network
        self.build_network_head(head_config)

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
            "head_config": self.head_net.net_config,
            "min_latent_dim": self.min_latent_dim,
            "max_latent_dim": self.max_latent_dim,
            "n_agents": self.n_agents,
            "latent_dim": self.latent_dim,
            "device": self.device
            }

    def build_network_head(self, net_config: Dict[str, Any]) -> None:
        """Builds the head of the network based on the passed configuration.
        
        :param net_config: Configuration of the network head.
        :type net_config: Dict[str, Any]
        """
        self.head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=self.num_actions,
            name="value",
            net_config=net_config
            )

    def forward(self, obs: TorchObsType) -> torch.Tensor:
        """Forward pass of the Q network.

        :param obs: Input to the network.
        :type obs: TorchObsType

        :return: Output of the network.
        :rtype: torch.Tensor
        """
        latent = self.encoder(obs)
        return self.head_net(latent)
    
    def recreate_network(self) -> None:
        """Recreates the network"""
        encoder = self._build_encoder(self.encoder.net_config)
        head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=self.num_actions,
            name="value",
            net_config=self.head_net.net_config
        )

        self.encoder = EvolvableModule.preserve_parameters(self.encoder, encoder)
        self.head_net = EvolvableModule.preserve_parameters(self.head_net, head_net) 


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
    :param min_latent_dim: Minimum dimension of the latent space representation. Defaults to 8.
    :type min_latent_dim: int
    :param max_latent_dim: Maximum dimension of the latent space representation. Defaults to 128.
    :type max_latent_dim: int
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
            support: torch.Tensor,
            num_atoms: int = 51,
            encoder_config: Optional[ConfigType] = None,
            head_config: Optional[ConfigType] = None,
            min_latent_dim: int = 8,
            max_latent_dim: int = 128,
            n_agents: Optional[int] = None,
            latent_dim: int = 32,
            device: str = "cpu"
            ):

        super().__init__(
            observation_space, 
            encoder_config=encoder_config,
            action_space=action_space,
            min_latent_dim=min_latent_dim, 
            max_latent_dim=max_latent_dim,
            n_agents=n_agents,
            latent_dim=latent_dim,
            device=device
            )

        if not isinstance(action_space, (spaces.Discrete, spaces.MultiDiscrete)):
            raise ValueError("Action space must be either Discrete or MultiDiscrete")

        if head_config is None:
            head_config = asdict(
                MlpNetConfig(
                    hidden_size=[16],
                    noisy=True,
                    output_activation=None
                    )
                )
        elif isinstance(head_config, NetConfig):
            head_config = asdict(head_config)

        self.num_actions = spaces.flatdim(action_space)
        self.num_atoms = num_atoms
        self.support = support

        # Build value and advantage networks
        self.build_network_head(head_config)

        # Register mutation hook
        self.register_mutation_hook(self._recreate_advantage_from_value)

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
            "head_config": self.head_net.net_config,
            "min_latent_dim": self.min_latent_dim,
            "max_latent_dim": self.max_latent_dim,
            "n_agents": self.n_agents,
            "latent_dim": self.latent_dim,
            "device": self.device
            }
    
    def build_network_head(self, net_config: Dict[str, Any]) -> None:
        """Builds the value and advantage heads of the network based on the passed configuration.
        
        :param net_config: Configuration of the network head.
        :type net_config: Dict[str, Any]
        """
        self.head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=self.num_atoms,
            name='value',
            net_config=net_config
            )
        
        self.advantage_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=self.num_actions * self.num_atoms,
            name='advantage',
            net_config=net_config
            )
        
        # We want the same mutations for both the value and advantage networks
        self.advantage_net.disable_mutations()
    
    def forward(self, obs: TorchObsType, q: bool = True, log: bool = False) -> torch.Tensor:
        """Forward pass of the Rainbow Q network.

        :param obs: Input to the network.
        :type obs: torch.Tensor, dict[str, torch.Tensor], or list[torch.Tensor]
        :param q: Whether to return Q values. Defaults to True.
        :type q: bool
        :param log: Whether to return log probabilities. Defaults to False.
        :type log: bool

        :return: Output of the network.
        :rtype: torch.Tensor
        """
        latent = self.encoder(obs)
        value: torch.Tensor = self.head_net(latent)
        advantage: torch.Tensor = self.advantage_net(latent)
        
        batch_size = value.size(0)
        value = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)

        x = value + advantage - advantage.mean(1, keepdim=True)
        if log:
            x = F.log_softmax(x.view(-1, self.num_atoms), dim=-1)
            return x.view(-1, self.num_actions, self.num_atoms)

        x = F.softmax(x.view(-1, self.num_atoms), dim=-1)
        x = x.view(-1, self.num_actions, self.num_atoms).clamp(min=1e-3)
        if q:
            x = torch.sum(x * self.support, dim=2)

        return x
    
    def _recreate_advantage_from_value(self) -> None:
        """Recreates the advantage network using the net_config of the value network 
        after a mutation on the value network."""
        if self.last_mutation_attr.split(".")[0] == "head_net":
            advantage_net = self.create_mlp(
                num_inputs=self.latent_dim,
                num_outputs=self.num_actions * self.num_atoms,
                name="advantage",
                net_config=self.head_net.net_config
            )
            self.advantage_net = EvolvableModule.preserve_parameters(self.advantage_net, advantage_net)

    def recreate_network(self) -> None:
        """Recreates the network"""
        encoder = self._build_encoder(self.encoder.net_config)
        head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=self.num_actions,
            name="value",
            net_config=self.head_net.net_config
        )

        advantage_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=self.num_actions * self.num_atoms,
            name="advantage",
            net_config=self.head_net.net_config
        )

        self.encoder = EvolvableModule.preserve_parameters(self.encoder, encoder)
        self.head_net = EvolvableModule.preserve_parameters(self.head_net, head_net) 
        self.advantage_net = EvolvableModule.preserve_parameters(self.advantage_net, advantage_net)

class ContinuousQNetwork(EvolvableNetwork):
    """ContinuousQNetwork is an extension of the QNetwork that is used for continuous action spaces.
    This is used in off-policy algorithms like DDPG and TD3. The network predicts the Q value for a
    given state-action pair.

    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
    :param action_space: Action space of the environment
    :type action_space: spaces.Box
    :param encoder_config: Configuration of the encoder network.
    :type encoder_config: ConfigType
    :param head_config: Configuration of the network MLP head.
    :type head_config: Optional[ConfigType]
    :param min_latent_dim: Minimum dimension of the latent space representation. Defaults to 8.
    :type min_latent_dim: int
    :param max_latent_dim: Maximum dimension of the latent space representation. Defaults to 128.
    :type max_latent_dim: int
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
            action_space: spaces.Box,
            encoder_config: Optional[ConfigType] = None,
            head_config: Optional[ConfigType] = None,
            min_latent_dim: int = 8,
            max_latent_dim: int = 128,
            n_agents: Optional[int] = None,
            latent_dim: int = 32,
            device: str = "cpu"
            ):

        super().__init__(
            observation_space, 
            encoder_config=encoder_config,
            action_space=action_space,
            min_latent_dim=min_latent_dim, 
            max_latent_dim=max_latent_dim,
            n_agents=n_agents,
            latent_dim=latent_dim,
            device=device
            )

        if head_config is None:
            head_config = asdict(
                MlpNetConfig(
                    hidden_size=[16],
                    output_activation=None
                    )
                )
            
        self.num_actions = spaces.flatdim(action_space)

        # Build value network
        self.build_network_head(head_config)

    @property
    def init_dict(self) -> Dict[str, Any]:
        """Initializes the configuration of the Rainbow Q network.
        
        :return: Configuration of the Rainbow Q network.
        :rtype: Dict[str, Any]
        """
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "encoder_config": self.encoder.net_config,
            "head_config": self.head_net.net_config,
            "min_latent_dim": self.min_latent_dim,
            "max_latent_dim": self.max_latent_dim,
            "n_agents": self.n_agents,
            "latent_dim": self.latent_dim,
            "device": self.device
            }
    
    
    def build_network_head(self, net_config: Optional[ConfigType] = None) -> None:
        """Builds the head of the network.

        :param head_config: Configuration of the head.
        :type head_config: Optional[ConfigType]
        """
        self.head_net = self.create_mlp(
            num_inputs=self.latent_dim + self.num_actions,
            num_outputs=1,
            name="value",
            net_config=net_config
        )

    def forward(self, obs: TorchObsType, actions: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        :param obs: Input tensor.
        :type obs: torch.Tensor, dict[str, torch.Tensor], or list[torch.Tensor]
        :param actions: Actions tensor.
        :type actions: torch.Tensor
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        x = self.encoder(obs)
        x = torch.cat([x, actions], dim=-1)
        return self.head_net(x)

    def recreate_network(self) -> None:
        """Recreates the network"""
        encoder = self._build_encoder(self.encoder.net_config)
        head_net = self.create_mlp(
            num_inputs=self.latent_dim + self.num_actions,
            num_outputs=1,
            name="value",
            net_config=self.head_net.net_config
        )

        self.encoder = EvolvableModule.preserve_parameters(self.encoder, encoder)
        self.head_net = EvolvableModule.preserve_parameters(self.head_net, head_net) 

