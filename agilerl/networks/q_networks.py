from dataclasses import asdict
from typing import Any, Dict, Optional, Type, Union

import torch
from gymnasium import spaces

from agilerl.modules.base import EvolvableModule
from agilerl.modules.configs import MlpNetConfig, NetConfig
from agilerl.networks.base import EvolvableNetwork
from agilerl.networks.custom_modules import DuelingDistributionalMLP
from agilerl.typing import ConfigType, TorchObsType
from agilerl.utils.evolvable_networks import is_image_space


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
    :param encoder_cls: Encoder class to use for the network. Defaults to None, whereby it is
        automatically built using an AgileRL module according the observation space.
    :type encoder_cls: Optional[Union[str, Type[EvolvableModule]]]
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
        action_space: spaces.Discrete,
        encoder_cls: Optional[Union[str, Type[EvolvableModule]]] = None,
        encoder_config: Optional[ConfigType] = None,
        head_config: Optional[ConfigType] = None,
        min_latent_dim: int = 8,
        max_latent_dim: int = 128,
        n_agents: Optional[int] = None,
        latent_dim: int = 32,
        simba: bool = False,
        device: str = "cpu",
    ):
        super().__init__(
            observation_space,
            action_space=action_space,
            encoder_cls=encoder_cls,
            encoder_config=encoder_config,
            min_latent_dim=min_latent_dim,
            max_latent_dim=max_latent_dim,
            n_agents=n_agents,
            latent_dim=latent_dim,
            simba=simba,
            device=device,
        )

        if not isinstance(action_space, (spaces.Discrete, spaces.MultiDiscrete)):
            raise ValueError("Action space must be either Discrete or MultiDiscrete")

        if head_config is None:
            head_config = asdict(MlpNetConfig(hidden_size=[16], output_activation=None))

        self.num_actions = spaces.flatdim(action_space)

        # Build value network
        self.build_network_head(head_config)

    def build_network_head(self, net_config: Dict[str, Any]) -> None:
        """Builds the head of the network based on the passed configuration.

        :param net_config: Configuration of the network head.
        :type net_config: Dict[str, Any]
        """
        self.head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=self.num_actions,
            name="value",
            net_config=net_config,
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
        self.recreate_encoder()

        head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=self.num_actions,
            name="value",
            net_config=self.head_net.net_config,
        )

        self.head_net = EvolvableModule.preserve_parameters(self.head_net, head_net)


class RainbowQNetwork(EvolvableNetwork):
    """RainbowQNetwork is an extension of the QNetwork that incorporates the Rainbow DQN improvements
    from "Rainbow: Combining Improvements in Deep Reinforcement Learning" (Hessel et al., 2017).

    Paper: https://arxiv.org/abs/1710.02298

    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
    :param action_space: Action space of the environment
    :type action_space: DiscreteSpace
    :param encoder_cls: Encoder class to use for the network. Defaults to None, whereby it is
        automatically built using an AgileRL module according the observation space.
    :type encoder_cls: Optional[Union[str, Type[EvolvableModule]]]
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
        action_space: spaces.Discrete,
        support: torch.Tensor,
        num_atoms: int = 51,
        noise_std: float = 0.5,
        encoder_config: Optional[ConfigType] = None,
        head_config: Optional[ConfigType] = None,
        min_latent_dim: int = 8,
        max_latent_dim: int = 128,
        n_agents: Optional[int] = None,
        latent_dim: int = 32,
        device: str = "cpu",
    ):

        if isinstance(observation_space, spaces.Box) and not is_image_space(
            observation_space
        ):
            if encoder_config is None:
                encoder_config = asdict(MlpNetConfig(hidden_size=[16]))

            encoder_config["noise_std"] = noise_std
            encoder_config["output_activation"] = encoder_config.get(
                "activation", "ReLU"
            )
            encoder_config["output_vanish"] = False
            encoder_config["init_layers"] = False
            encoder_config["layer_norm"] = True

        super().__init__(
            observation_space,
            encoder_config=encoder_config,
            action_space=action_space,
            min_latent_dim=min_latent_dim,
            max_latent_dim=max_latent_dim,
            n_agents=n_agents,
            latent_dim=latent_dim,
            device=device,
        )

        if not isinstance(action_space, (spaces.Discrete, spaces.MultiDiscrete)):
            raise ValueError("Action space must be either Discrete or MultiDiscrete")

        if head_config is None:
            head_config = asdict(
                MlpNetConfig(
                    hidden_size=[16], output_activation=None, noise_std=noise_std
                )
            )
        elif isinstance(head_config, NetConfig):
            head_config = asdict(head_config)
            head_config["noise_std"] = noise_std

        # The heads should have no output activation
        head_config["output_activation"] = None

        for arg in ["noisy", "init_layers", "layer_norm", "output_vanish"]:
            if head_config.get(arg, None) is not None:
                head_config.pop(arg)

        self.num_actions = spaces.flatdim(action_space)
        self.num_atoms = num_atoms
        self.support = support
        self.noise_std = noise_std

        # Build value and advantage networks
        self.build_network_head(head_config)

    def build_network_head(self, net_config: Dict[str, Any]) -> None:
        """Builds the value and advantage heads of the network based on the passed configuration.

        :param net_config: Configuration of the network head.
        :type net_config: Dict[str, Any]
        """
        self.head_net = DuelingDistributionalMLP(
            num_inputs=self.latent_dim,
            num_outputs=self.num_actions,
            num_atoms=self.num_atoms,
            support=self.support,
            device=self.device,
            **net_config
        )

    def forward(
        self, obs: TorchObsType, q: bool = True, log: bool = False
    ) -> torch.Tensor:
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
        return self.head_net(latent, q=q, log=log)

    def recreate_network(self) -> None:
        """Recreates the network."""
        self.recreate_encoder()

        head_net = DuelingDistributionalMLP(
            num_inputs=self.latent_dim,
            num_outputs=self.num_actions,
            num_atoms=self.num_atoms,
            support=self.support,
            device=self.device,
            **self.head_net.net_config
        )

        self.head_net = EvolvableModule.preserve_parameters(self.head_net, head_net)


class ContinuousQNetwork(EvolvableNetwork):
    """ContinuousQNetwork is an extension of the QNetwork that is used for continuous action spaces.
    This is used in off-policy algorithms like DDPG and TD3. The network predicts the Q value for a
    given state-action pair.

    Paper: https://arxiv.org/abs/1509.02971

    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
    :param action_space: Action space of the environment
    :type action_space: spaces.Box
    :param encoder_cls: Encoder class to use for the network. Defaults to None, whereby it is
        automatically built using an AgileRL module according the observation space.
    :type encoder_cls: Optional[Union[str, Type[EvolvableModule]]]
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
        encoder_cls: Optional[Type[EvolvableModule]] = None,
        encoder_config: Optional[ConfigType] = None,
        head_config: Optional[ConfigType] = None,
        min_latent_dim: int = 8,
        max_latent_dim: int = 128,
        n_agents: Optional[int] = None,
        latent_dim: int = 32,
        simba: bool = False,
        device: str = "cpu",
    ):

        super().__init__(
            observation_space,
            encoder_cls=encoder_cls,
            encoder_config=encoder_config,
            action_space=action_space,
            min_latent_dim=min_latent_dim,
            max_latent_dim=max_latent_dim,
            n_agents=n_agents,
            latent_dim=latent_dim,
            simba=simba,
            device=device,
        )

        if head_config is None:
            head_config = asdict(MlpNetConfig(hidden_size=[16], output_activation=None))

        self.num_actions = spaces.flatdim(action_space)

        # Build value network
        self.build_network_head(head_config)

    def build_network_head(self, net_config: Optional[ConfigType] = None) -> None:
        """Builds the head of the network.

        :param head_config: Configuration of the head.
        :type head_config: Optional[ConfigType]
        """
        self.head_net = self.create_mlp(
            num_inputs=self.latent_dim + self.num_actions,
            num_outputs=1,
            name="value",
            net_config=net_config,
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
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32).to(self.device)

        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)

        x = self.encoder(obs)
        x = torch.cat([x, actions], dim=-1)
        return self.head_net(x)

    def recreate_network(self) -> None:
        """Recreates the network."""
        self.recreate_encoder()

        head_net = self.create_mlp(
            num_inputs=self.latent_dim + self.num_actions,
            num_outputs=1,
            name="value",
            net_config=self.head_net.net_config,
        )

        self.head_net = EvolvableModule.preserve_parameters(self.head_net, head_net)
