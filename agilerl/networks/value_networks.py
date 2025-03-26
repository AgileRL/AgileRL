from dataclasses import asdict
from typing import Optional, Type, Union

import torch
from gymnasium import spaces

from agilerl.modules import EvolvableModule
from agilerl.modules.configs import MlpNetConfig
from agilerl.networks.base import EvolvableNetwork
from agilerl.typing import ConfigType, TorchObsType


class ValueNetwork(EvolvableNetwork):
    """Value functions are used in reinforcement learning to estimate the expected value of a state.
    For any given observation, we predict a single scalar value that represents
    the discounted return from that state. Used in e.g. PPO.

    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
    :param encoder_cls: Encoder class to use for the network. Defaults to None, whereby it is
        automatically built using an AgileRL module according the observation space.
    :type encoder_cls: Optional[Union[str, Type[EvolvableModule]]]
    :param encoder_config: Configuration of the encoder.
    :type encoder_config: ConfigType
    :param head_config: Configuration of the head.
    :type head_config: Optional[ConfigType]
    :param min_latent_dim: Minimum latent dimension.
    :type min_latent_dim: int
    :param max_latent_dim: Maximum latent dimension.
    :type max_latent_dim: int
    :param n_agents: Number of agents.
    :type n_agents: Optional[int]
    :param latent_dim: Latent dimension.
    :type latent_dim: int
    :param simba: Whether to use the SimBa architecture for training the network.
    :type simba: bool
    :param recurrent: Whether to use a recurrent network.
    :type recurrent: bool
    :param device: Device to run the network on.
    :type device: str
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        encoder_cls: Optional[Union[str, Type[EvolvableModule]]] = None,
        encoder_config: Optional[ConfigType] = None,
        head_config: Optional[ConfigType] = None,
        min_latent_dim: int = 8,
        max_latent_dim: int = 128,
        n_agents: Optional[int] = None,
        latent_dim: int = 32,
        simba: bool = False,
        recurrent: bool = False,
        device: str = "cpu",
    ):

        super().__init__(
            observation_space,
            encoder_cls=encoder_cls,
            encoder_config=encoder_config,
            action_space=None,
            min_latent_dim=min_latent_dim,
            max_latent_dim=max_latent_dim,
            n_agents=n_agents,
            latent_dim=latent_dim,
            simba=simba,
            recurrent=recurrent,
            device=device,
        )

        if head_config is None:
            head_config = asdict(MlpNetConfig(hidden_size=[16], output_activation=None))

        # Build the network head
        self.build_network_head(head_config)

    def get_output_dense(self) -> torch.nn.Linear:
        """Returns the output dense layer of the network.

        :return: Output dense layer.
        :rtype: torch.nn.Linear
        """
        return self.head_net.get_output_dense()

    def build_network_head(self, net_config: Optional[ConfigType] = None) -> None:
        """Builds the head of the network.

        :param net_config: Configuration of the head.
        :type net_config: Optional[ConfigType]
        """
        self.head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=1,
            name="value",
            net_config=net_config,
        )

    def forward(self, x: TorchObsType) -> torch.Tensor:
        """Forward pass of the network.

        :param x: Input tensor.
        :type x: torch.Tensor, dict[str, torch.Tensor], or list[torch.Tensor]
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        latent = self.extract_features(x)
        return self.head_net(latent)

    def recreate_network(self) -> None:
        """Recreates the network."""
        self.recreate_encoder()

        head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=1,
            name="value",
            net_config=self.head_net.net_config,
        )

        self.head_net = EvolvableModule.preserve_parameters(self.head_net, head_net)
