from dataclasses import asdict
from typing import Optional

import torch
from gymnasium import spaces

from agilerl.modules.base import EvolvableModule
from agilerl.modules.configs import MlpNetConfig
from agilerl.networks.base import EvolvableNetwork
from agilerl.typing import ConfigType, TorchObsType


class ValueNetwork(EvolvableNetwork):
    """Value functions are used in reinforcement learning to estimate the expected value of a state.
    Therefore, for any given observation, we predict a single scalar value that represents
    the discounted return from that state.

    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
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
    :param device: Device to run the network on.
    :type device: str
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        encoder_config: Optional[ConfigType] = None,
        head_config: Optional[ConfigType] = None,
        min_latent_dim: int = 8,
        max_latent_dim: int = 128,
        n_agents: Optional[int] = None,
        latent_dim: int = 32,
        device: str = "cpu",
    ):

        super().__init__(
            observation_space,
            encoder_config=encoder_config,
            action_space=None,
            min_latent_dim=min_latent_dim,
            max_latent_dim=max_latent_dim,
            n_agents=n_agents,
            latent_dim=latent_dim,
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
        return self.head_net(self.encoder(x))

    def recreate_network(self) -> None:
        """Recreates the network"""
        encoder = self._build_encoder(self.encoder.net_config)
        head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=1,
            name="value",
            net_config=self.head_net.net_config,
        )

        self.encoder = EvolvableModule.preserve_parameters(self.encoder, encoder)
        self.head_net = EvolvableModule.preserve_parameters(self.head_net, head_net)


# TODO: Implement DistributionalValueNetwork
