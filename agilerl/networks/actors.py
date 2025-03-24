from typing import Optional, Type, Union

import numpy as np
import torch
from gymnasium import spaces
from torch.distributions import Bernoulli, Categorical, Distribution, Normal

from agilerl.modules.base import EvolvableModule, EvolvableWrapper
from agilerl.modules.configs import MlpNetConfig
from agilerl.networks.base import EvolvableNetwork
from agilerl.typing import ArrayOrTensor, ConfigType, DeviceType, TorchObsType


class EvolvableDistribution(EvolvableWrapper):
    """Wrapper to output a distribution over an action space for an evolvable module. It provides methods
    to sample actions and compute log probabilities, relevant for many policy-gradient algorithms such as
    PPO, A2C, TRPO.

    :param action_space: Action space of the environment.
    :type action_space: spaces.Space
    :param network: Network that outputs the logits of the distribution.
    :type network: EvolvableModule
    """

    wrapped: EvolvableModule

    def __init__(
        self,
        action_space: spaces.Space,
        network: EvolvableModule,
        log_std_init: float = 0.0,
        device: DeviceType = "cpu",
    ):

        super().__init__(network)

        self.action_space = action_space
        self.log_std_init = log_std_init
        self.device = device

        # For continuous action spaces, we also learn the standard deviation (log_std)
        # of the action distribution
        if isinstance(action_space, spaces.Box):
            self.log_std = torch.nn.Parameter(
                torch.ones(spaces.flatdim(action_space)) * log_std_init,
                requires_grad=True,
            ).to(device)
        else:
            self.log_std = None

    @property
    def net_config(self) -> ConfigType:
        """Configuration of the network.

        :return: Configuration of the network.
        :rtype: ConfigType
        """
        return self.wrapped.net_config

    def get_distribution(
        self, probs: torch.Tensor, log_std: Optional[torch.Tensor] = None
    ) -> Distribution:
        """Get the distribution over the action space given an observation.

        :param probs: Logits output by the network.
        :type probs: torch.Tensor
        :param log_std: Log standard deviation of the action distribution. Defaults to None.
        :type log_std: Optional[torch.Tensor]
        :return: Distribution over the action space.
        :rtype: Distribution
        """
        if isinstance(self.action_space, spaces.Box):
            assert (
                log_std is not None
            ), "log_std must be provided for continuous action spaces."
            std = torch.ones_like(probs) * log_std.exp()
            return Normal(loc=probs, scale=std)

        elif isinstance(self.action_space, spaces.Discrete):
            return Categorical(probs=probs)

        elif isinstance(self.action_space, spaces.MultiDiscrete):
            return [
                Categorical(probs=split)
                for split in torch.split(probs, list(self.action_space.nvec), dim=1)
            ]

        elif isinstance(self.action_space, spaces.MultiBinary):
            return Bernoulli(probs=probs)

        else:
            raise NotImplementedError(
                f"Action space {self.action_space} not supported."
            )

    def forward(
        self, latent: torch.Tensor, action_mask: Optional[ArrayOrTensor] = None
    ) -> Distribution:
        """Forward pass of the network.

        :param obs: Observation input.
        :type obs: TorchObsType
        :param action_mask: Mask to apply to the logits. Defaults to None.
        :type action_mask: Optional[ArrayOrTensor]
        :return: Distribution over the action space.
        :rtype: Distribution
        """
        probs = self.wrapped(latent)

        if action_mask is not None and isinstance(self.action_space, spaces.Discrete):
            # Safe to assume the action masks will always be NumPy arrays
            if isinstance(action_mask, np.ndarray):
                action_mask = (
                    np.stack(action_mask)
                    if action_mask.dtype == np.object_ or isinstance(action_mask, list)
                    else action_mask
                )
            action_mask = torch.as_tensor(
                action_mask, dtype=torch.bool, device=self.device
            ).reshape(probs.shape)
            probs = torch.where(
                action_mask,
                probs,
                torch.tensor(0.0, dtype=probs.dtype, device=self.device),
            )

        return self.get_distribution(probs, self.log_std)

    def clone(self) -> "EvolvableDistribution":
        """Clones the distribution.

        :return: Cloned distribution.
        :rtype: EvolvableDistribution
        """
        return EvolvableDistribution(
            action_space=self.action_space,
            network=self.wrapped.clone(),
            log_std_init=self.log_std_init,
            device=self.device,
        )


class DeterministicActor(EvolvableNetwork):
    """Deterministic actor network for policy-gradient algorithms. Given an observation, it outputs
    the mean of the action distribution. This is useful for e.g. DDPG, SAC, TD3.

    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
    :param action_space: Action space of the environment
    :type action_space: spaces.Space
    :param encoder_cls: Encoder class to use for the network. Defaults to None, whereby it is
        automatically built using an AgileRL module according the observation space.
    :type encoder_cls: Optional[Union[str, Type[EvolvableModule]]]
    :param encoder_config: Configuration of the encoder network.
    :type encoder_config: ConfigType
    :param head_config: Configuration of the network MLP head.
    :type head_config: Optional[ConfigType]
    :param min_latent_dim: Minimum dimension of the latent space representation.
    :type min_latent_dim: int
    :param max_latent_dim: Maximum dimension of the latent space representation.
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
        action_space: spaces.Space,
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

        self.min_action = (
            action_space.low if isinstance(action_space, spaces.Box) else None
        )
        self.max_action = (
            action_space.high if isinstance(action_space, spaces.Box) else None
        )

        # Set output activation based on action space
        if isinstance(action_space, (spaces.Discrete, spaces.MultiDiscrete)):
            output_activation = "Softmax"
        elif np.any(self.min_action < 0):
            output_activation = "Tanh"
        else:
            output_activation = "Sigmoid"

        if head_config is None:
            head_config = MlpNetConfig(
                hidden_size=[16], output_activation=output_activation
            )
        elif head_config.get("output_activation", None) is None:
            head_config["output_activation"] = output_activation

        self.build_network_head(head_config)
        self.output_activation = head_config.get("output_activation", output_activation)

    def build_network_head(self, net_config: Optional[ConfigType] = None) -> None:
        """Builds the head of the network.

        :param net_config: Configuration of the head.
        :type net_config: Optional[ConfigType]
        """
        self.head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=spaces.flatdim(self.action_space),
            name="actor",
            net_config=net_config,
        )

    def forward(self, obs: TorchObsType) -> torch.Tensor:
        """Forward pass of the network.

        :param obs: Observation input.
        :type obs: TorchObsType
        :return: Output of the network.
        :rtype: torch.Tensor
        """
        latent = self.encoder(obs)
        return self.head_net(latent)

    def recreate_network(self) -> None:
        """Recreates the network."""
        self.recreate_encoder()

        head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=spaces.flatdim(self.action_space),
            name="actor",
            net_config=self.head_net.net_config,
        )

        self.head_net = EvolvableModule.preserve_parameters(self.head_net, head_net)


class StochasticActor(DeterministicActor):
    """Stochastic actor network for policy-gradient algorithms. Given an observation, it outputs
    a distribution over the action space. This is useful for on-policy policy-gradient algorithms
    like PPO, A2C, TRPO.

    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
    :param action_space: Action space of the environment
    :type action_space: spaces.Space
    :param encoder_cls: Encoder class to use for the network. Defaults to None, whereby it is
        automatically built using an AgileRL module according the observation space.
    :type encoder_cls: Optional[Union[str, Type[EvolvableModule]]]
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

    head_net: EvolvableDistribution

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        encoder_cls: Optional[Union[str, Type[EvolvableModule]]] = None,
        encoder_config: Optional[ConfigType] = None,
        head_config: Optional[ConfigType] = None,
        log_std_init: float = 0.0,
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
            head_config=head_config,
            min_latent_dim=min_latent_dim,
            max_latent_dim=max_latent_dim,
            n_agents=n_agents,
            latent_dim=latent_dim,
            simba=simba,
            device=device,
        )

        self.log_std_init = log_std_init
        self.head_net = EvolvableDistribution(
            action_space, self.head_net, log_std_init=log_std_init, device=device
        )

    def forward(
        self, obs: TorchObsType, action_mask: Optional[ArrayOrTensor] = None
    ) -> Distribution:
        """Forward pass of the network.

        :param obs: Observation input.
        :type obs: TorchObsType
        :return: Distribution over the action space.
        :rtype: Distribution
        """
        latent = self.encoder(obs)
        return self.head_net.forward(latent, action_mask)

    def __call__(
        self, obs: TorchObsType, action_mask: Optional[ArrayOrTensor] = None
    ) -> Distribution:
        """Calls the forward method.

        :param obs: Observation input.
        :type obs: TorchObsType
        :return: Distribution over the action space.
        :rtype: Distribution
        """
        return self.forward(obs, action_mask)

    def recreate_network(self) -> None:
        """Recreates the network with the same parameters as the current network.

        :param shrink_params: Whether to shrink the parameters of the network. Defaults to False.
        :type shrink_params: bool
        """
        self.recreate_encoder()

        head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=spaces.flatdim(self.action_space),
            name="actor",
            net_config=self.head_net.net_config,
        )

        head_net = EvolvableDistribution(
            self.action_space,
            head_net,
            log_std_init=self.log_std_init,
            device=self.device,
        )

        self.head_net = EvolvableModule.preserve_parameters(self.head_net, head_net)
