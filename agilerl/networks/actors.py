from typing import List, Optional, Tuple, Type, Union

import numpy as np
import torch
from gymnasium import spaces
from torch.distributions import Bernoulli, Categorical, Distribution, Normal

from agilerl.modules.base import EvolvableModule, EvolvableWrapper
from agilerl.modules.configs import MlpNetConfig
from agilerl.networks.base import EvolvableNetwork
from agilerl.typing import ArrayOrTensor, ConfigType, DeviceType, TorchObsType

DistributionType = Union[Distribution, List[Distribution]]


def sum_independent_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Sum the values of a tensor across the independent dimensions. Assume
    dim=1 if the tensor has more than 1 dimension.

    :param tensor: Tensor to sum.
    :type tensor: torch.Tensor
    :return: Sum of the tensor.
    :rtype: torch.Tensor
    """
    return tensor.sum(dim=1) if len(tensor.shape) > 1 else tensor


def apply_action_mask_discrete(
    logits: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Apply a mask to the logits.

    :param logits: Logits.
    :type logits: torch.Tensor
    :param mask: Mask.
    :type mask: torch.Tensor
    :return: Logits with mask applied.
    :rtype: torch.Tensor
    """
    return torch.where(
        mask, logits, torch.tensor(-1e8, dtype=logits.dtype, device=logits.device)
    )


class TorchDistribution:
    """Wrapper to output a distribution over an action space for an evolvable module. It provides methods
    to sample actions and compute log probabilities, relevant for many policy-gradient algorithms such as
    PPO, A2C, TRPO.

    :param dist: Distribution to wrap.
    :type dist: Union[Distribution, List[Distribution]]
    """

    def __init__(self, dist: DistributionType):
        if isinstance(dist, list):
            assert all(
                isinstance(d, Categorical) for d in dist
            ), "Only list of Categorical distributions are supported (for MultiDiscrete action spaces)."

        self.dist = dist

    def sample(self) -> torch.Tensor:
        """Sample an action from the distribution.

        :return: Action from the distribution.
        :rtype: torch.Tensor
        """
        if isinstance(self.dist, Normal):
            return self.dist.rsample()
        elif isinstance(self.dist, (Bernoulli, Categorical)):
            return self.dist.sample()
        elif isinstance(self.dist, list):
            return torch.stack([dist.sample() for dist in self.dist], dim=1)
        else:
            raise NotImplementedError(f"Distribution {self.dist} not supported.")

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Get the log probability of the action.

        :param action: Action.
        :type action: torch.Tensor
        :return: Log probability of the action.
        :rtype: torch.Tensor
        """
        if isinstance(self.dist, (Normal, Bernoulli, Categorical)):
            return sum_independent_tensor(self.dist.log_prob(action))
        elif isinstance(self.dist, list):
            return torch.stack(
                [
                    dist.log_prob(action)
                    for dist, action in zip(self.dist, torch.unbind(action, dim=1))
                ],
                dim=1,
            )
        else:
            raise NotImplementedError(f"Distribution {self.dist} not supported.")

    def entropy(self) -> torch.Tensor:
        """Get the entropy of the action distribution.

        :return: Entropy of the action distribution.
        :rtype: torch.Tensor
        """
        if isinstance(self.dist, (Normal, Bernoulli, Categorical)):
            return sum_independent_tensor(self.dist.entropy())
        elif isinstance(self.dist, list):
            return torch.stack([dist.entropy() for dist in self.dist], dim=1)
        else:
            raise NotImplementedError(f"Distribution {self.dist} not supported.")


class EvolvableDistribution(EvolvableWrapper):
    """Wrapper to output a distribution over an action space for an evolvable module. It provides methods
    to sample actions and compute log probabilities, relevant for many policy-gradient algorithms such as
    PPO, A2C, TRPO.

    :param action_space: Action space of the environment.
    :type action_space: spaces.Space
    :param network: Network that outputs the logits of the distribution.
    :type network: EvolvableModule
    :param log_std_init: Initial log standard deviation of the action distribution. Defaults to 0.0.
    :type log_std_init: float
    :param device: Device to use for the network.
    :type device: DeviceType
    """

    wrapped: EvolvableModule
    dist: Optional[TorchDistribution]
    mask: Optional[ArrayOrTensor]

    def __init__(
        self,
        action_space: spaces.Space,
        network: EvolvableModule,
        log_std_init: float = 0.0,
        device: DeviceType = "cpu",
    ):

        super().__init__(network)

        self.action_space = action_space
        self.action_dim = spaces.flatdim(action_space)
        self.log_std_init = log_std_init
        self.device = device

        self.dist = None
        self.mask = None

        # For continuous action spaces, we also learn the standard deviation (log_std)
        # of the action distribution
        if isinstance(action_space, spaces.Box):
            self.log_std = torch.nn.Parameter(
                torch.ones(self.action_dim) * log_std_init,
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
        self, logits: torch.Tensor, log_std: Optional[torch.Tensor] = None
    ) -> TorchDistribution:
        """Get the distribution over the action space given an observation.

        :param logits: Output of the network, either logits or probabilities.
        :type logits: torch.Tensor
        :param log_std: Log standard deviation of the action distribution. Defaults to None.
        :type log_std: Optional[torch.Tensor]
        :return: Distribution over the action space.
        :rtype: Distribution
        """
        # Normal distribution for Continuous action spaces
        if isinstance(self.action_space, spaces.Box):
            assert (
                log_std is not None
            ), "log_std must be provided for continuous action spaces."
            std = torch.ones_like(logits) * log_std.exp()
            dist = Normal(loc=logits, scale=std)

        # Categorical distribution for Discrete action spaces
        elif isinstance(self.action_space, spaces.Discrete):
            dist = Categorical(logits=logits)

        # List of categorical distributions for MultiDiscrete action spaces
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            dist = [
                Categorical(logits=split)
                for split in torch.split(logits, list(self.action_space.nvec), dim=1)
            ]

        # Bernoulli distribution for MultiBinary action spaces
        elif isinstance(self.action_space, spaces.MultiBinary):
            dist = Bernoulli(logits=logits)
        else:
            raise NotImplementedError(
                f"Action space {self.action_space} not supported."
            )

        return TorchDistribution(dist)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Get the log probability of the action.

        :param action: Action.
        :type action: torch.Tensor
        :return: Log probability of the action.
        :rtype: torch.Tensor
        """
        if self.dist is None:
            raise ValueError("Distribution not initialized. Call forward first.")

        return self.dist.log_prob(action)

    def entropy(self) -> torch.Tensor:
        """Get the entropy of the action distribution.

        :return: Entropy of the action distribution.
        :rtype: torch.Tensor
        """
        if self.dist is None:
            raise ValueError("Distribution not initialized. Call forward first.")

        return self.dist.entropy()

    def apply_mask(self, logits: torch.Tensor, mask: ArrayOrTensor) -> torch.Tensor:
        """Apply a mask to the logits.

        :param logits: Logits.
        :type logits: torch.Tensor
        :param mask: Mask.
        :type mask: ArrayOrTensor
        :return: Logits with mask applied.
        :rtype: torch.Tensor
        """
        # Convert mask to tensor and reshape to match logits shape
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device).view(
            logits.shape
        )

        if isinstance(self.action_space, spaces.Discrete):
            masked_logits = apply_action_mask_discrete(logits, mask)
        elif isinstance(self.action_space, (spaces.MultiDiscrete, spaces.MultiBinary)):
            splits = (
                list(self.action_space.nvec)
                if isinstance(self.action_space, spaces.MultiDiscrete)
                else [2] * self.action_space.n
            )
            # Split mask and logits into separate distributions
            split_masks = torch.split(mask, splits, dim=1)
            split_logits = torch.split(logits, splits, dim=1)

            # Apply mask to each split
            masked_logits = []
            for split_logits, split_mask in zip(split_logits, split_masks):
                masked_logits.append(
                    apply_action_mask_discrete(split_logits, split_mask)
                )

            masked_logits = torch.cat(masked_logits, dim=1)
        else:
            raise NotImplementedError(
                f"Action space {self.action_space} not supported."
            )

        return masked_logits

    def forward(
        self, latent: torch.Tensor, action_mask: Optional[ArrayOrTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the network.

        :param latent: Latent space representation.
        :type latent: torch.Tensor
        :param action_mask: Mask to apply to the logits. Defaults to None.
        :type action_mask: Optional[ArrayOrTensor]
        :return: Action and log probability of the action.
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        logits = self.wrapped(latent)

        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = (
                    np.stack(action_mask)
                    if action_mask.dtype == np.object_ or isinstance(action_mask, list)
                    else action_mask
                )

            logits = self.apply_mask(logits, action_mask)

        # Distribution from logits
        self.dist = self.get_distribution(logits, self.log_std)

        # Sample action, compute log probability and entropy
        action = self.dist.sample()
        log_prob = self.dist.log_prob(action)
        entropy = self.dist.entropy()
        return action, log_prob, entropy

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
    :param simba: Whether to use the SimBa architecture for training the network.
    :type simba: bool
    :param recurrent: Whether to use a recurrent network.
    :type recurrent: bool
    :param device: Device to use for the network.
    :type device: str
    """

    supported_spaces = (spaces.Box,)

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
        recurrent: bool = False,
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
            recurrent=recurrent,
            device=device,
        )

        self.min_action = (
            action_space.low if isinstance(action_space, spaces.Box) else None
        )
        self.max_action = (
            action_space.high if isinstance(action_space, spaces.Box) else None
        )

        # Set output activation based on action space
        if isinstance(action_space, spaces.Discrete):
            output_activation = "Softmax"
        elif isinstance(action_space, spaces.MultiDiscrete):
            output_activation = None  # Output logits for MultiDiscrete action spaces
        elif isinstance(action_space, spaces.Box) and np.any(self.min_action < 0):
            output_activation = "Tanh"
        else:
            output_activation = "Sigmoid"

        if head_config is None:
            head_config = MlpNetConfig(
                hidden_size=[32], output_activation=output_activation
            )
        elif "output_activation" not in head_config:
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
        latent = self.extract_features(obs)
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
    :param simba: Whether to use the SimBa architecture for training the network.
    :type simba: bool
    :param recurrent: Whether to use a recurrent network.
    :type recurrent: bool
    :param device: Device to use for the network.
    :type device: str
    """

    head_net: EvolvableDistribution
    supported_spaces = (
        spaces.Box,
        spaces.Discrete,
        spaces.MultiDiscrete,
        spaces.MultiBinary,
    )

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
        recurrent: bool = False,
        device: str = "cpu",
    ):
        # Output logits forcefully -> override user-defined output activation
        if head_config is None:
            head_config = MlpNetConfig(hidden_size=[32], output_activation=None)
        else:
            head_config["output_activation"] = None

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
            recurrent=recurrent,
            device=device,
        )

        self.log_std_init = log_std_init
        self.head_net = EvolvableDistribution(
            action_space, self.head_net, log_std_init=log_std_init, device=device
        )

    def forward(
        self, obs: TorchObsType, action_mask: Optional[ArrayOrTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the network.

        :param obs: Observation input.
        :type obs: TorchObsType
        :return: Action and log probability of the action.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        latent = self.extract_features(obs)
        action, log_prob, entropy = self.head_net.forward(latent, action_mask)
        return action, log_prob, entropy

    def action_log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Get the log probability of the action.

        :param action: Action.
        :type action: torch.Tensor
        :return: Log probability of the action.
        :rtype: torch.Tensor
        """
        return self.head_net.log_prob(action)

    def action_entropy(self) -> torch.Tensor:
        """Get the entropy of the action distribution.

        :return: Entropy of the action distribution.
        :rtype: torch.Tensor
        """
        return self.head_net.entropy()

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
