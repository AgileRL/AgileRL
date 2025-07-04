from typing import Dict, List, Optional, Protocol, Tuple, Type, Union

import numpy as np
import torch
from gymnasium import spaces
from torch.distributions import Bernoulli, Categorical, Distribution, Normal

from agilerl.modules.base import EvolvableModule, EvolvableWrapper
from agilerl.typing import ArrayOrTensor, DeviceType, NetConfigType

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
    return torch.where(mask, logits, torch.full_like(logits, -1e8).to(logits.device))


class DistributionHandler(Protocol):
    """Protocol for distribution handlers that implement sampling, log_prob, and entropy methods."""

    def sample(self, distribution: DistributionType) -> torch.Tensor:
        """Sample an action from the distribution."""
        ...

    def log_prob(
        self, distribution: DistributionType, action: torch.Tensor
    ) -> torch.Tensor:
        """Get the log probability of the action."""
        ...

    def entropy(self, distribution: DistributionType) -> Optional[torch.Tensor]:
        """Get the entropy of the action distribution."""
        ...


class NormalHandler:
    """Handler for Normal distributions."""

    def sample(self, distribution: Normal) -> torch.Tensor:
        """Sample an action from the distribution using reparameterization trick.

        :param distribution: Distribution to sample from.
        :type distribution: Normal
        :return: Sampled action.
        :rtype: torch.Tensor
        """
        return distribution.sample()

    def log_prob(self, distribution: Normal, action: torch.Tensor) -> torch.Tensor:
        """Get the log probability of the action.

        :param distribution: Distribution to compute log probability for.
        :type distribution: Normal
        :param action: Action.
        :type action: torch.Tensor
        """
        return sum_independent_tensor(distribution.log_prob(action))

    def entropy(self, distribution: Normal) -> torch.Tensor:
        """Get the entropy of the action distribution.

        :param distribution: Distribution to compute entropy for.
        :type distribution: Normal
        :return: Entropy of the action distribution.
        :rtype: torch.Tensor
        """
        return sum_independent_tensor(distribution.entropy())


class BernoulliHandler:
    """Handler for Bernoulli distributions."""

    def sample(self, distribution: Bernoulli) -> torch.Tensor:
        """Sample an action from the distribution.

        :param distribution: Distribution to sample from.
        :type distribution: Bernoulli
        :return: Sampled action.
        :rtype: torch.Tensor
        """
        return distribution.sample()

    def log_prob(self, distribution: Bernoulli, action: torch.Tensor) -> torch.Tensor:
        """Get the log probability of the action.

        :param distribution: Distribution to compute log probability for.
        :type distribution: Bernoulli
        :param action: Action.
        :type action: torch.Tensor
        """
        return distribution.log_prob(action).sum(dim=1)

    def entropy(self, distribution: Bernoulli) -> torch.Tensor:
        """Get the entropy of the action distribution.

        :param distribution: Distribution to compute entropy for.
        :type distribution: Bernoulli
        :return: Entropy of the action distribution.
        :rtype: torch.Tensor
        """
        return distribution.entropy().sum(dim=1)


class CategoricalHandler:
    """Handler for Categorical distributions."""

    def sample(self, distribution: Categorical) -> torch.Tensor:
        """Sample an action from the distribution.

        :param distribution: Distribution to sample from.
        :type distribution: Categorical
        :return: Sampled action.
        :rtype: torch.Tensor
        """
        return distribution.sample()

    def log_prob(self, distribution: Categorical, action: torch.Tensor) -> torch.Tensor:
        """Get the log probability of the action.

        :param distribution: Distribution to compute log probability for.
        :type distribution: Categorical
        :param action: Action.
        :type action: torch.Tensor
        """
        return distribution.log_prob(action)

    def entropy(self, distribution: Categorical) -> torch.Tensor:
        """Get the entropy of the action distribution.

        :param distribution: Distribution to compute entropy for.
        :type distribution: Categorical
        :return: Entropy of the action distribution.
        :rtype: torch.Tensor
        """
        return distribution.entropy()


class MultiCategoricalHandler:
    """Handler for list of Categorical distributions (MultiDiscrete action spaces)."""

    def sample(self, distribution: List[Categorical]) -> torch.Tensor:
        """Sample an action from the distribution.

        :param distribution: List of Categorical distributions to sample from.
        :type distribution: List[Categorical]
        :return: Sampled action.
        :rtype: torch.Tensor
        """
        return torch.stack([dist.sample() for dist in distribution], dim=1)

    def log_prob(
        self, distribution: List[Categorical], action: torch.Tensor
    ) -> torch.Tensor:
        """Get the log probability of the action.

        :param distribution: List of Categorical distributions to compute log probability for.
        :type distribution: List[Categorical]
        :param action: Action.
        :type action: torch.Tensor
        """
        unbinded_actions = torch.unbind(action, dim=1)
        multi_log_prob = [
            dist.log_prob(act) for dist, act in zip(distribution, unbinded_actions)
        ]
        return torch.stack(multi_log_prob, dim=1).sum(dim=1)

    def entropy(self, distribution: List[Categorical]) -> torch.Tensor:
        """Get the entropy of the action distribution.

        :param distribution: List of Categorical distributions to compute entropy for.
        :type distribution: List[Categorical]
        :return: Entropy of the action distribution.
        :rtype: torch.Tensor
        """
        return torch.stack([dist.entropy() for dist in distribution], dim=1).sum(dim=1)


class TorchDistribution:
    """Wrapper to output a distribution over an action space for an evolvable module. It provides methods
    to sample actions and compute log probabilities, relevant for many policy-gradient algorithms such as
    PPO, A2C, TRPO.

    :param distribution: Distribution to wrap.
    :type distribution: Union[Distribution, List[Distribution]]
    :param squash_output: Whether to squash the output to the action space.
    :type squash_output: bool
    """

    # Map distribution types to their handlers
    _handlers: Dict[Type, DistributionHandler] = {
        Normal: NormalHandler(),
        Bernoulli: BernoulliHandler(),
        Categorical: CategoricalHandler(),
        list: MultiCategoricalHandler(),
    }

    def __init__(
        self,
        distribution: DistributionType,
        squash_output: bool = False,
    ) -> None:
        if isinstance(distribution, list):
            assert all(
                isinstance(d, Categorical) for d in distribution
            ), "Only list of Categorical distributions are supported (for MultiDiscrete action spaces)."

        self.distribution = distribution
        self.squash_output = squash_output
        self.sampled_action = None
        self._handler = self._get_handler(distribution)

    def _get_handler(self, distribution: DistributionType) -> DistributionHandler:
        """Get the appropriate handler for the distribution type.

        :param distribution: Distribution to get handler for.
        :type distribution: DistributionType
        :return: Appropriate handler for the distribution type.
        :rtype: DistributionHandler
        """
        if isinstance(distribution, list):
            return self._handlers[list]

        for dist_type, handler in self._handlers.items():
            if isinstance(distribution, dist_type) and dist_type is not list:
                return handler

        raise NotImplementedError(f"Distribution {type(distribution)} not supported.")

    def sample(self) -> torch.Tensor:
        """Sample an action from the distribution.

        :return: Action from the distribution.
        :rtype: torch.Tensor
        """
        self.sampled_action = self._handler.sample(self.distribution)

        if self.squash_output:
            return torch.tanh(self.sampled_action)

        return self.sampled_action

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Get the log probability of the action.

        :param action: Action.
        :type action: torch.Tensor
        :return: Log probability of the action.
        :rtype: torch.Tensor
        """
        _action = action if not self.squash_output else self.sampled_action

        log_prob = self._handler.log_prob(self.distribution, _action)

        # Correction for squashed outputs as per SAC paper:
        # See https://arxiv.org/html/2410.16739v1
        if self.squash_output:
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=1)

        return log_prob

    def entropy(self) -> Optional[torch.Tensor]:
        """Get the entropy of the action distribution.

        :return: Entropy of the action distribution.
        :rtype: torch.Tensor, None
        """
        # No analytical form for entropy with squashed outputs so must
        # use -log_prob.mean() in algorithm instead
        if self.squash_output:
            return None

        return self._handler.entropy(self.distribution)


class EvolvableDistribution(EvolvableWrapper):
    """Wrapper to output a distribution over an action space for an evolvable module. It provides methods
    to sample actions and compute log probabilities, relevant for many policy-gradient algorithms such as
    PPO, A2C, TRPO.

    :param action_space: Action space of the environment.
    :type action_space: spaces.Space
    :param network: Network that outputs the logits of the distribution.
    :type network: EvolvableModule
    :param action_std_init: Initial log standard deviation of the action distribution. Defaults to 0.0.
    :type action_std_init: float
    :param squash_output: Whether to squash the output to the action space.
    :type squash_output: bool
    :param device: Device to use for the network.
    :type device: DeviceType
    """

    wrapped: EvolvableModule
    dist: Optional[TorchDistribution]
    mask: Optional[ArrayOrTensor]
    log_std: Optional[torch.nn.Parameter]

    def __init__(
        self,
        action_space: spaces.Space,
        network: EvolvableModule,
        action_std_init: float = 0.0,
        squash_output: bool = False,
        device: DeviceType = "cpu",
    ):
        super().__init__(network)

        self.action_space = action_space
        self.action_dim = spaces.flatdim(action_space)
        self.action_std_init = action_std_init
        self.device = device
        self.squash_output = squash_output and isinstance(action_space, spaces.Box)
        self.dist = None
        self.mask = None

        # For continuous action spaces, we also learn the standard
        # deviation (log_std) of the action distribution
        if isinstance(action_space, spaces.Box):
            self.log_std = torch.nn.Parameter(
                torch.ones(1, np.prod(action_space.shape), device=device)
                * action_std_init
            )

    @property
    def net_config(self) -> NetConfigType:
        """Configuration of the network.

        :return: Configuration of the network.
        :rtype: NetConfigType
        """
        return self.wrapped.net_config

    def get_distribution(self, logits: torch.Tensor) -> TorchDistribution:
        """Get the distribution over the action space given an observation.

        :param logits: Output of the network, either logits or probabilities.
        :type logits: torch.Tensor
        :return: Distribution over the action space.
        :rtype: Distribution
        """
        # Normal distribution for Continuous action spaces
        if isinstance(self.action_space, spaces.Box):
            log_std = self.log_std.expand_as(logits)
            action_std = torch.exp(log_std)
            dist = Normal(loc=logits, scale=action_std)

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

        return TorchDistribution(dist, self.squash_output)

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
                else [self.action_space.n]
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
        self,
        latent: torch.Tensor,
        action_mask: Optional[ArrayOrTensor] = None,
        sample: bool = True,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[None, None, torch.Tensor]
    ]:
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
            if isinstance(action_mask, (np.ndarray, list)):
                action_mask = (
                    np.stack(action_mask)
                    if action_mask.dtype == np.object_ or isinstance(action_mask, list)
                    else action_mask
                )

            logits = self.apply_mask(logits, action_mask)

        # Distribution from logits
        self.dist = self.get_distribution(logits)

        # Sample action, compute log probability and entropy
        if sample:
            action = self.dist.sample()
            log_prob = self.dist.log_prob(action)
        else:
            action = None
            log_prob = None

        entropy = self.dist.entropy()
        return action, log_prob, entropy

    def clone(self) -> "EvolvableDistribution":
        """Clones the distribution.

        :return: Cloned distribution.
        :rtype: EvolvableDistribution
        """
        clone = EvolvableDistribution(
            action_space=self.action_space,
            network=self.wrapped.clone(),
            action_std_init=self.action_std_init,
            squash_output=self.squash_output,
            device=self.device,
        )
        clone.rng = self.rng
        return clone
