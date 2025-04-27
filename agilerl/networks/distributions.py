from typing import Dict, List, Optional, Protocol, Tuple, Type, Union

import numpy as np
import torch
from gymnasium import spaces
from torch.distributions import Bernoulli, Categorical, Distribution, Normal

from agilerl.modules.base import EvolvableModule, EvolvableWrapper
from agilerl.typing import ArrayOrTensor, ConfigType, DeviceType

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
        """Sample an action from the distribution using a faster method
        based on torch.searchsorted.

        :param distribution: Distribution to sample from.
        :type distribution: Categorical
        :return: Sampled action.
        :rtype: torch.Tensor
        """
        probs = distribution.probs
        # Generate random numbers on the correct device and shape
        # Calculate cumulative probabilities
        # Use searchsorted to find the sampled indices efficiently
        # right=True ensures correct handling for probabilities summing to 1
        return torch.searchsorted(
            probs.cumsum(-1),
            torch.rand(probs.shape[:-1], device=probs.device)[..., None],
        ).squeeze(-1)

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
    :param squash_output: Whether to squash the output to the action space (for Box spaces).
    :type squash_output: bool
    :param device: Device to use for the network.
    :type device: DeviceType
    """

    wrapped: EvolvableModule
    mask: Optional[ArrayOrTensor]
    log_std: Optional[torch.nn.Parameter]
    # Attributes determined in __init__
    _dist_class: Type[Distribution] | List[Type[Categorical]]
    _handler: DistributionHandler
    _action_splits: Optional[List[int]]
    # Cached values from last forward pass
    _last_distribution: Optional[DistributionType]
    _last_sampled_action: Optional[torch.Tensor]  # Action before potential squashing
    _last_action: Optional[torch.Tensor]  # Action after potential squashing
    _last_log_prob: Optional[torch.Tensor]
    _last_entropy: Optional[torch.Tensor]

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
        self.mask = None
        self.log_std = None

        # Handlers mapping - reuse instances for efficiency
        self._handlers_map: Dict[Type, DistributionHandler] = {
            Normal: NormalHandler(),
            Bernoulli: BernoulliHandler(),
            Categorical: CategoricalHandler(),
            list: MultiCategoricalHandler(),  # Use list type for MultiCategorical
        }

        self._action_splits = None  # For MultiDiscrete/MultiBinary

        # Determine distribution class and handler based on action space
        if isinstance(self.action_space, spaces.Box):
            self._dist_class = Normal
            self._handler = self._handlers_map[Normal]
            # Learn standard deviation for continuous actions
            self.log_std = torch.nn.Parameter(
                torch.ones(1, np.prod(action_space.shape), device=device)
                * action_std_init
            )
        elif isinstance(self.action_space, spaces.Discrete):
            self._dist_class = Categorical
            self._handler = self._handlers_map[Categorical]
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            self._dist_class = list  # Represent as list of Categorical
            self._handler = self._handlers_map[list]
            self._action_splits = list(self.action_space.nvec)
        elif isinstance(self.action_space, spaces.MultiBinary):
            # Although MultiBinary is often handled by Bernoulli,
            # if we need multiple independent Bernoulli outputs, we might
            # treat it similarly to MultiDiscrete but with Bernoulli.
            # For simplicity, let's assume a single Bernoulli output per feature
            # or treat as MultiDiscrete with n=2 if that's the intended use.
            # Current implementation uses single Bernoulli.
            # If it needs to be MultiDiscrete-like (list of Bernoulli), adjust here.
            self._dist_class = Bernoulli
            self._handler = self._handlers_map[Bernoulli]
            # self._action_splits = [1] * self.action_space.n # If treated like MultiDiscrete
        else:
            raise NotImplementedError(
                f"Action space {self.action_space} not supported."
            )

        # Initialize caches
        self._last_distribution = None
        self._last_sampled_action = None
        self._last_action = None
        self._last_log_prob = None
        self._last_entropy = None

    @property
    def net_config(self) -> ConfigType:
        """Configuration of the network."""
        return self.wrapped.net_config

    def _create_torch_distribution(self, logits: torch.Tensor) -> DistributionType:
        """Creates the underlying PyTorch distribution object."""
        if self._dist_class is Normal:
            assert self.log_std is not None, (
                "log_std must be initialized for Box spaces"
            )
            log_std_expanded = self.log_std.expand_as(logits)
            action_std = torch.exp(log_std_expanded)
            return Normal(loc=logits, scale=action_std)
        elif self._dist_class is Categorical:
            return Categorical(logits=logits)
        elif self._dist_class is list:  # MultiDiscrete
            assert self._action_splits is not None, (
                "Action splits needed for MultiDiscrete"
            )
            return [
                Categorical(logits=split)
                for split in torch.split(logits, self._action_splits, dim=1)
            ]
        elif self._dist_class is Bernoulli:
            return Bernoulli(logits=logits)
        else:
            # Should not happen due to __init__ checks
            raise TypeError(f"Distribution class {self._dist_class} not recognized.")

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Get the log probability of a given action using the *last computed distribution*.

        :param action: Action.
        :type action: torch.Tensor
        :return: Log probability of the action.
        :rtype: torch.Tensor
        """
        if self._last_distribution is None:
            raise ValueError("Distribution not created. Call forward first.")

        # Determine which action to use for log_prob calculation based on squashing
        # If squashed, we need the *original sampled* action before tanh
        # However, the standard way to evaluate a policy's log_prob is on the
        # *given* action (which might come from a buffer, not the last sample).
        # The SAC correction applies based on the *given* action (after tanh).
        action_for_log_prob = action

        # If we squashed the output during sampling, the log_prob needs correction.
        # The correction uses the action *after* tanh (the input `action`).
        # The log_prob itself from the *original* distribution (e.g., Normal)
        # needs to be calculated using the *inverse* tanh of the action,
        # which is the sample *before* tanh.
        # This seems contradictory. Let's follow the SAC paper / common implementations.
        # Log prob is calculated for the *given* action `a` using the distribution `pi`.
        # The correction term `log(1 - tanh(mu)^2)` or `log(1 - a^2)` is subtracted.
        # The `log_prob` method of the base distribution (e.g., Normal) needs the
        # pre-squashed value `z` where `a = tanh(z)`.
        # `z = atanh(a)`. Need to handle potential numerical instability near +/- 1.

        if self.squash_output:
            # Clip action to avoid +/- 1 values for atanh
            clipped_action = torch.clamp(action, -1.0 + 1e-6, 1.0 + 1e-6)
            # Inverse tanh: pre-squashed action
            pre_tanh_action = torch.atanh(clipped_action)
            # Calculate log_prob using the pre-squashed action
            log_prob_base = self._handler.log_prob(
                self._last_distribution, pre_tanh_action
            )
            # Apply SAC correction using the squashed action
            # log(1 - a^2) = sum(log(1 - a_i^2)) over action dims
            sac_correction = torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=1)
            log_prob = log_prob_base - sac_correction
        else:
            # No squashing, just calculate log_prob directly
            log_prob = self._handler.log_prob(
                self._last_distribution, action_for_log_prob
            )

        return log_prob

    def entropy(self) -> Optional[torch.Tensor]:
        """Get the entropy of the *last computed* action distribution.

        :return: Entropy of the action distribution, or None if squashed.
        :rtype: torch.Tensor, None
        """
        if self._last_distribution is None:
            raise ValueError("Distribution not created. Call forward first.")

        # Entropy is not well-defined analytically for squashed distributions
        if self.squash_output:
            return None

        # Return cached entropy from forward pass if available and valid
        if self._last_entropy is not None:
            return self._last_entropy
        else:
            # Recompute if needed (should have been computed in forward)
            return self._handler.entropy(self._last_distribution)

    def apply_mask(self, logits: torch.Tensor, mask: ArrayOrTensor) -> torch.Tensor:
        """Apply a mask to the logits.

        :param logits: Logits.
        :type logits: torch.Tensor
        :param mask: Mask.
        :type mask: ArrayOrTensor
        :return: Logits with mask applied.
        :rtype: torch.Tensor
        """
        # Convert mask to tensor and reshape if necessary
        # Assuming mask shape matches or is broadcastable to logits shape
        mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
        if mask.shape != logits.shape:
            # Attempt to reshape based on common patterns (e.g., batch dim only)
            if mask.ndim == 1 and mask.shape[0] == logits.shape[0]:
                mask = mask.unsqueeze(1).expand_as(logits)  # Expand along action dim
            elif mask.shape == self.action_space.shape:
                mask = mask.unsqueeze(0).expand_as(logits)  # Add batch dim
            else:
                try:
                    mask = mask.view(logits.shape)
                except RuntimeError as e:
                    raise ValueError(
                        f"Mask shape {mask.shape} incompatible with logits shape {logits.shape}"
                    ) from e

        # Apply mask based on distribution type
        if self._dist_class is Categorical:
            masked_logits = apply_action_mask_discrete(logits, mask)
        elif self._dist_class is list:  # MultiDiscrete
            assert self._action_splits is not None, (
                "Action splits needed for MultiDiscrete masking"
            )
            split_masks = torch.split(mask, self._action_splits, dim=1)
            split_logits = torch.split(logits, self._action_splits, dim=1)
            masked_logits_parts = [
                apply_action_mask_discrete(split_logits_part, split_mask)
                for split_logits_part, split_mask in zip(split_logits, split_masks)
            ]
            masked_logits = torch.cat(masked_logits_parts, dim=1)
        elif self._dist_class is Bernoulli:
            # Masking for Bernoulli typically means forcing certain actions (0 or 1)
            # or disallowing them. Simple logit masking might not be standard.
            # Assuming masking sets disallowed action logits to -inf, similar to Categorical.
            masked_logits = apply_action_mask_discrete(logits, mask)
        elif self._dist_class is Normal:
            # Masking for continuous spaces (Normal) is less common or standard.
            # It might involve clamping actions post-sampling, or adjusting mean/std.
            # Raising error as simple logit masking isn't applicable.
            raise NotImplementedError(
                f"Action masking not implemented for {self._dist_class}"
            )
        else:
            raise NotImplementedError(
                f"Action masking not supported for action space type associated with {self._dist_class}"
            )

        return masked_logits

    def forward(
        self, latent: torch.Tensor, action_mask: Optional[ArrayOrTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass: compute distribution, sample action, compute log_prob and entropy.

        :param latent: Latent space representation.
        :type latent: torch.Tensor
        :param action_mask: Mask to apply to the logits. Defaults to None.
        :type action_mask: Optional[ArrayOrTensor]
        :return: Tuple of (action, log_prob, entropy). Entropy is None if using squash_output.
        :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        """
        logits = self.wrapped(latent)

        # Apply mask if provided
        if action_mask is not None:
            # Preprocess mask if it's a list of arrays/objects (e.g., from vec envs)
            if isinstance(action_mask, list) or (
                isinstance(action_mask, np.ndarray) and action_mask.dtype == np.object_
            ):
                try:
                    action_mask = np.stack(action_mask)
                except ValueError as e:
                    raise ValueError(
                        f"Could not stack action_mask: {action_mask}"
                    ) from e

            logits = self.apply_mask(logits, action_mask)

        # Create the PyTorch distribution object
        distribution = self._create_torch_distribution(logits)
        self._last_distribution = distribution  # Cache distribution

        # Sample action using the handler
        sampled_action = self._handler.sample(distribution)
        self._last_sampled_action = sampled_action  # Cache pre-squashed action

        # Apply squashing if needed (tanh)
        if self.squash_output:
            action = torch.tanh(sampled_action)
        else:
            action = sampled_action
        self._last_action = action  # Cache final action

        # Calculate log probability using the handler and apply SAC correction if squashed
        log_prob = self.log_prob(
            action
        )  # Use the dedicated log_prob method which handles caching

        # Calculate entropy using the handler
        # Returns None if squashed
        entropy = (
            self.entropy()
        )  # Use the dedicated entropy method which handles caching

        # Cache results
        self._last_log_prob = log_prob
        self._last_entropy = entropy  # May be None

        return action, log_prob, entropy

    def clone(self) -> "EvolvableDistribution":
        """Clones the distribution wrapper and its underlying network."""
        cloned_net = self.wrapped.clone()
        cloned_dist = EvolvableDistribution(
            action_space=self.action_space,
            network=cloned_net,
            action_std_init=self.action_std_init,
            squash_output=self.squash_output,
            device=self.device,
        )
        # Clone log_std parameter if it exists
        if self.log_std is not None:
            cloned_dist.log_std = torch.nn.Parameter(self.log_std.clone())
            cloned_dist.log_std.requires_grad = self.log_std.requires_grad

        return cloned_dist
