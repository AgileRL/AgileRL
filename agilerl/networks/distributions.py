from typing import Optional, Union

import numpy as np
import torch
from gymnasium import spaces

from agilerl.modules.base import EvolvableModule, EvolvableWrapper
from agilerl.typing import ArrayOrTensor, DeviceType, NetConfigType
from agilerl.utils.torch_utils import (
    entropy_from_space,
    log_prob_from_space,
    sample_from_space,
)


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


class TorchDistribution:
    """
    Lightweight distribution‑like helper.
    *   keeps only **raw tensors** (``logits`` or ``mu``/``log_std``)
    *   implements ``sample``, ``log_prob`` and ``entropy`` with pure tensor ops
        → no Python allocations per call, all kernels run on GPU.

    :param action_space: Action space of the environment.
    :type action_space: spaces.Space
    :param logits: Logits.
    :type logits: torch.Tensor | None
    :param mu: Mean.
    :type mu: torch.Tensor | None
    :param log_std: Log standard deviation.
    :type log_std: torch.Tensor | None
    :param squash_output: Whether to squash the output to the action space.
    :type squash_output: bool

    """

    def __init__(
        self,
        *,
        action_space: spaces.Space,
        logits: torch.Tensor | None = None,
        mu: torch.Tensor | None = None,
        log_std: torch.Tensor | None = None,
        squash_output: bool = False,
    ):
        self.action_space = action_space
        self.logits = logits
        self.mu = mu
        self.log_std = log_std
        self.squash_output = squash_output and isinstance(action_space, spaces.Box)
        self._sampled_action: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # fast tensor‑only primitives                                        #
    # ------------------------------------------------------------------ #
    def sample(self) -> torch.Tensor:
        """Sample from the distribution for the given action space.

        :return: Sampled action.
        :rtype: torch.Tensor
        """
        self._sampled_action = sample_from_space(
            self.action_space,
            logits=self.logits,
            mu=self.mu,
            log_std=self.log_std,
            squash_output=self.squash_output,
        )
        return self._sampled_action

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Log probability of the action.

        :param action: Action.
        :type action: torch.Tensor
        :return: Log probability of the action.
        :rtype: torch.Tensor
        """
        return log_prob_from_space(
            self.action_space,
            action,
            logits=self.logits,
            mu=self.mu,
            log_std=self.log_std,
        )

    def entropy(self) -> torch.Tensor:
        """Entropy of the distribution.

        :return: Entropy of the distribution.
        :rtype: torch.Tensor
        """
        return entropy_from_space(
            self.action_space,
            logits=self.logits,
            mu=self.mu,
            log_std=self.log_std,
        )


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
        :rtype: Distribution # This should ideally be TorchDistribution, but keeping for consistency with old file if Distribution was a type alias
        """
        # Normal distribution for Continuous action spaces
        if isinstance(self.action_space, spaces.Box):
            log_std = self.log_std.expand_as(logits)
            # Pass mu and log_std directly to TorchDistribution
            return TorchDistribution(
                action_space=self.action_space,
                mu=logits,
                log_std=log_std,
                squash_output=self.squash_output,
            )

        # Categorical distribution for Discrete action spaces
        elif isinstance(self.action_space, spaces.Discrete):
            # Pass logits directly to TorchDistribution
            return TorchDistribution(
                action_space=self.action_space,
                logits=logits,
                squash_output=self.squash_output,  # squash_output is ignored for discrete
            )

        # List of categorical distributions for MultiDiscrete action spaces
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            # Pass logits directly to TorchDistribution
            return TorchDistribution(
                action_space=self.action_space,
                logits=logits,
                squash_output=self.squash_output,  # squash_output is ignored for discrete
            )

        # Bernoulli distribution for MultiBinary action spaces
        elif isinstance(self.action_space, spaces.MultiBinary):
            # Pass logits directly to TorchDistribution
            return TorchDistribution(
                action_space=self.action_space,
                logits=logits,
                squash_output=self.squash_output,  # squash_output is ignored for discrete
            )
        else:
            raise NotImplementedError(
                f"Action space {self.action_space} not supported."
            )

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Get the log probability of the action.

        :param action: Action.
        :type action: torch.Tensor
        :return: Log probability of the action.
        :rtype: torch.Tensor
        """
        if self.dist is None:
            raise ValueError("Distribution not initialized. Call forward first.")

        # The new TorchDistribution handles squashing correction internally for Box space
        return self.dist.log_prob(action)

    def entropy(self) -> torch.Tensor:
        """Get the entropy of the action distribution.

        :return: Entropy of the action distribution.
        :rtype: torch.Tensor
        """
        if self.dist is None:
            raise ValueError("Distribution not initialized. Call forward first.")

        # The new TorchDistribution returns analytical entropy for supported spaces
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
                else [
                    self.action_space.n
                ]  # For MultiBinary, nvec is not present, use n
            )
            # Split mask and logits into separate distributions
            split_masks = torch.split(mask, splits, dim=1)
            split_logits = torch.split(logits, splits, dim=1)

            # Apply mask to each split
            masked_logits = []
            for split_logits_i, split_mask_i in zip(
                split_logits, split_masks
            ):  # Renamed for clarity
                masked_logits.append(
                    apply_action_mask_discrete(split_logits_i, split_mask_i)
                )

            masked_logits = torch.cat(masked_logits, dim=1)
        else:
            # This should ideally not be reached if get_distribution handles the space,
            # but keeping for safety.
            raise NotImplementedError(
                f"Action space {self.action_space} not supported for masking."
            )

        return masked_logits

    def forward(
        self,
        latent: torch.Tensor,
        action_mask: Optional[ArrayOrTensor] = None,
        sample: bool = True,
    ) -> Union[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[None, None, torch.Tensor]
    ]:
        """Forward pass of the network.

        :param latent: Latent space representation.
        :type latent: torch.Tensor
        :param action_mask: Mask to apply to the logits. Defaults to None.
        :type action_mask: Optional[ArrayOrTensor]
        :param sample: Whether to sample an action or return the mode/mean. Defaults to True.
        :type sample: bool
        :return: Action and log probability of the action.
        :rtype: Union[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[None, torch.Tensor, torch.Tensor]]
        """
        logits = self.wrapped(latent)

        if action_mask is not None:
            if isinstance(action_mask, (np.ndarray, list)):
                # Attempt to stack if it's a list of arrays or object array, typical for vectorized envs
                if isinstance(action_mask, list) or (
                    isinstance(action_mask, np.ndarray)
                    and action_mask.dtype == np.object_
                ):
                    try:
                        action_mask = np.stack(action_mask)
                    except Exception:
                        # If stacking fails, it might be a non-uniform list or other structure not directly convertible.
                        # This path assumes action_mask should become a single tensor.
                        # If it's already a correct tensor, as_tensor below handles it.
                        pass  # Allow as_tensor to handle or raise error if still problematic

            # Ensure action_mask is a tensor before applying.
            # The view in apply_mask expects a compatible shape or will error.
            action_mask = torch.as_tensor(
                action_mask, device=self.device, dtype=torch.bool
            )

            logits = self.apply_mask(logits, action_mask)

        # Distribution from logits
        # get_distribution now creates the new TorchDistribution object
        self.dist = self.get_distribution(logits)

        # Sample action, compute log probability and entropy
        if sample:
            action = self.dist.sample()
            log_prob = self.dist.log_prob(action)
        else:
            action = None  # Mode/mean might be more appropriate if not sampling
            log_prob = (
                None  # Log prob of mode/mean typically not used in PPO sample step
            )

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
