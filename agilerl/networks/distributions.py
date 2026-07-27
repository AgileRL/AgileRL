from typing import Literal, overload

import numpy as np
import numpy.typing as npt
import torch
from gymnasium import spaces
from jaxtyping import Shaped

from agilerl.modules.base import EvolvableModule, EvolvableWrapper
from agilerl.typing import (
    ActionEntropy,
    ActionLogits,
    ActionMaskInput,
    ActionMaskTensor,
    ArrayOrTensor,
    DeviceType,
    LatentTensor,
    LogProbs,
    NetConfigType,
    SampledAction,
    numpy_action_mask,
)
from agilerl.utils.torch_utils import (
    entropy_from_space,
    log_prob_from_space,
    sample_from_space,
)


def apply_action_mask_discrete(
    logits: ActionLogits, mask: ActionMaskTensor
) -> ActionLogits:
    """Apply a mask to the logits.

    :param logits: Logits.
    :type logits: ActionLogits
    :param mask: Mask.
    :type mask: ActionMaskTensor
    :return: Logits with mask applied.
    :rtype: ActionLogits
    """
    return torch.where(mask, logits, torch.full_like(logits, -1e8).to(logits.device))


class TorchDistribution:
    """Lightweight distribution-like helper.
    *   keeps only **raw tensors** (``logits`` or ``mu``/``log_std``)
    *   implements ``sample``, ``log_prob`` and ``entropy`` with pure tensor ops
        -> no Python allocations per call, all kernels run on GPU.

    :param action_space: Action space of the environment.
    :type action_space: spaces.Space
    :param logits: Logits.
    :type logits: ActionLogits | None
    :param mu: Mean.
    :type mu: ActionLogits | None
    :param log_std: Log standard deviation.
    :type log_std: ActionLogits | None
    :param squash_output: Whether to squash the output to the action space.
    :type squash_output: bool
    """

    def __init__(
        self,
        *,
        action_space: spaces.Space,
        logits: ActionLogits | None = None,
        mu: ActionLogits | None = None,
        log_std: ActionLogits | None = None,
        squash_output: bool = False,
    ) -> None:
        self.action_space = action_space
        self.logits = logits
        self.mu = mu
        self.log_std = log_std
        self.squash_output = squash_output and isinstance(action_space, spaces.Box)
        self._sampled_action: SampledAction | None = None

    def sample(self) -> SampledAction:
        """Sample from the distribution for the given action space.

        :return: Sampled action.
        :rtype: SampledAction
        """
        self._sampled_action = sample_from_space(
            self.action_space,
            logits=self.logits,
            mu=self.mu,
            log_std=self.log_std,
            squash_output=self.squash_output,
        )
        return self._sampled_action

    def log_prob(self, action: SampledAction) -> LogProbs:
        """Log probability of the action.

        :param action: Action.
        :type action: SampledAction
        :return: Log probability of the action.
        :rtype: LogProbs
        """
        return log_prob_from_space(
            self.action_space,
            action,
            logits=self.logits,
            mu=self.mu,
            log_std=self.log_std,
        )

    def entropy(self) -> ActionEntropy:
        """Entropy of the distribution.

        :return: Entropy of the distribution.
        :rtype: ActionEntropy
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
    dist: TorchDistribution | None
    mask: ArrayOrTensor | None
    log_std: torch.nn.Parameter | None

    def __init__(
        self,
        action_space: spaces.Space,
        network: EvolvableModule,
        action_std_init: float = 0.0,
        squash_output: bool = False,
        device: DeviceType = "cpu",
    ) -> None:
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
                torch.ones(1, int(np.prod(action_space.shape)), device=device)
                * action_std_init
            )
        else:
            self.log_std = None

    @property
    def net_config(self) -> NetConfigType:
        """Configuration of the network.

        :return: Configuration of the network.
        :rtype: NetConfigType
        """
        return self.wrapped.net_config

    def get_distribution(self, logits: ActionLogits) -> TorchDistribution:
        """Get the distribution over the action space given an observation.

        :param logits: Output of the network, either logits or probabilities.
        :type logits: ActionLogits
        :return: Distribution over the action space.
        :rtype: Distribution # This should ideally be TorchDistribution, but keeping for consistency with old file if Distribution was a type alias
        """
        # Normal distribution for Continuous action spaces
        if isinstance(self.action_space, spaces.Box):
            assert self.log_std is not None, (
                "log_std is initialized for Box action spaces."
            )
            log_std = self.log_std.expand_as(logits)
            return TorchDistribution(
                action_space=self.action_space,
                mu=logits,
                log_std=log_std,
                squash_output=self.squash_output,
            )

        # Categorical distribution for Discrete action spaces
        if isinstance(self.action_space, spaces.Discrete):
            return TorchDistribution(
                action_space=self.action_space,
                logits=logits,
                squash_output=self.squash_output,
            )

        # List of categorical distributions for MultiDiscrete action spaces
        if isinstance(self.action_space, spaces.MultiDiscrete):
            return TorchDistribution(
                action_space=self.action_space,
                logits=logits,
                squash_output=self.squash_output,
            )

        # Bernoulli distribution for MultiBinary action spaces
        if isinstance(self.action_space, spaces.MultiBinary):
            return TorchDistribution(
                action_space=self.action_space,
                logits=logits,
                squash_output=self.squash_output,
            )
        msg = f"Action space {self.action_space} not supported."
        raise NotImplementedError(msg)

    def log_prob(self, action: SampledAction) -> LogProbs:
        """Get the log probability of the action.

        :param action: Action.
        :type action: SampledAction
        :return: Log probability of the action.
        :rtype: LogProbs
        """
        if self.dist is None:
            msg = "Distribution not initialized. Call forward first."
            raise ValueError(msg)

        # Handles squashing correction internally for Box space
        return self.dist.log_prob(action)

    def entropy(self) -> ActionEntropy:
        """Get the entropy of the action distribution.

        :return: Entropy of the action distribution.
        :rtype: ActionEntropy
        """
        if self.dist is None:
            msg = "Distribution not initialized. Call forward first."
            raise ValueError(msg)

        # Returns analytical entropy for supported spaces
        return self.dist.entropy()

    def apply_mask(
        self,
        logits: ActionLogits,
        mask: Shaped[torch.Tensor, "batch action_dim"]
        | Shaped[npt.NDArray, "batch action_dim"],
    ) -> ActionLogits:
        """Apply a mask to the logits.

        :param logits: Logits.
        :type logits: ActionLogits
        :param mask: Mask. Any dtype ``torch.as_tensor`` can cast to bool; env
            masks arrive as 1/0 integers.
        :type mask: Shaped[torch.Tensor, "batch action_dim"] | Shaped[npt.NDArray, "batch action_dim"]
        :return: Logits with mask applied.
        :rtype: ActionLogits
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
                split_logits, split_masks, strict=False
            ):
                masked_logits.append(
                    apply_action_mask_discrete(split_logits_i, split_mask_i)
                )

            masked_logits = torch.cat(masked_logits, dim=1)
        else:
            # This should ideally not be reached if get_distribution handles the space,
            # but keeping for safety.
            msg = f"Action space {self.action_space} not supported for masking."
            raise NotImplementedError(msg)

        return masked_logits

    @overload
    def forward(
        self,
        latent: LatentTensor,
        action_mask: ActionMaskInput = None,
        sample: Literal[True] = True,
    ) -> tuple[SampledAction, LogProbs, ActionEntropy]: ...

    @overload
    def forward(
        self,
        latent: LatentTensor,
        action_mask: ActionMaskInput,
        sample: Literal[False],
    ) -> tuple[None, None, ActionEntropy]: ...

    def forward(
        self,
        latent: LatentTensor,
        action_mask: ActionMaskInput = None,
        sample: bool = True,
    ) -> (
        tuple[SampledAction, LogProbs, ActionEntropy] | tuple[None, None, ActionEntropy]
    ):
        """Forward pass of the network.

        :param latent: Latent space representation.
        :type latent: LatentTensor
        :param action_mask: Mask to apply to the logits. Defaults to None.
        :type action_mask: ActionMaskInput
        :param sample: Whether to sample an action or return the mode/mean. Defaults to True.
        :type sample: bool
        :return: Action, log probability of the action, and entropy of the distribution.
        :rtype: tuple[SampledAction, LogProbs, ActionEntropy] | tuple[None, None, ActionEntropy]
        """
        logits = self.wrapped(latent)

        if action_mask is not None:
            action_mask = torch.as_tensor(
                numpy_action_mask(action_mask),
                device=self.device,
                dtype=torch.bool,
            )
            logits = self.apply_mask(logits, action_mask)

        # Distribution from logits
        self.dist = self.get_distribution(logits)

        # Sample action, compute log probability and entropy
        if sample:
            action = self.dist.sample()
            log_prob = self.dist.log_prob(action)
            return action, log_prob, self.dist.entropy()

        return None, None, self.dist.entropy()

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
