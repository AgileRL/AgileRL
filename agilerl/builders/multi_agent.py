# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Builder for multi-agent reinforcement learning algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agilerl.algorithms.core import MultiAgentAlgorithm
from agilerl.builders.base import (
    AlgorithmBuilder,
    apply_checkpoint,
    constructor_kwargs,
    spec_kwargs,
)

if TYPE_CHECKING:
    import torch
    from accelerate import Accelerator
    from gymnasium import spaces

    from agilerl.algorithms.core.registry import HyperparameterConfig
    from agilerl.arena.models.algorithms import AlgoSpec
else:
    HyperparameterConfig = Any


class MultiAgentBuilder(AlgorithmBuilder):
    """Multi-agent reinforcement learning."""

    @classmethod
    def algo_class(cls, spec: AlgoSpec) -> type[MultiAgentAlgorithm]:
        resolved = super().algo_class(spec)
        if not issubclass(resolved, MultiAgentAlgorithm):
            msg = (
                f"{type(spec).__name__} resolved to {resolved.__name__}, "
                "which is not a subclass of MultiAgentAlgorithm."
            )
            raise TypeError(msg)
        return resolved

    @classmethod
    def build(
        cls,
        spec: AlgoSpec,
        observation_spaces: dict[str, spaces.Space] | None = None,
        action_spaces: dict[str, spaces.Space] | None = None,
        *,
        index: int | None = None,
        resume_from_checkpoint: str | None = None,
        load_weights_from: str | None = None,
        device: str | torch.device = "cpu",
        accelerator: Accelerator | None = None,
        hp_config: HyperparameterConfig | None = None,
        **networks: Any,
    ) -> MultiAgentAlgorithm:
        """Build a multi-agent algorithm.

        :param spec: The algorithm spec.
        :type spec: AlgoSpec
        :param observation_spaces: Per-agent observation spaces.
        :type observation_spaces: dict[str, spaces.Space] | None
        :param action_spaces: Per-agent action spaces.
        :type action_spaces: dict[str, spaces.Space] | None
        :param index: Index of the agent in the population.
        :type index: int | None
        :param resume_from_checkpoint: Checkpoint to continue an interrupted run
            from, restoring optimizer state and the hyperparameters it belongs to.
            Mutually exclusive with ``load_weights_from``.
        :type resume_from_checkpoint: str | None
        :param load_weights_from: Checkpoint to warm-start a new run from, taking
            only the weights. Mutually exclusive with ``resume_from_checkpoint``.
        :type load_weights_from: str | None
        :param device: Torch device. Defaults to "cpu".
        :type device: str | torch.device
        :param accelerator: Accelerator object for distributed computing.
        :type accelerator: Accelerator | None
        :param hp_config: Resolved hyperparameter config for HPO.
        :type hp_config: HyperparameterConfig | None
        :param networks: Pre-built modules to hand the constructor, e.g.
            ``actor_networks`` and ``critic_networks``.
        :type networks: ModuleDict
        :returns: Multi-agent algorithm instance.
        :rtype: MultiAgentAlgorithm
        :raises ValueError: If observation_spaces, action_spaces, or index is None.
        """
        if observation_spaces is None or action_spaces is None or index is None:
            msg = (
                "MultiAgentBuilder.build requires observation_spaces, "
                "action_spaces, and index."
            )
            raise ValueError(msg)
        algo_cls = cls.algo_class(spec)
        algo = algo_cls(
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            index=index,
            device=device,
            accelerator=accelerator,
            **constructor_kwargs(algo_cls, spec_kwargs(spec, hp_config=hp_config)),
            **networks,
        )
        apply_checkpoint(algo, resume_from_checkpoint, load_weights_from, index=index)
        return algo
