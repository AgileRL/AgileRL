# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Builder for single-agent reinforcement learning algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agilerl.algorithms.core import SingleAgentAlgorithm
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


class SingleAgentBuilder(AlgorithmBuilder):
    """Single-agent reinforcement learning."""

    @classmethod
    def algo_class(cls, spec: AlgoSpec) -> type[SingleAgentAlgorithm]:
        resolved = super().algo_class(spec)
        if not issubclass(resolved, SingleAgentAlgorithm):
            msg = (
                f"{type(spec).__name__} resolved to {resolved.__name__}, "
                "which is not a subclass of SingleAgentAlgorithm."
            )
            raise TypeError(msg)
        return resolved

    @classmethod
    def build(
        cls,
        spec: AlgoSpec,
        observation_space: spaces.Space | None = None,
        action_space: spaces.Space | None = None,
        *,
        index: int | None = None,
        resume_from_checkpoint: str | None = None,
        load_weights_from: str | None = None,
        device: str | torch.device = "cpu",
        accelerator: Accelerator | None = None,
        hp_config: HyperparameterConfig | None = None,
        **networks: Any,
    ) -> SingleAgentAlgorithm:
        """Build a single-agent algorithm.

        :param spec: The algorithm spec.
        :type spec: AlgoSpec
        :param observation_space: Observation space.
        :type observation_space: spaces.Space | None
        :param action_space: Action space.
        :type action_space: spaces.Space | None
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
            ``actor_network`` and ``critic_network``. Only pass the ones the
            algorithm takes.
        :type networks: EvolvableModule
        :returns: Single-agent algorithm instance.
        :rtype: SingleAgentAlgorithm
        :raises ValueError: If observation_space, action_space, or index is None.
        """
        if observation_space is None or action_space is None or index is None:
            msg = "SingleAgentBuilder.build requires observation_space, action_space, and index."
            raise ValueError(msg)
        algo_cls = cls.algo_class(spec)
        algo = algo_cls(
            observation_space=observation_space,
            action_space=action_space,
            index=index,
            device=device,
            accelerator=accelerator,
            **constructor_kwargs(algo_cls, spec_kwargs(spec, hp_config=hp_config)),
            **networks,
        )
        apply_checkpoint(algo, resume_from_checkpoint, load_weights_from, index=index)
        return algo
