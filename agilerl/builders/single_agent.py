# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Builder for single-agent reinforcement learning algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agilerl.algorithms.core import RLAlgorithm
from agilerl.arena.models.algorithms import AlgorithmSpec
from agilerl.builders.base import (
    AlgorithmBuilder,
    AlgorithmBuildRuntime,
    apply_checkpoint,
    constructor_kwargs,
    spec_kwargs,
)

if TYPE_CHECKING:
    from gymnasium import spaces


class SingleAgentBuilder(AlgorithmBuilder):
    """Single-agent reinforcement learning."""

    @classmethod
    def algo_class(cls, spec: AlgorithmSpec) -> type[RLAlgorithm]:
        resolved = super().algo_class(spec)
        if not issubclass(resolved, RLAlgorithm):
            msg = (
                f"{type(spec).__name__} resolved to {resolved.__name__}, "
                "which is not a subclass of RLAlgorithm."
            )
            raise TypeError(msg)
        return resolved

    @classmethod
    def build(
        cls,
        spec: AlgorithmSpec,
        observation_space: spaces.Space | None = None,
        action_space: spaces.Space | None = None,
        *,
        runtime: AlgorithmBuildRuntime | None = None,
        **networks: Any,
    ) -> RLAlgorithm:
        """Build a single-agent algorithm.

        :param spec: The algorithm spec.
        :type spec: AlgorithmSpec
        :param observation_space: Observation space.
        :type observation_space: spaces.Space | None
        :param action_space: Action space.
        :type action_space: spaces.Space | None
        :param runtime: Population slot, device, HPO, and optional checkpoint.
        :type runtime: AlgorithmBuildRuntime | None
        :param networks: Pre-built modules to hand the constructor, e.g.
            ``actor_network`` and ``critic_network``. Only pass the ones the
            algorithm takes.
        :type networks: EvolvableModule
        :returns: Single-agent algorithm instance.
        :rtype: RLAlgorithm
        :raises ValueError: If observation_space, action_space, or index is None.
        """
        runtime = runtime or AlgorithmBuildRuntime()
        index = runtime.index
        if observation_space is None or action_space is None or index is None:
            msg = "SingleAgentBuilder.build requires observation_space, action_space, and index."
            raise ValueError(msg)
        algo_cls = cls.algo_class(spec)
        algo = algo_cls(
            observation_space=observation_space,
            action_space=action_space,
            index=index,
            device=runtime.device,
            accelerator=runtime.accelerator,
            **constructor_kwargs(
                algo_cls, spec_kwargs(spec, hp_config=runtime.hp_config)
            ),
            **networks,
        )
        apply_checkpoint(
            algo,
            runtime.resume_from_checkpoint,
            runtime.load_weights_from,
            index=index,
        )
        return algo
