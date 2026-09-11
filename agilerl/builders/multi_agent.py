# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Builder for multi-agent reinforcement learning algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agilerl.algorithms.core import MultiAgentAlgorithm
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


class MultiAgentBuilder(AlgorithmBuilder):
    """Multi-agent reinforcement learning."""

    @classmethod
    def algo_class(cls, spec: AlgorithmSpec) -> type[MultiAgentAlgorithm]:
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
        spec: AlgorithmSpec,
        observation_spaces: dict[str, spaces.Space] | None = None,
        action_spaces: dict[str, spaces.Space] | None = None,
        *,
        runtime: AlgorithmBuildRuntime | None = None,
        **networks: Any,
    ) -> MultiAgentAlgorithm:
        """Build a multi-agent algorithm.

        :param spec: The algorithm spec.
        :type spec: AlgorithmSpec
        :param observation_spaces: Per-agent observation spaces.
        :type observation_spaces: dict[str, spaces.Space] | None
        :param action_spaces: Per-agent action spaces.
        :type action_spaces: dict[str, spaces.Space] | None
        :param runtime: Population slot, device, HPO, and optional checkpoint.
        :type runtime: AlgorithmBuildRuntime | None
        :param networks: Pre-built modules to hand the constructor, e.g.
            ``actor_networks`` and ``critic_networks``.
        :type networks: ModuleDict
        :returns: Multi-agent algorithm instance.
        :rtype: MultiAgentAlgorithm
        :raises ValueError: If observation_spaces, action_spaces, or index is None.
        """
        runtime = runtime or AlgorithmBuildRuntime()
        index = runtime.index
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
