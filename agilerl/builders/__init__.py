# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Paradigm-keyed algorithm builders. Look up a spec's builder with :func:`select_builder`."""

from __future__ import annotations

from agilerl.arena.models.algorithms import (
    AlgorithmSpec,
    LLMAlgorithmSpec,
    MultiAgentAlgorithmSpec,
    SingleAgentAlgorithmSpec,
)
from agilerl.builders.base import AlgorithmBuilder, AlgorithmBuildRuntime
from agilerl.builders.llm import LLMBuilder
from agilerl.builders.multi_agent import MultiAgentBuilder
from agilerl.builders.single_agent import SingleAgentBuilder


def select_builder(spec: AlgorithmSpec) -> type[AlgorithmBuilder]:
    """Return the builder class for *spec*'s paradigm.

    :param spec: The algorithm spec.
    :type spec: AlgorithmSpec
    :returns: The paradigm's builder class.
    :rtype: type[AlgorithmBuilder]
    :raises TypeError: If *spec* is not one of the contract's algorithm specs.
    """
    if isinstance(spec, LLMAlgorithmSpec):
        return LLMBuilder
    if isinstance(spec, MultiAgentAlgorithmSpec):
        return MultiAgentBuilder
    if isinstance(spec, SingleAgentAlgorithmSpec):
        return SingleAgentBuilder
    msg = f"{type(spec).__name__} is not an algorithm spec."
    raise TypeError(msg)


__all__ = [
    "AlgorithmBuildRuntime",
    "AlgorithmBuilder",
    "LLMBuilder",
    "MultiAgentBuilder",
    "SingleAgentBuilder",
    "select_builder",
]
