# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Whether a Ray run generates on async rollout engines, and how many it starts."""

from __future__ import annotations

from typing import Any, Literal

from agilerl.arena.models.algorithms.base import AlgorithmSpec
from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec


def async_rollout_algorithm(algorithm: type[AlgorithmSpec]) -> bool:
    """Whether Ray runs this algorithm on async rollout engines.

    Ray runs every algorithm that generates its own rollouts async and ignores
    ``training.rollout_mode``. The other env type fails in Ray either way.

    :param algorithm: The manifest's algorithm spec class.
    :return: ``True`` for the rollout LLM algorithms.
    """
    return issubclass(algorithm, RolloutLLMSpec)


def run_rollout_engines(
    async_rollout: bool, configured: int | Literal["auto"]
) -> int | Literal["auto"]:
    """Rollout engines per member a Ray run starts: an async ``0`` is ``auto``.

    :param async_rollout: :func:`async_rollout_algorithm` of the run.
    :param configured: ``training.rollout_engines_per_agent`` as written.
    :return: ``configured``, or ``auto`` for an async run that leaves it 0.
    """
    if async_rollout and not configured:
        return "auto"
    return configured


def rollout_engines_blocker(
    async_rollout: bool, configured: int | Literal["auto"]
) -> str | None:
    """Why a run cannot start the configured rollout engines, or ``None``.

    :param async_rollout: :func:`async_rollout_algorithm` of the run.
    :param configured: ``training.rollout_engines_per_agent`` as written.
    :return: The error for engines on a run that does not generate, or ``None``.
    """
    if async_rollout or configured == 0:
        return None
    return f"rollout_engines_per_agent={configured} requires async rollout."


def apply_run_rollout_mode(training: dict[str, Any], async_rollout: bool) -> None:
    """Set a raw training section's rollout mode and engines as Ray runs them.

    :param training: The manifest's ``training`` section, edited in place.
    :param async_rollout: :func:`async_rollout_algorithm` of the run.
    """
    if not async_rollout:
        training.pop("rollout_mode", None)
        return
    training["rollout_mode"] = "async"
    training["rollout_engines_per_agent"] = run_rollout_engines(
        True, training.get("rollout_engines_per_agent", 0)
    )
