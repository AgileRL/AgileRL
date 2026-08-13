# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from agilerl.arena import AgentType
from agilerl.arena.models.env import LLMEnvType
from agilerl.arena.models.hpo import RLHyperparameter
from agilerl.arena.models.networks import LoraConfigDict

logger = logging.getLogger(__name__)

# TypeVar over the AlgoSpec union so registration decorators return the concrete spec subclass.
AlgoSpecT = TypeVar("AlgoSpecT", bound="AlgoSpec")


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    """A single entry in the algorithm registry.

    :param spec_cls: The algorithm spec class.
    :type spec_cls: type[AlgoSpec]
    """

    spec_cls: type[AlgoSpec]


class AlgorithmRegistry:
    """Central registry mapping algorithm names to their spec classes.

    Populated at import time by the :func:`register` decorator applied to
    each concrete :class:`AlgorithmSpec` subclass.
    """

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}

    def add(self, name: str, spec_cls: type[AlgoSpec]) -> None:
        """Register a spec class under *name*.

        :param name: Algorithm name (e.g. ``"DQN"``).
        :type name: str
        :param spec_cls: The spec class to register.
        :type spec_cls: type[AlgoSpec]
        """
        if name in self._entries:
            logger.warning("Overriding existing registration for algorithm %r", name)

        self._entries[name] = RegistryEntry(spec_cls=spec_cls)

    def get(self, name: str) -> RegistryEntry:
        """Look up an entry by algorithm name.

        :param name: Algorithm name.
        :type name: str
        :returns: The registry entry.
        :rtype: RegistryEntry
        :raises KeyError: If *name* is not registered.
        """
        try:
            return self._entries[name]
        except KeyError as err:
            supported = ", ".join(sorted(self._entries))
            msg = f"No registry entry for algorithm {name!r}. Registered: {supported}"
            raise KeyError(msg) from err


ARENA_REGISTRY = AlgorithmRegistry()


def register() -> Callable[[type[AlgoSpecT]], type[AlgoSpecT]]:
    """Class decorator that registers an algorithm spec for Arena.

    The registry key is derived from the spec class name by stripping
    the ``"Spec"`` suffix (e.g. ``DQNSpec`` -> ``"DQN"``).

    :returns: The decorator function.
    :rtype: Callable[[type[AlgoSpecT]], type[AlgoSpecT]]

    Example::

        @register()
        class DQNSpec(RLAlgorithmSpec):
            ...
    """

    def decorator(spec_cls: type[AlgoSpecT]) -> type[AlgoSpecT]:
        name = spec_cls.__name__.removesuffix("Spec")
        ARENA_REGISTRY.add(name, spec_cls)
        return spec_cls

    return decorator


class AlgorithmSpec(BaseModel):
    """Base specification for all algorithms.

    Defines common fields and behavior for algorithm specifications, including
    batch size and hyperparameter configuration.

    """

    batch_size: int | None = Field(default=None, ge=1)
    hp_config: dict[str, RLHyperparameter] | None = None

    default_evo_steps: ClassVar[int] = 10_000

    agent_type: ClassVar[AgentType]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return self.__class__.__name__.removesuffix("Spec")


class RLAlgorithmSpec(AlgorithmSpec):
    """Specification for single-agent reinforcement learning algorithms.

    Extends :class:`AlgorithmSpec` with single-agent specific fields like
    network configuration, learning step frequency, and discount factor.
    """

    learn_step: int = Field(default=5, ge=1)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)

    agent_type: ClassVar[AgentType] = AgentType.SingleAgent


class MultiAgentRLAlgorithmSpec(AlgorithmSpec):
    """Specification for multi-agent reinforcement learning algorithms.

    Extends :class:`AlgorithmSpec` with multi-agent specific fields and
    support for multiple observation/action spaces and agent IDs.
    """

    learn_step: int = Field(default=2048, ge=1)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    torch_compiler: str | None = Field(default=None)

    agent_type: ClassVar[AgentType] = AgentType.MultiAgent


class LLMAlgorithmSpec(AlgorithmSpec):
    """Specification for LLM fine-tuning algorithms.

    Extends :class:`AlgorithmSpec` with LLM-specific fields including LoRA
    configuration, model parameters, and training hyperparameters.

    Subclasses must set the :attr:`env_type` class variable to indicate
    which LLM gym type the algorithm requires (``"reasoning"``,
    ``"preference"``, ``"sft"``, ``"multiturn"``).
    """

    beta: float | None = Field(default=None, ge=0.0, le=1.0)
    max_grad_norm: float = Field(default=0.1, ge=0.0)
    update_epochs: int = Field(default=1, ge=1)
    reduce_memory_peak: bool = Field(default=False)
    use_separate_reference_adapter: bool | None = Field(default=None)
    calc_position_embeddings: bool = Field(default=True)
    gradient_checkpointing: bool = Field(default=True)
    use_liger_loss: bool | None = Field(default=None)
    seed: int = Field(default=42)

    # These fields come from the "network" section of the manifest
    pretrained_model_name_or_path: str | None = Field(default=None, min_length=1)
    max_model_len: int = Field(default=1024, ge=1)
    lora_config: LoraConfigDict | None = Field(default=None)

    agent_type: ClassVar[AgentType] = AgentType.LLMAgent
    default_evo_steps: ClassVar[int] = 5
    env_type: ClassVar[LLMEnvType]


AlgoSpec = RLAlgorithmSpec | MultiAgentRLAlgorithmSpec | LLMAlgorithmSpec
