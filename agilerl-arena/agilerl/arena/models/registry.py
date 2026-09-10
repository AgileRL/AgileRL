# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Algorithm name registry for the manifest's ``algorithm`` section."""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from agilerl.arena.models.algorithms.base import AlgoSpec

logger = logging.getLogger(__name__)

AlgoSpecT = TypeVar("AlgoSpecT", bound="AlgoSpec")


class AgentType(Enum):
    """Enumeration of supported agent types."""

    SingleAgent = "single_agent"
    MultiAgent = "multi_agent"
    LLMAgent = "llm_agent"
    OfflineAgent = "offline_agent"
    SupervisedAgent = "supervised_agent"
    LatentPPOAgent = "latent_ppo_agent"
    BanditAgent = "bandit_agent"


class AlgorithmRegistry:
    """Maps the manifest's ``algorithm.name`` to the spec class that validates it."""

    def __init__(self) -> None:
        self._entries: dict[str, type[AlgoSpec]] = {}

    def add(self, name: str, spec_cls: type[AlgoSpec]) -> None:
        """Register *spec_cls* under *name*.

        :param name: Algorithm name (e.g. ``"DQN"``).
        :type name: str
        :param spec_cls: The spec class to register.
        :type spec_cls: type[AlgoSpec]
        """
        if name in self._entries:
            logger.warning("Overriding existing registration for algorithm %r", name)
        self._entries[name] = spec_cls

    def get(self, name: str) -> type[AlgoSpec]:
        """Look up a spec class by algorithm name.

        :param name: Algorithm name.
        :type name: str
        :returns: The registered spec class.
        :rtype: type[AlgoSpec]
        :raises KeyError: If *name* is not registered.
        """
        try:
            return self._entries[name]
        except KeyError as err:
            supported = ", ".join(sorted(self._entries))
            msg = f"No registry entry for algorithm {name!r}. Registered: {supported}"
            raise KeyError(msg) from err

    def create(self, name: str, /, **fields: Any) -> AlgoSpec:
        """Build a spec, applying fields the name implies.

        :param name: Algorithm name.
        :type name: str
        :param fields: Field values for the spec.
        :returns: The spec instance.
        :rtype: AlgoSpec
        :raises KeyError: If *name* is not registered.
        """
        spec_cls = self.get(name)
        implied = spec_cls.alias_implies.get(name, {})
        for field, value in implied.items():
            if field in fields and fields[field] != value:
                msg = (
                    f"algorithm.name {name!r} implies {field}={value!r}, "
                    f"but the manifest sets {field}={fields[field]!r}."
                )
                raise ValueError(msg)
            fields[field] = value
        return spec_cls(**fields)

    def names(self) -> list[str]:
        """Return every registered algorithm name, sorted.

        :returns: The registered algorithm names.
        :rtype: list[str]
        """
        return sorted(self._entries)

    def items(self) -> list[tuple[str, type[AlgoSpec]]]:
        """Return every ``(name, spec class)`` pair, sorted by name.

        :returns: The registered entries.
        :rtype: list[tuple[str, type[AlgoSpec]]]
        """
        return [(name, self._entries[name]) for name in self.names()]


MANIFEST_REGISTRY = AlgorithmRegistry()


def register(name: str | None = None) -> Callable[[type[AlgoSpecT]], type[AlgoSpecT]]:
    """Register an algorithm spec under *name*, or its class name minus ``Spec``.

    :param name: Explicit registry key; defaults to ``DQNSpec`` -> ``"DQN"``.
    :type name: str | None
    :returns: The class decorator.
    :rtype: Callable[[type[AlgoSpecT]], type[AlgoSpecT]]
    """

    def decorator(spec_cls: type[AlgoSpecT]) -> type[AlgoSpecT]:
        MANIFEST_REGISTRY.add(name or spec_cls.__name__.removesuffix("Spec"), spec_cls)
        return spec_cls

    return decorator
