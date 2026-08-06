# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Name and resolve the source an LLM env comes from.

A leaf module with no optional dependencies, so the client side
(:meth:`RolloutHarness.from_spec`), the server side (:func:`resolve_env`) and any
orchestrator scheduling the env can share one answer to "what does this manifest
name?" rather than each reimplementing the branch.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any

    from agilerl.protocols import TextEnvProtocol


class EnvSource(str, Enum):
    """Where a rollout env's prompts come from."""

    DATASET = "dataset"
    ENTRYPOINT = "entrypoint"
    ENV_URL = "env_url"


def find_source(**sources: Any) -> str | None:
    """Return the name of the single source that is set, or ``None`` for zero or several.

    The one place the rule lives: a rollout env comes from exactly one source.
    Callers pass the sources they support, so an orchestrator offering more than
    the framework can build in-process gets the same rule over its own list, and
    keeps its own wording for the failure.

    :param sources: The candidate values, keyed by manifest field name.
    :returns: The name of the source that is set, or ``None``.
    :rtype: str | None
    """
    named = [field for field, value in sources.items() if value is not None]
    return named[0] if len(named) == 1 else None


def name_source(**sources: Any) -> str:
    """Return the single source that is set, rejecting zero or several.

    :param sources: The candidate values, keyed by manifest field name.
    :returns: The name of the source that is set.
    :rtype: str
    :raises ValueError: If the number of sources set is not exactly one.
    """
    found = find_source(**sources)
    if found is None:
        options = ", ".join(sources)
        msg = f"Exactly one of {options} is required for rollout environments."
        raise ValueError(msg)
    return found


def source_of(manifest: Mapping[str, Any]) -> EnvSource:
    """Name the source an env manifest declares, by its manifest keys.

    :param manifest: The manifest's ``environment`` section.
    :returns: The source it names.
    :rtype: EnvSource
    :raises ValueError: If it names zero or several.
    """
    return EnvSource(name_source(**{s.value: manifest.get(s.value) for s in EnvSource}))


def is_url(spec: str) -> bool:
    """Whether ``spec`` is an HTTP(S) URL (already hosted) rather than an env to load."""
    return isinstance(spec, str) and spec.startswith(("http://", "https://"))


def spec_to_factory(spec: str) -> Callable[..., TextEnvProtocol]:
    """Resolve a non-URL env spec to the callable that builds the env.

    ``spec`` is a ``module:attr`` / ``path.py:attr`` entrypoint naming any
    callable that returns a text env — a class, or a library's own factory
    (``gem:make``, say, with the env id passed through ``env_config``).
    """
    if ":" not in spec:
        msg = (
            f"env spec {spec!r} is neither a URL nor a "
            "'module:Class' / 'path.py:Class' entrypoint"
        )
        raise ValueError(msg)
    # Lazy: entrypoint loading drags the gym/pettingzoo closure, which a slim
    # host does not otherwise need.
    from agilerl.utils.env_utils import resolve_entrypoint_target

    return resolve_entrypoint_target(spec)
