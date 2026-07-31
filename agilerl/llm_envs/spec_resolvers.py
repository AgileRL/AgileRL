# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Env spec resolvers: map spec strings (GEM registry ids, ...) to env factories.

A leaf module with no optional dependencies, so both the client side
(:meth:`RolloutEnv.from_spec`) and the server side (:func:`resolve_env`) can
consult one registry.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from agilerl.protocols import TextEnvProtocol


def register_env_spec_resolver(
    name: str,
    resolver: Callable[[str], Callable[..., TextEnvProtocol] | None],
) -> None:
    """Register a resolver mapping an env spec string to an env factory.

    Consulted for non-URL specs before entrypoint loading; re-registering a
    ``name`` replaces its resolver.

    :param name: Registry key naming the resolver (for replacement/removal).
    :param resolver: ``spec -> factory | None`` (``None`` = unclaimed).
    """
    _ENV_SPEC_RESOLVERS[name] = resolver


def _gem_spec_resolver(spec: str) -> Callable[..., TextEnvProtocol] | None:
    """Claim `GEM registry ids`_ (e.g. ``game:GuessTheNumber-v0``) via ``gem.make``.

    .. _GEM registry ids: https://github.com/axon-rl/gem
    """
    try:
        import gem
        import gem.envs  # populates the registry
        from gem.envs.registration import (
            ENV_REGISTRY,
        )
    except ImportError:
        return None
    if spec in ENV_REGISTRY:
        return partial(gem.make, spec)
    return None


_ENV_SPEC_RESOLVERS: dict[str, Any] = {"gem": _gem_spec_resolver}


def resolve_spec_factory(spec: str) -> Callable[..., TextEnvProtocol] | None:
    """First registered resolver's factory claiming ``spec``, or ``None``."""
    for resolver in _ENV_SPEC_RESOLVERS.values():
        factory = resolver(spec)
        if factory is not None:
            return factory
    return None


def is_url(spec: str) -> bool:
    """Whether ``spec`` is an HTTP(S) URL (already hosted) rather than an env to load."""
    return isinstance(spec, str) and spec.startswith(("http://", "https://"))


def spec_to_factory(spec: str) -> Callable[..., TextEnvProtocol]:
    """Resolve a non-URL env spec string to an env factory.

    Registered resolvers are consulted first; the fallback loads a
    ``module:Class`` / ``path.py:Class`` entrypoint.
    """
    factory = resolve_spec_factory(spec)
    if factory is not None:
        return factory
    if ":" not in spec:
        msg = (
            f"env spec {spec!r} is neither a URL, a resolver-claimed id, nor a "
            "'module:Class' / 'path.py:Class' entrypoint"
        )
        raise ValueError(msg)
    # Lazy: entrypoint loading drags the gym/pettingzoo closure, which a
    # slim host serving a resolver-claimed id never needs.
    from agilerl.utils.env_utils import resolve_entrypoint_target

    return resolve_entrypoint_target(spec)
