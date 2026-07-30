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

    :meth:`RolloutEnv.from_spec` and :func:`resolve_env` consult resolvers for any
    spec that is not a URL, before falling back to ``module:Class`` entrypoint
    loading. A resolver returns a factory (called with the spec's ``env_config``)
    for specs it claims, ``None`` otherwise. Re-registering a ``name`` replaces
    its resolver.

    :param name: Registry key naming the resolver (for replacement/removal).
    :param resolver: ``spec -> factory | None``.
    """
    _ENV_SPEC_RESOLVERS[name] = resolver


def _gem_spec_resolver(spec: str) -> Callable[..., TextEnvProtocol] | None:
    """Claim `GEM registry ids`_ (e.g. ``game:GuessTheNumber-v0``) via ``gem.make``.

    .. _GEM registry ids: https://github.com/axon-rl/gem
    """
    try:
        import gem  # ty: ignore[unresolved-import]
        import gem.envs  # ty: ignore[unresolved-import]  -- populates the registry
        from gem.envs.registration import (  # ty: ignore[unresolved-import]
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
