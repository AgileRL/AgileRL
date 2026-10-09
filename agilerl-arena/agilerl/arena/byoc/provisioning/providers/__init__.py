# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Registry of cloud providers that can provision an Arena cluster."""

from __future__ import annotations

from agilerl.arena.byoc.provisioning.providers.base import CloudProvider

PROVIDER_NAMES = ("nebius", "aws")


def provider_names() -> tuple[str, ...]:
    """Return the cloud providers the CLI can provision."""
    return PROVIDER_NAMES


def get_provider(name: str) -> CloudProvider:
    """Return the cloud provider named *name*.

    :param name: Provider id from the cluster spec, such as ``nebius``.
    :return: The provider implementation.
    """
    if name == "nebius":
        # Cycle: nebius.provider imports terraform, which imports get_provider.
        from agilerl.arena.byoc.provisioning.providers.nebius.provider import NEBIUS

        return NEBIUS
    if name == "aws":
        # Cycle: aws.provider imports terraform, which imports get_provider.
        from agilerl.arena.byoc.provisioning.providers.aws.provider import AWS

        return AWS
    msg = f"Unsupported cluster provider {name!r}."
    raise ValueError(msg)
