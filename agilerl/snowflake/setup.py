# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GPU pool sizing checks and the messages that refuse a family or a GPU cap."""

from __future__ import annotations

import math

from agilerl.snowflake.catalog import (
    GpuFamily,
    family_sized,
)
from agilerl.snowflake.profile import (
    GPU_FAMILIES,
)

ORG_KEY_PREFIX = "arena_org_"


def nodes_for_gpus(max_gpus: int, gpus_per_node: int) -> int:
    """Nodes needed to hold ``max_gpus`` on a family.

    :param max_gpus: GPU cap.
    :param gpus_per_node: GPUs on one node of the family.
    :return: ``ceil(max_gpus / gpus_per_node)``.
    """
    return math.ceil(max_gpus / gpus_per_node)


def check_gpu_cap(family: str, max_gpus: int, gpus_per_node: int) -> None:
    """Refuse a GPU cap that is not whole nodes of the family.

    :param family: GPU instance family.
    :param max_gpus: GPU cap.
    :param gpus_per_node: GPUs on one node of the family.
    :raises ValueError: ``max_gpus`` is not a multiple of ``gpus_per_node``.
    """
    if max_gpus % gpus_per_node == 0:
        return
    msg = (
        f"{family} has {gpus_per_node} GPUs per node, so its cap is a multiple of "
        f"{gpus_per_node}: {gpus_per_node}, {2 * gpus_per_node}, "
        f"{3 * gpus_per_node} and so on. {max_gpus} is not.\n"
        "No pool was changed."
    )
    raise ValueError(msg)


def family_listed(family: str, families: list[GpuFamily]) -> bool:
    """Whether ``SHOW COMPUTE POOL INSTANCE FAMILIES`` returned ``family``.

    :param family: Requested instance family.
    :param families: Families in this region.
    :return: True when the name is present.
    """
    return any(item.name.upper() == family.upper() for item in families)


def format_family_lines(families: list[GpuFamily]) -> str:
    """One family per line, or ``(none)``.

    :param families: Families returned for this region.
    :return: The block used in the no-GPU message.
    """
    if not families:
        return "  (none)"
    return "\n".join(
        f"  {item.name}  {item.gpu_count}x {item.gpu or 'GPU'}" for item in families
    )


def no_gpu_message(
    family: str,
    families: list[GpuFamily],
    rerun: str,
) -> str:
    """Message when the requested family is not a GPU pool Arena can size in this region.

    :param family: Requested family. A key of :data:`GPU_FAMILIES`.
    :param families: Families the account returned.
    :param rerun: Command that retries with another family.
    :return: The message.
    """
    spec = GPU_FAMILIES[family]
    listed = [item for item in families if item.name.upper() == family]
    if listed:
        reason = (
            f"{family} is {listed[0].gpu_count}x {listed[0].gpu or spec.gpu} per node "
            f"in this region, not the {spec.gpus}x {spec.gpu} Arena sizes."
        )
    else:
        reason = f"This account has no {family} pool in this region ({spec.gpus}x {spec.gpu})."
    usable = [item for item in families if family_sized(item.name, families)]
    return (
        f"{reason} Arena did not create a compute pool.\n"
        "GPU families Arena can size in this region:\n"
        f"{format_family_lines(usable)}\n"
        "Rerun with one of those names:\n"
        f"  {rerun}\n"
        "A CPU pool cannot run this trainer.\n"
        "Nothing was created."
    )


def missing_family_message(family: str, role: str) -> str:
    """Message when a CPU family is not in this region.

    :param family: Requested instance family.
    :param role: ``broker`` or ``head``.
    :return: The message.
    """
    return (
        f"This account has no {family} instance family in this region.\n"
        f"The {role} pool was not created.\n"
        "Nothing was created."
    )


def unsupported_gpu_message(family: str) -> str:
    """Message when Arena has no GPU count for ``family``.

    :param family: Requested instance family.
    :return: The message.
    """
    return (
        f"{family} is not a GPU family Arena sizes.\n"
        f"Known families: {', '.join(GPU_FAMILIES)}.\n"
        "Nothing was created."
    )
