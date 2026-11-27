# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Requests and reply of Arena's memory estimate and solve endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agilerl.exceptions import ArenaAPIError

MEMORY_ESTIMATE_PATH = "/api/cli/v1/memory/estimate"
MEMORY_SOLVE_PATH = "/api/cli/v1/memory/solve"
# Least timeout for a memory request: Arena gives arena-service 60 s to size a manifest.
MEMORY_ESTIMATE_TIMEOUT_S = 90

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_OVER_BUDGET = 3


@dataclass(frozen=True)
class MemoryEstimate:
    """Arena's verdict on a manifest: the exit code and the ``--json`` report."""

    exit_code: int
    report: dict[str, Any]

    @property
    def usage_error(self) -> str:
        """Why Arena could not size the manifest; Arena sends it with ``exit_code`` 2."""
        return self.report["error"]


def memory_estimate_body(
    manifest: dict[str, Any],
    gpu: str | None = None,
    device_gb: float | None = None,
    num_gpus: int | None = None,
    gpus_per_node: int | None = None,
) -> dict[str, Any]:
    """The request body, without unset device fields.

    :param manifest: The raw manifest document.
    :param gpu: GPU type or name.
    :param device_gb: GPU memory in GiB.
    :param num_gpus: GPUs in the allocation.
    :param gpus_per_node: GPUs on one node of the allocation.
    :return: The JSON body.
    """
    device = {
        "gpu": gpu,
        "device_gb": device_gb,
        "num_gpus": num_gpus,
        "gpus_per_node": gpus_per_node,
    }
    return {"manifest": manifest} | {k: v for k, v in device.items() if v is not None}


def memory_solve_body(
    manifest: dict[str, Any],
    field: str,
    gpu: str | None = None,
    device_gb: float | None = None,
    num_gpus: int | None = None,
    hi: int | None = None,
) -> dict[str, Any]:
    """The solve request body, without unset device fields.

    :param manifest: The raw manifest document.
    :param field: The setting to solve.
    :param gpu: GPU type or name.
    :param device_gb: GPU memory in GiB.
    :param num_gpus: GPUs on the node.
    :param hi: Search ceiling.
    :return: The JSON body.
    """
    options = {"gpu": gpu, "device_gb": device_gb, "num_gpus": num_gpus, "hi": hi}
    return {"manifest": manifest, "field": field} | {
        k: v for k, v in options.items() if v is not None
    }


def parse_memory_estimate(data: object, status_code: int) -> MemoryEstimate:
    """Read the reply.

    :param data: The decoded body.
    :param status_code: HTTP status, for the error.
    :return: The verdict.
    :raises ArenaAPIError: The body is not ``{"exit_code": 0 | 2 | 3, "report": {...}}``.
    """
    if (
        isinstance(data, dict)
        and data.get("exit_code") in (EXIT_OK, EXIT_USAGE, EXIT_OVER_BUDGET)
        and isinstance(data.get("report"), dict)
    ):
        return MemoryEstimate(data["exit_code"], data["report"])
    msg = f"Arena returned an unexpected memory estimate: {str(data)[:200]}"
    raise ArenaAPIError(msg, status_code=status_code)
