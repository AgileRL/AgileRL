# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Size a Snowflake run's manifest with Arena's memory estimate."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import httpx

from agilerl.arena.client_version import (
    CLIENT_VERSION_HEADER,
    SNOWFLAKE_UPGRADE_COMMAND,
    is_upgrade_required,
    print_update_notice,
    upgrade_required_message,
)
from agilerl.arena.estimator_api import (
    EXIT_OK,
    EXIT_USAGE,
    MEMORY_ESTIMATE_PATH,
    MEMORY_ESTIMATE_TIMEOUT_S,
    memory_estimate_body,
    parse_memory_estimate,
)
from agilerl.arena.exceptions import ArenaAPIError
from agilerl.arena.models import __version__
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.snowflake.license import SnowflakeUpgradeRequiredError
from agilerl.arena.snowflake.profile import GpuPool


def manifest_fits(
    manifest_path: Path,
    pool: GpuPool,
    gpus: int,
    base_url: str,
    headers: Mapping[str, str],
) -> bool:
    """Whether Arena's memory estimate says the manifest fits the run's GPUs.

    :param manifest_path: Training manifest.
    :param pool: The family's pool, for per-GPU memory and GPUs per node.
    :param gpus: GPUs for the whole run, a multiple of ``pool.gpus_per_node``.
        The trainer lays the run out on ``gpus // pool.gpus_per_node`` nodes.
    :param base_url: Arena origin.
    :param headers: Authorization headers for Arena.
    :return: Whether both phases fit with the underprediction buffer.
    :raises SnowflakeUpgradeRequiredError: Arena refuses this agilerl-arena.
    :raises ValueError: Arena cannot be reached, rejects the request, or
        cannot size the manifest.
    """
    origin = base_url.rstrip("/")
    body = memory_estimate_body(
        TrainingManifest.load(str(manifest_path)),
        device_gb=float(pool.gpu_memory_gib),
        num_gpus=gpus,
        gpus_per_node=pool.gpus_per_node,
    )
    try:
        with httpx.Client() as client:
            response = client.post(
                f"{origin}{MEMORY_ESTIMATE_PATH}",
                headers={**headers, CLIENT_VERSION_HEADER: __version__},
                json=body,
                timeout=MEMORY_ESTIMATE_TIMEOUT_S,
            )
    except httpx.HTTPError as exc:
        msg = f"Could not reach Arena at {origin} for the memory estimate: {exc}"
        raise ValueError(msg) from exc
    print_update_notice(response, SNOWFLAKE_UPGRADE_COMMAND)
    if is_upgrade_required(response.status_code, response.text):
        raise SnowflakeUpgradeRequiredError(upgrade_required_message(response.text))
    if response.status_code in {401, 403}:
        msg = (
            f"Arena at {origin} rejected the credentials for the memory "
            f"estimate (HTTP {response.status_code})."
        )
        raise ValueError(msg)
    if not response.is_success:
        detail = ArenaAPIError.from_response_body(
            response.text, status_code=response.status_code
        ).detail
        msg = f"Arena could not estimate the manifest's memory: {detail}"
        raise ValueError(msg)
    try:
        data = response.json()
    except ValueError as exc:
        msg = f"Arena at {origin} returned a memory estimate that is not JSON."
        raise ValueError(msg) from exc
    try:
        estimate = parse_memory_estimate(data, response.status_code)
    except ArenaAPIError as exc:
        raise ValueError(exc.detail) from exc
    if estimate.exit_code == EXIT_USAGE:
        raise ValueError(
            estimate.usage_error or "the memory estimator could not size this manifest"
        )
    return estimate.exit_code == EXIT_OK
