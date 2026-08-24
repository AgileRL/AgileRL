# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""``EnvResponse``: the per-episode observation payload a rollout engine consumes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EnvResponse:
    """Standardized response payload returned by token-observation env workers."""

    episode_id: str
    observation: Any
    reward: float
    done: bool
    info: dict[str, Any]
