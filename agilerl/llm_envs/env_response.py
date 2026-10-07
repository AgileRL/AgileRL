# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""``EnvResponse``: the per-episode observation payload a rollout engine consumes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EnvResponse:
    """Standardized response payload returned by token-observation env workers.

    ``terminated`` and ``truncated`` follow gymnasium: the env ended the episode,
    or a limit (turns, context length) cut it off.
    """

    episode_id: str
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    @property
    def done(self) -> bool:
        """Whether the episode is over, terminated or truncated."""
        return self.terminated or self.truncated
