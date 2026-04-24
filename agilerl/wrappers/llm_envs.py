"""Compatibility exports for moved LLM environment modules."""

from __future__ import annotations

import warnings

from agilerl import llm_envs as _llm_envs
from agilerl.llm_envs import *  # noqa: F403

warnings.warn(
    (
        "Importing from agilerl.wrappers.llm_envs is deprecated and will be removed "
        "in a future release. Import from agilerl.llm_envs instead."
    ),
    FutureWarning,
    stacklevel=2,
)

requests = _llm_envs.requests
