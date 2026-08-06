# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Environments for LLM training.

The OpenEnv-backed transports (``OpenEnvServer``, ``RemoteEnvClient``,
``InProcessEnvClient``, ...) live in :mod:`agilerl.llm_envs.openenv`, which requires
the ``llm`` extra; import them from there. Only the base-safe names are re-exported
here so ``import agilerl.llm_envs`` works without that extra.
"""

from agilerl.llm_envs.dataset_env import DatasetEnv
from agilerl.llm_envs.rollout import (
    RolloutCollector,
    RolloutHarness,
    TaskAssigner,
    process_observation,
)
from agilerl.utils.llm_utils import apply_chat_template

__all__ = [
    "DatasetEnv",
    "RolloutCollector",
    "RolloutHarness",
    "TaskAssigner",
    "apply_chat_template",
    "process_observation",
]
