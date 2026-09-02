# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Token-level rollout envs for generative LLM tasks over an env client.

``RolloutHarness`` runs the tokenisation + turn loop for one episode; ``RolloutCollector``
steps a batch of them in lock-step, a shared ``TaskAssigner`` giving each GRPO group
a common task (dataset row or reset seed).
"""

from agilerl.llm_envs.collector import RolloutCollector
from agilerl.llm_envs.harness import RolloutHarness
from agilerl.llm_envs.observation import (
    DEFAULT_OBSERVATION_ROLE,
    OBSERVATION_ROLES,
    observation_role,
    process_observation,
)
from agilerl.llm_envs.task_assigner import TaskAssigner
from agilerl.llm_envs.task_assigner import _mix_seed as _mix_seed

__all__ = [
    "DEFAULT_OBSERVATION_ROLE",
    "OBSERVATION_ROLES",
    "RolloutCollector",
    "RolloutHarness",
    "TaskAssigner",
    "observation_role",
    "process_observation",
]
