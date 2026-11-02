# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Environments for LLM training.

The OpenEnv-backed transports (``OpenEnvServer``, ``RemoteEnvClient``,
``InProcessEnvClient``, ...) live in :mod:`agilerl.llm_envs.openenv`, which requires
the ``llm`` extra; import them from there. Only the base-safe names are re-exported
here so ``import agilerl.llm_envs`` works without that extra.
"""

from agilerl.llm_envs.async_collector import AsyncBatchCollector
from agilerl.llm_envs.collector import RolloutCollector
from agilerl.llm_envs.dataset_env import DatasetEnv
from agilerl.llm_envs.env_response import EnvResponse
from agilerl.llm_envs.harness import RolloutHarness
from agilerl.llm_envs.observation import process_observation
from agilerl.llm_envs.task_assigner import TaskAssigner
from agilerl.llm_envs.tool_parsers import (
    Gemma4ToolParser,
    Granite3ToolParser,
    HarmonyToolParser,
    ParsedToolCall,
    ToolCallParseStatus,
    ToolCallTagParser,
    detect_tool_parser,
    get_tool_parser,
)
from agilerl.utils.llm_utils import apply_chat_template

__all__ = [
    "AsyncBatchCollector",
    "DatasetEnv",
    "EnvResponse",
    "Gemma4ToolParser",
    "Granite3ToolParser",
    "HarmonyToolParser",
    "ParsedToolCall",
    "RolloutCollector",
    "RolloutHarness",
    "TaskAssigner",
    "ToolCallParseStatus",
    "ToolCallTagParser",
    "apply_chat_template",
    "detect_tool_parser",
    "get_tool_parser",
    "process_observation",
]
