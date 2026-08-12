# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""HTTP clients and persistence for deployed Arena agents."""

from agilerl.arena.inference.agent import (
    Agent,
    AgentInfo,
    AgentMetadata,
    GenerateCompletion,
    GenerateParams,
    GenerateResult,
    LLMCompletionResult,
    LLMParams,
    LLMResults,
    PredictResult,
    SessionDetail,
    SessionInfo,
    SessionMessage,
    StatusResponse,
    deserialize,
    get_batch_size,
    serialize,
)
from agilerl.arena.inference.cache import (
    ActiveAgentSelection,
    load_active_agent,
    load_binding,
    normalized_deployment_name,
    save_active_agent,
    save_binding,
)
from agilerl.arena.inference.serde import RLData, SerializedRLData

__all__ = [
    "ActiveAgentSelection",
    "Agent",
    "AgentInfo",
    "AgentMetadata",
    "GenerateCompletion",
    "GenerateParams",
    "GenerateResult",
    "LLMCompletionResult",
    "LLMParams",
    "LLMResults",
    "PredictResult",
    "RLData",
    "SerializedRLData",
    "SessionDetail",
    "SessionInfo",
    "SessionMessage",
    "StatusResponse",
    "deserialize",
    "get_batch_size",
    "load_active_agent",
    "load_binding",
    "normalized_deployment_name",
    "save_active_agent",
    "save_binding",
    "serialize",
]
