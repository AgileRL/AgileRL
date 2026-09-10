# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Paradigm-keyed training strategies. Look up a spec's strategy with :func:`select_strategy`."""

from __future__ import annotations

from agilerl.models.algo import (
    AlgoSpec,
    LLMAlgorithmSpec,
    MultiAgentRLAlgorithmSpec,
    RLAlgorithmSpec,
)
from agilerl.models.env_types import LLMEnvType
from agilerl.strategies.bandit import BanditStrategy
from agilerl.strategies.base import TrainingStrategy
from agilerl.strategies.llm import (
    LLMDatasetStrategy,
    LLMRolloutStrategy,
    LLMStrategy,
)
from agilerl.strategies.multi_agent import (
    MultiAgentOffPolicyStrategy,
    MultiAgentOnPolicyStrategy,
)
from agilerl.strategies.offline import OfflineStrategy
from agilerl.strategies.single_agent import (
    SingleAgentOffPolicyStrategy,
    SingleAgentOnPolicyStrategy,
)

SINGLE_AGENT_ON_POLICY = SingleAgentOnPolicyStrategy()
SINGLE_AGENT_OFF_POLICY = SingleAgentOffPolicyStrategy()
OFFLINE = OfflineStrategy()
BANDIT = BanditStrategy()
MULTI_AGENT_ON_POLICY = MultiAgentOnPolicyStrategy()
MULTI_AGENT_OFF_POLICY = MultiAgentOffPolicyStrategy()
LLM_ROLLOUT = LLMRolloutStrategy()
LLM_DATASET = LLMDatasetStrategy()


def select_strategy(spec: AlgoSpec) -> TrainingStrategy:
    """Return the strategy that trains *spec*, from its paradigm flags.

    The contract declares ``off_policy`` / ``offline`` / ``bandit`` on the RL
    specs and ``env_type`` on the LLM specs; a spec subclassed elsewhere
    inherits them, so it trains like its parent.

    :param spec: The algorithm spec.
    :type spec: AlgoSpec
    :returns: The paradigm's strategy.
    :rtype: TrainingStrategy
    :raises TypeError: If *spec* is not one of the contract's algorithm specs.
    :raises KeyError: If an LLM spec's ``env_type`` has no strategy.
    """
    if isinstance(spec, LLMAlgorithmSpec):
        env_type = getattr(spec, "env_type", None)
        if env_type == LLMEnvType.ROLLOUT:
            return LLM_ROLLOUT
        if env_type == LLMEnvType.DATASET:
            return LLM_DATASET
        msg = f"No training strategy for LLM env_type {env_type!r}."
        raise KeyError(msg)
    if isinstance(spec, MultiAgentRLAlgorithmSpec):
        return MULTI_AGENT_OFF_POLICY if spec.off_policy else MULTI_AGENT_ON_POLICY
    if isinstance(spec, RLAlgorithmSpec):
        if spec.bandit:
            return BANDIT
        if spec.offline:
            return OFFLINE
        return SINGLE_AGENT_OFF_POLICY if spec.off_policy else SINGLE_AGENT_ON_POLICY
    msg = f"{type(spec).__name__} is not an algorithm spec."
    raise TypeError(msg)


__all__ = [
    "BANDIT",
    "LLM_DATASET",
    "LLM_ROLLOUT",
    "MULTI_AGENT_OFF_POLICY",
    "MULTI_AGENT_ON_POLICY",
    "OFFLINE",
    "SINGLE_AGENT_OFF_POLICY",
    "SINGLE_AGENT_ON_POLICY",
    "BanditStrategy",
    "LLMDatasetStrategy",
    "LLMRolloutStrategy",
    "LLMStrategy",
    "MultiAgentOffPolicyStrategy",
    "MultiAgentOnPolicyStrategy",
    "OfflineStrategy",
    "SingleAgentOffPolicyStrategy",
    "SingleAgentOnPolicyStrategy",
    "TrainingStrategy",
    "select_strategy",
]
