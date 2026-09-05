# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Algorithm base classes (re-exported for the historical import path)."""

from agilerl.algorithms.core.evolvable_algorithm import (
    EvolvableAlgorithm,
    RLAlgorithm,
)
from agilerl.algorithms.core.evolvable_checkpoint import get_checkpoint_dict
from agilerl.algorithms.core.evolvable_helpers import (
    ClassicRLPopulationFactory,
    RegistryMeta,
    _is_readonly_property,
    _per_neuron_grad,
    build_classic_rl_population,
    get_optimizer_cls,
)
from agilerl.algorithms.core.llm_algorithm import LLMAlgorithm
from agilerl.algorithms.core.llm_vllm import (
    LLM,
    CompletionOutput,
    SamplingParams,
    _vllm_sampled_token_logprobs,
)
from agilerl.algorithms.core.multi_agent import MultiAgentRLAlgorithm
from agilerl.algorithms.core.optimizer_wrapper import OptimizerWrapper
from agilerl.typing import ActionResult

__all__ = [
    "LLM",
    "ActionResult",
    "ClassicRLPopulationFactory",
    "CompletionOutput",
    "EvolvableAlgorithm",
    "LLMAlgorithm",
    "MultiAgentRLAlgorithm",
    "OptimizerWrapper",
    "RLAlgorithm",
    "RegistryMeta",
    "SamplingParams",
    "_is_readonly_property",
    "_per_neuron_grad",
    "_vllm_sampled_token_logprobs",
    "build_classic_rl_population",
    "get_checkpoint_dict",
    "get_optimizer_cls",
]
