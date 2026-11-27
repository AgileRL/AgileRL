# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The AgileRL training manifest contract."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from agilerl.models.algorithms import (
    AlgorithmSpec,
    AlgoSpec,
    CISPOSpec,
    DPOSpec,
    GRPOSpec,
    GSPOSpec,
    LLMAlgorithmSpec,
    LLMPPOSpec,
    LLMREINFORCESpec,
    OPDSpec,
    RolloutLLMSpec,
    SFTSpec,
)
from agilerl.models.env import (
    EnvContainerSpec,
    EnvHostVolume,
    EnvServiceSpec,
    EnvSpec,
    JudgeSpec,
    LLMEnvSpec,
    LLMEnvType,
)
from agilerl.models.hpo import (
    MultiFrequencySelectionSpec,
    MutationProbabilities,
    MutationSpec,
    RLHyperparameter,
    SelectionStrategySpec,
    TournamentSelectionSpec,
    mutation_ceiling,
    resolve_frequency_ratios,
    resolve_selection_brackets,
)
from agilerl.models.manifest import (
    API_VERSION,
    TrainingManifest,
)
from agilerl.models.model_info import (
    SUPPORTED_MODEL_INFO,
    ModelArchitecture,
    ModelInfo,
    ModelStatus,
)
from agilerl.models.networks import (
    CosineLRScheduleConfig,
    FinetuningNetworkSpec,
    LoraConfigDict,
    VLLMConfig,
    default_rollout_vllm_config,
)
from agilerl.models.registry import MANIFEST_REGISTRY, AgentType, register
from agilerl.models.schema import SCHEMA_ID, manifest_schema
from agilerl.models.training import (
    BufferSpec,
    CheckpointExportSpec,
    LLMRolloutBufferSpec,
    RolloutMode,
    RolloutVersionStamp,
    TrainingSpec,
)

try:
    __version__ = _distribution_version("agilerl")
except PackageNotFoundError:  # running straight from a source tree
    __version__ = "0+unknown"

__all__ = [
    "API_VERSION",
    "MANIFEST_REGISTRY",
    "SCHEMA_ID",
    "SUPPORTED_MODEL_INFO",
    "AgentType",
    "AlgoSpec",
    "AlgorithmSpec",
    "BufferSpec",
    "CISPOSpec",
    "CheckpointExportSpec",
    "CosineLRScheduleConfig",
    "DPOSpec",
    "EnvContainerSpec",
    "EnvHostVolume",
    "EnvServiceSpec",
    "EnvSpec",
    "FinetuningNetworkSpec",
    "GRPOSpec",
    "GSPOSpec",
    "JudgeSpec",
    "LLMAlgorithmSpec",
    "LLMEnvSpec",
    "LLMEnvType",
    "LLMPPOSpec",
    "LLMREINFORCESpec",
    "LLMRolloutBufferSpec",
    "LoraConfigDict",
    "ModelArchitecture",
    "ModelInfo",
    "ModelStatus",
    "MultiFrequencySelectionSpec",
    "MutationProbabilities",
    "MutationSpec",
    "OPDSpec",
    "RLHyperparameter",
    "RolloutLLMSpec",
    "RolloutMode",
    "RolloutVersionStamp",
    "SFTSpec",
    "SelectionStrategySpec",
    "TournamentSelectionSpec",
    "TrainingManifest",
    "TrainingSpec",
    "VLLMConfig",
    "default_rollout_vllm_config",
    "manifest_schema",
    "mutation_ceiling",
    "register",
    "resolve_frequency_ratios",
    "resolve_selection_brackets",
]
