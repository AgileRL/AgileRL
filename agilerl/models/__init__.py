# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import lazy_loader as lazy

from agilerl.arena.models.algorithms import (
    AlgoSpec,
    CISPOSpec,
    CQNSpec,
    DDPGSpec,
    DPOSpec,
    DQNSpec,
    GRPOSpec,
    GSPOSpec,
    IPPOSpec,
    LLMAlgorithmSpec,
    LLMPPOSpec,
    LLMREINFORCESpec,
    MADDPGSpec,
    MATD3Spec,
    MultiAgentAlgorithmSpec,
    NeuralTSSpec,
    NeuralUCBSpec,
    PPOSpec,
    RainbowDQNSpec,
    SFTSpec,
    SingleAgentAlgorithmSpec,
    TD3Spec,
)
from agilerl.arena.models.registry import MANIFEST_REGISTRY
from agilerl.models.hpo import (
    MultiFrequencySelectionSpec,
    MutationProbabilities,
    MutationSpec,
    NetworkMutationRanges,
    SelectionStrategySpec,
    TournamentSelectionSpec,
)
from agilerl.models.manifest import TrainingManifest
from agilerl.models.networks import (
    CnnSpec,
    ContinuousQNetworkSpec,
    DeterministicActorSpec,
    FinetuningNetworkSpec,
    LstmSpec,
    MlpSpec,
    MultiInputSpec,
    NetworkSpec,
    QNetworkSpec,
    RainbowQNetworkSpec,
    SimbaSpec,
    StochasticActorSpec,
    ValueNetworkSpec,
    normalize_manifest_network,
)
from agilerl.models.training import ReplayBufferSpec, TrainingSpec

# NOTE: env has heavy imports (gymnasium, pandas, datasets, pettingzoo)
# so we lazy-load it to keep imports from agilerl.models lightweight.
__getattr__, __dir__, _ = lazy.attach(
    __name__,
    submod_attrs={
        "env": [
            "BanditEnvSpec",
            "EnvSpec",
            "GymEnvSpec",
            "LLMEnvSpec",
            "LLMEnvType",
            "OfflineEnvSpec",
        ],
    },
)

if TYPE_CHECKING:
    from agilerl.models.env import (
        BanditEnvSpec,
        GymEnvSpec,
        LLMEnvSpec,
        OfflineEnvSpec,
    )

    EnvironmentSpecType = GymEnvSpec | OfflineEnvSpec | LLMEnvSpec | BanditEnvSpec
    ReplayBufferSpecType = ReplayBufferSpec | None
    TrainingSpecType = TrainingSpec | None
    MutationSpecType = MutationSpec | None

__all__ = [
    "MANIFEST_REGISTRY",
    "AlgoSpec",
    "CISPOSpec",
    "CQNSpec",
    "CnnSpec",
    "ContinuousQNetworkSpec",
    "DDPGSpec",
    "DPOSpec",
    "DQNSpec",
    "DeterministicActorSpec",
    "FinetuningNetworkSpec",
    "GRPOSpec",
    "GSPOSpec",
    "IPPOSpec",
    "LLMAlgorithmSpec",
    "LLMPPOSpec",
    "LLMREINFORCESpec",
    "LstmSpec",
    "MADDPGSpec",
    "MATD3Spec",
    "MlpSpec",
    "MultiAgentAlgorithmSpec",
    "MultiFrequencySelectionSpec",
    "MultiInputSpec",
    "MutationProbabilities",
    "MutationSpec",
    "NetworkMutationRanges",
    "NetworkSpec",
    "NeuralTSSpec",
    "NeuralUCBSpec",
    "PPOSpec",
    "QNetworkSpec",
    "RainbowDQNSpec",
    "RainbowQNetworkSpec",
    "ReplayBufferSpec",
    "SFTSpec",
    "SelectionStrategySpec",
    "SimbaSpec",
    "SingleAgentAlgorithmSpec",
    "StochasticActorSpec",
    "TD3Spec",
    "TournamentSelectionSpec",
    "TrainingManifest",
    "TrainingSpec",
    "ValueNetworkSpec",
    "normalize_manifest_network",
]
