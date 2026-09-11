# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import lazy_loader as lazy

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.models.algo import (  # noqa: F401
    ALGO_REGISTRY,
    AlgoSpec,
    LLMAlgorithmSpec,
    MultiAgentRLAlgorithmSpec,
    RLAlgorithmSpec,
)
from agilerl.models.algorithms import (  # noqa: F401
    CQNSpec,
    DDPGSpec,
    DQNSpec,
    IPPOSpec,
    MADDPGSpec,
    MATD3Spec,
    NeuralTSSpec,
    NeuralUCBSpec,
    PPOSpec,
    RainbowDQNSpec,
    TD3Spec,
)
from agilerl.models.hpo import (  # noqa: F401
    MultiFrequencySelectionSpec,
    MutationProbabilities,
    MutationSpec,
    SelectionStrategySpec,
    TournamentSelectionSpec,
)
from agilerl.models.manifest import TrainingManifest  # noqa: F401
from agilerl.models.networks import (  # noqa: F401
    CnnSpec,
    ContinuousQNetworkSpec,
    DeterministicActorSpec,
    FinetuningNetworkSpec,
    LstmSpec,
    MlpSpec,
    MultiInputSpec,
    NetworkSpec,
    QNetworkSpec,
    SimbaSpec,
    StochasticActorSpec,
    ValueNetworkSpec,
    normalize_manifest_network,
)
from agilerl.models.training import ReplayBufferSpec, TrainingSpec

if HAS_LLM_DEPENDENCIES:
    from agilerl.models.algorithms import DPOSpec, GRPOSpec  # noqa: F401

# NOTE: env has heavy imports (gymnasium, pandas, datasets, pettingzoo)
# so we lazy-load it to keep imports from agilerl.models lightweight.
__getattr__, __dir__, _ = lazy.attach(
    __name__,
    submod_attrs={
        "env": [
            "GymEnvSpec",
            "LLMEnvSpec",
            "OfflineEnvSpec",
            "PzEnvSpec",
        ],
    },
)

if TYPE_CHECKING:
    from agilerl.models.env import (
        GymEnvSpec,
        LLMEnvSpec,
        OfflineEnvSpec,
        PzEnvSpec,
    )

    EnvironmentSpecType = GymEnvSpec | PzEnvSpec | LLMEnvSpec | OfflineEnvSpec
    ReplayBufferSpecType = ReplayBufferSpec | None
    TrainingSpecType = TrainingSpec | None
    MutationSpecType = MutationSpec | None
