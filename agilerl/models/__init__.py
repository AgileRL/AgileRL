from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.models.algo import (
    ALGO_REGISTRY,
    AlgorithmSpec,
    LLMAlgorithmSpec,
    MultiAgentRLAlgorithmSpec,
    RLAlgorithmSpec,
)
from agilerl.models.algorithms import (
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
from agilerl.models.env import (
    ArenaEnvSpec,
    GymEnvSpec,
    LLMEnvSpec,
    OfflineEnvSpec,
    PzEnvSpec,
)
from agilerl.models.hpo import MutationSpec, TournamentSelectionSpec
from agilerl.models.manifest import TrainingManifest
from agilerl.models.networks import NetworkSpec
from agilerl.models.training import ReplayBufferSpec, TrainingSpec

if HAS_LLM_DEPENDENCIES:
    from agilerl.models.algorithms import DPOSpec, GRPOSpec

AlgoSpecT = RLAlgorithmSpec | MultiAgentRLAlgorithmSpec | LLMAlgorithmSpec
EnvironmentSpecT = GymEnvSpec | PzEnvSpec | LLMEnvSpec | OfflineEnvSpec
ArenaEnvSpecT = ArenaEnvSpec | dict[str, str]
ReplayBufferSpecT = ReplayBufferSpec | None
TrainingSpecT = TrainingSpec | None
MutationSpecT = MutationSpec | None


class JobStatus(str, Enum):
    """Status of an Arena training job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    def __str__(self) -> str:
        return str(self.value)


class ArenaVM(BaseModel):
    """Hardware specification for an Arena virtual machine."""

    name: str
    cpus: int = Field(ge=1)
    gpus: int = Field(ge=0)
    ram_gi: int = Field(ge=1, description="RAM in GiB.")
    gram_gi: int = Field(default=0, ge=0, description="GPU memory in GiB.")

    def __str__(self) -> str:
        gpu_str = f"{self.gpus} GPU, " if self.gpus else ""
        gram_str = f", {self.gram_gi}Gi GRAM" if self.gram_gi else ""
        return f"{self.name}: {self.cpus} CPU, {gpu_str}{self.ram_gi}Gi RAM{gram_str}"


class ArenaResource(Enum):
    """Available compute clusters for Arena training."""

    MEDIUM_ACCELERATED = ArenaVM(name="Medium Accelerated", cpus=16, gpus=1, ram_gi=64)
    LARGE_ACCELERATED = ArenaVM(name="Large Accelerated", cpus=32, gpus=1, ram_gi=128)
    XL_ACCELERATED = ArenaVM(name="XL Accelerated", cpus=96, gpus=8, ram_gi=384)
    XL = ArenaVM(name="XL", cpus=96, gpus=0, ram_gi=192)
    XXL = ArenaVM(name="XXL", cpus=192, gpus=0, ram_gi=384)
    A100_2GPU = ArenaVM(name="A100 2xGPU", cpus=24, gpus=2, ram_gi=170)
    A100_4GPU = ArenaVM(name="A100 4xGPU", cpus=48, gpus=4, ram_gi=340, gram_gi=160)
    LARGE_4XL4 = ArenaVM(name="Large 4xL4", cpus=48, gpus=4, ram_gi=192, gram_gi=96)

    def __str__(self) -> str:
        return str(self.value)


class ArenaCluster(BaseModel):
    """Specification for an Arena compute cluster."""

    resource: ArenaResource
    num_nodes: int = Field(ge=1)


__all__ = [
    "ALGO_REGISTRY",
    "AlgorithmSpec",
    "ArenaCluster",
    "ArenaResource",
    "CQNSpec",
    "DDPGSpec",
    "DQNSpec",
    "GymEnvSpec",
    "IPPOSpec",
    "LLMAlgorithmSpec",
    "LLMEnvSpec",
    "MADDPGSpec",
    "MATD3Spec",
    "MultiAgentRLAlgorithmSpec",
    "MutationSpec",
    "NetworkSpec",
    "NeuralTSSpec",
    "NeuralUCBSpec",
    "OfflineEnvSpec",
    "PPOSpec",
    "PzEnvSpec",
    "RLAlgorithmSpec",
    "RainbowDQNSpec",
    "ReplayBufferSpec",
    "TD3Spec",
    "TournamentSelectionSpec",
    "TrainingManifest",
    "TrainingSpec",
]

if HAS_LLM_DEPENDENCIES:
    __all__ += ["DPOSpec", "GRPOSpec"]
