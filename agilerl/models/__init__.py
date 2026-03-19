from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .algo import (
    ALGO_REGISTRY,
    AlgorithmMeta,
    AlgorithmSpec,
    DDPGSpec,
    DPOSpec,
    DQNSpec,
    GRPOSpec,
    IPPOSpec,
    MADDPGSpec,
    MATD3Spec,
    PPOSpec,
    RainbowDQNSpec,
    RLAlgorithmSpec,
    TD3Spec,
)
from .env import EnvironmentSpec
from .hpo import MutationSpec, TournamentSelectionSpec
from .networks import NetworkSpec
from .training import JobStatus, ReplayBufferSpec, TrainingSpec


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


class ArenaTrainingManifest(BaseModel):
    """Complete Arena training job manifest.

    Combines algorithm, environment, mutation, network, replay buffer,
    tournament selection, and training into a single validated document
    that can be submitted to the Arena API.
    """

    algorithm: AlgorithmSpec
    environment: EnvironmentSpec
    mutation: MutationSpec | None = Field(default=None)
    network: NetworkSpec
    replay_buffer: ReplayBufferSpec | None = Field(default=None)
    tournament_selection: TournamentSelectionSpec | None = Field(default=None)
    training: TrainingSpec

    def to_yaml(self) -> str:
        """Serialize the manifest to a YAML string."""
        try:
            import yaml
        except ImportError as exc:
            msg = "PyYAML is required for YAML serialization: pip install pyyaml"
            raise ImportError(msg) from exc

        data = self.model_dump(mode="json", exclude_none=True)
        return yaml.dump(data, default_flow_style=False, sort_keys=False)


__all__ = [
    "ALGO_REGISTRY",
    "AlgorithmMeta",
    "AlgorithmSpec",
    "ArenaCluster",
    "ArenaResource",
    "ArenaTrainingManifest",
    "DDPGSpec",
    "DPOSpec",
    "DQNSpec",
    "EnvironmentSpec",
    "GRPOSpec",
    "IPPOSpec",
    "JobStatus",
    "MADDPGSpec",
    "MATD3Spec",
    "MutationSpec",
    "NetworkSpec",
    "PPOSpec",
    "RLAlgorithmSpec",
    "RainbowDQNSpec",
    "ReplayBufferSpec",
    "TD3Spec",
    "TournamentSelectionSpec",
    "TrainingSpec",
]
