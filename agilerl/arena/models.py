from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EnvironmentRef(BaseModel):
    """Reference to a registered Arena environment."""

    model_config = ConfigDict(strict=True)

    name: str = Field(min_length=1, max_length=256)
    version: str = Field(min_length=1, max_length=64)


class AlgorithmSpec(BaseModel):
    """Algorithm selection and hyperparameter overrides."""

    model_config = ConfigDict(strict=True)

    name: str = Field(min_length=1, max_length=128)
    config: dict[str, Any] = Field(default_factory=dict)


class MutationSpec(BaseModel):
    """Evolutionary mutation configuration.

    All probability fields must be in [0, 1].
    """

    model_config = ConfigDict(strict=True)

    mutation_p: float = Field(ge=0.0, le=1.0)
    no_mutation: float = Field(ge=0.0, le=1.0)
    architecture: float = Field(ge=0.0, le=1.0)
    parameters: float = Field(ge=0.0, le=1.0)
    activation: float = Field(ge=0.0, le=1.0)
    rl_hp: float = Field(ge=0.0, le=1.0)


class TournamentSpec(BaseModel):
    """Tournament selection configuration."""

    model_config = ConfigDict(strict=True)

    tournament_size: int = Field(ge=2)
    elitism: bool = True


class ResourceSpec(BaseModel):
    """Compute resource requirements for a training job."""

    model_config = ConfigDict(strict=True)

    accelerator: Literal["cpu", "gpu"] = "cpu"
    accelerator_count: int = Field(default=1, ge=1)
    memory_gb: int | None = Field(default=None, ge=1)


class TrainingJobConfig(BaseModel):
    """Full specification for an Arena evolutionary training job."""

    model_config = ConfigDict(strict=True)

    environment: EnvironmentRef
    algorithm: AlgorithmSpec
    pop_size: int = Field(default=6, ge=1)
    max_generations: int = Field(default=100, ge=1)
    target_fitness: float | None = None
    mutation: MutationSpec | None = None
    tournament: TournamentSpec | None = None
    resources: ResourceSpec = Field(default_factory=ResourceSpec)


class AgentConfig(BaseModel):
    """Configuration for connecting to a deployed Arena agent endpoint."""

    model_config = ConfigDict(strict=True)

    endpoint: str = Field(min_length=1)
    api_key: str | None = Field(default=None, min_length=1)
