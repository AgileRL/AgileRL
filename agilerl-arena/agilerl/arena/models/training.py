# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The manifest's ``training`` and ``replay_buffer`` sections."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)
from typing_extensions import Self

RolloutMode = Literal["colocated", "async"]
RolloutVersionStamp = Literal["publish", "oldest_turn"]


class NStepBufferArgs(BaseModel):
    """Arguments for the n-step replay buffer."""

    model_config = ConfigDict(extra="forbid")

    n_step: int = Field(
        default=3,
        ge=1,
        description="Steps of reward accumulated into each stored transition.",
    )

    @field_validator("n_step", mode="before")
    @classmethod
    def _default_none(cls, value: int | None) -> int:
        return 3 if value is None else value


class PerBufferArgs(BaseModel):
    """Arguments for the prioritized experience replay buffer."""

    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "How strongly sampling follows TD error. 0 samples uniformly, 1 "
            "samples in full proportion to priority."
        ),
    )

    @field_validator("alpha", mode="before")
    @classmethod
    def _default_none(cls, value: float | None) -> float:
        return 0.5 if value is None else value


class ReplayBufferSpec(BaseModel):
    """Classic branch of the ``replay_buffer`` section."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["classic"] = Field(
        default="classic",
        description="Selects the experience-replay buffer. The LLM deque is ``kind: llm``.",
    )
    name: str | None = Field(default=None, description="Label for the buffer in logs.")
    max_size: int = Field(
        default=100_000,
        ge=1,
        validation_alias=AliasChoices("max_size", "memory_size"),
        description=(
            "Transitions held before the oldest are evicted. Larger buffers "
            "keep more off-policy data at a proportional memory cost."
        ),
    )
    standard_buffer: bool = Field(
        default=True, description="Sample transitions uniformly at random."
    )
    n_step_buffer: bool = Field(
        default=False,
        description=(
            "Accumulate rewards over several steps before storing, which "
            "propagates reward back through time faster."
        ),
    )
    n_step_buffer_args: NStepBufferArgs = Field(
        default_factory=NStepBufferArgs, description="Settings for the n-step buffer."
    )
    per_buffer: bool = Field(
        default=False,
        description=(
            "Sample transitions in proportion to their TD error rather than "
            "uniformly. Rainbow DQN only."
        ),
    )
    per_buffer_args: PerBufferArgs = Field(
        default_factory=PerBufferArgs,
        description="Settings for prioritized experience replay.",
    )


def _llm_rollout_json_schema(schema: dict[str, Any]) -> None:
    """Advertise the YAML tag that selects this spec. It is not a field."""
    schema.setdefault("properties", {})["kind"] = {
        "type": "string",
        "const": "llm",
        "default": "llm",
        "description": "Selects the async rollout deque.",
    }


class LLMRolloutBufferSpec(BaseModel):
    """LLM branch of the ``replay_buffer`` section: the async rollout deque."""

    model_config = ConfigDict(
        extra="forbid", json_schema_extra=_llm_rollout_json_schema
    )

    name: str | None = Field(default=None, description="Label for the buffer in logs.")
    max_size: int = Field(
        default=10_000,
        ge=1,
        validation_alias=AliasChoices("max_size", "memory_size"),
        description=(
            "Rollout groups queued per learner shard before the oldest is "
            "evicted. Counts generated groups, not RL transitions."
        ),
    )
    buffer_occupancy_multiplier: int = Field(
        default=1,
        ge=1,
        description="Headroom the buffer keeps over what the trainer consumes per cycle.",
    )
    max_rollout_version_lag: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Weight-sync versions a group may fall behind before it is dropped "
            "as too off-policy. Unset keeps every queued group."
        ),
    )

    @model_serializer(mode="wrap")
    def _dump_kind_tag(
        self, handler: Callable[[Self], dict[str, Any]]
    ) -> dict[str, Any]:
        """Tag the dumped section so reload selects this spec."""
        return {"kind": "llm", **handler(self)}


def _parse_buffer_section(value: object) -> object:
    """Route ``replay_buffer`` to classic or LLM.

    ``kind: llm`` selects :class:`LLMRolloutBufferSpec`; it is not a field on
    that spec, so it is stripped before the spec is built. Omitted kind is
    classic. Returning a constructed LLM spec here is required: a stripped
    dict would match :class:`ReplayBufferSpec` first.
    """
    if isinstance(value, ReplayBufferSpec | LLMRolloutBufferSpec):
        return value
    if not isinstance(value, dict):
        return value
    kind = value.get("kind", "classic")
    if kind == "llm":
        return LLMRolloutBufferSpec.model_validate(
            {k: v for k, v in value.items() if k != "kind"}
        )
    if kind != "classic":
        msg = f"replay_buffer.kind must be 'classic' or 'llm', got {kind!r}"
        raise ValueError(msg)
    if "kind" not in value:
        return {**value, "kind": "classic"}
    return value


BufferSpec = Annotated[
    ReplayBufferSpec | LLMRolloutBufferSpec,
    BeforeValidator(_parse_buffer_section),
]


class TrainingSpec(BaseModel):
    """The manifest's ``training`` section."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(
        default=None, description="Label for the training loop in logs."
    )
    max_steps: int = Field(
        default=1_000_000,
        ge=1,
        description="Total steps to train for. The run stops here.",
    )
    evo_steps: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("evo_steps", "metrics_interval"),
        description=(
            "Steps between evolution rounds, where the population is ranked "
            "and mutated. Unset resolves to the algorithm's own cadence."
        ),
    )
    pop_size: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("pop_size", "population_size"),
        description=(
            "Agents trained in parallel. Population-based HPO needs more than "
            "one; a single agent just trains normally."
        ),
    )
    eval_steps: int | None = Field(
        default=None,
        ge=1,
        description="Steps per evaluation episode. Unset runs a full episode.",
    )
    eval_loop: int = Field(
        default=1,
        ge=1,
        description=(
            "Evaluation episodes averaged into each fitness score. More "
            "episodes give a less noisy ranking."
        ),
    )
    hpo: bool = Field(
        default=True,
        description=(
            "Mutate hyperparameters and architecture between evolution rounds. "
            "Off trains the population without changing it."
        ),
    )
    target_score: float | None = Field(
        default=None,
        description="Stop early once this fitness is reached. Unset trains to max_steps.",
    )

    learning_delay: int = Field(
        default=0,
        ge=0,
        description=(
            "Steps collected before the first learn step, so the buffer holds "
            "some experience first. Off-policy algorithms only."
        ),
    )
    eps_start: float | None = Field(
        default=None,
        description="Initial probability of taking a random action.",
    )
    eps_end: float | None = Field(
        default=None,
        description="Floor the exploration probability decays to.",
    )
    eps_decay: float | None = Field(
        default=None,
        description="Per-step multiplier applied to the exploration probability.",
    )

    checkpoint_steps: int | None = Field(
        default=None, ge=1, description="Steps between checkpoints. Unset never writes."
    )
    checkpoint_path: str | None = Field(
        default=None, description="Where checkpoints are written."
    )
    overwrite_checkpoints: bool = Field(
        default=False,
        description="Overwrite the previous checkpoint instead of keeping each one.",
    )

    evaluation_interval: int = Field(
        default=10,
        ge=1,
        description="Steps between evaluations during LLM fine-tuning.",
    )
    num_epochs: int | None = Field(
        default=None,
        ge=1,
        description="Passes over the dataset, when the loop is epoch-driven rather than step-driven.",
    )
    evo_epochs: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Evolution interval in epochs, for an epoch-driven loop. The "
            "trainer converts it to steps from the dataset size, so it needs "
            "num_epochs set too; use evo_steps for a step-driven run."
        ),
    )
    max_wall_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Wall-clock limit for the run. Unset trains to max_steps.",
    )

    episode_steps: int = Field(
        default=500, ge=1, description="Steps per episode. Bandits only."
    )
    sum_scores: bool = Field(
        default=True,
        description=(
            "Sum sub-agent scores into one fitness rather than averaging. "
            "Multi-agent only; usually True for cooperative environments."
        ),
    )

    reporting_interval: int = Field(
        default=1024, ge=1, description="Steps between metric reports."
    )
    experience_sharing: bool = Field(
        default=False,
        description=(
            "Let population members learn from each other's transitions. Not "
            "supported under async rollout."
        ),
    )
    env_resource_ratio: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Share of each node's CPU reserved for environment workers.",
    )

    rollout_mode: RolloutMode = Field(
        default="colocated",
        description=(
            "Where generation runs. 'colocated' alternates generation and "
            "learning on the same GPU; 'async' moves generation onto "
            "dedicated rollout engines that run continuously. Staleness is "
            "bounded by replay_buffer.max_rollout_version_lag."
        ),
    )
    rollout_batch_size: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Rollout groups the trainer consumes per cycle. Must divide evenly "
            "by training_gpus_per_agent so every learner shard gets data."
        ),
    )
    rollout_engines_per_agent: int | Literal["auto"] = Field(
        default=0,
        description=(
            "Generation engines per population member, one GPU each. 'auto' "
            "takes every GPU the trainers leave over."
        ),
    )
    training_gpus_per_agent: int = Field(
        default=1,
        ge=1,
        description=(
            "Data-parallel trainer GPUs per population member. The replay "
            "buffer shards rollouts across them."
        ),
    )
    weight_sync_backend: Literal["filesystem", "nccl"] | None = Field(
        default=None,
        description=(
            "How updated LoRA adapters reach the rollout engines. 'nccl' "
            "broadcasts over the network; 'filesystem' writes checkpoints. "
            "Both are LoRA-only — the base model is never re-sent."
        ),
    )
    weight_sync_interval: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Learn chunks between rollout weight syncs under async rollout. "
            "Unset resolves to 1: fresh weights every cycle."
        ),
    )
    rollout_version_stamp: RolloutVersionStamp = Field(
        default="oldest_turn",
        description=(
            "Which weight version a published group is fenced by. "
            "'oldest_turn' ages a group by how long it ran, so long episodes "
            "are dropped first; 'publish' makes buffer residency independent "
            "of episode duration."
        ),
    )

    @property
    def async_rollout(self) -> bool:
        """Whether generation runs on rollout engines rather than the trainer."""
        return self.rollout_mode == "async"

    @model_validator(mode="before")
    @classmethod
    def _reject_retired_async_rollout_key(cls, data: object) -> object:
        """Reject the ``async_rollout`` training key."""
        if isinstance(data, dict) and "async_rollout" in data:
            msg = (
                "`async_rollout` is a retired training key; set "
                "`rollout_mode: async` instead of `async_rollout: true`."
            )
            raise ValueError(msg)
        return data

    @model_validator(mode="after")
    def _validate_training_parameters(self) -> Self:
        if self.evo_steps is not None and self.evo_steps > self.max_steps:
            msg = f"evo_steps ({self.evo_steps}) must be less than or equal to max_steps ({self.max_steps})."
            raise ValueError(msg)
        if self.evo_epochs is not None and self.num_epochs is None:
            msg = (
                "training.evo_epochs needs training.num_epochs: the trainer only "
                "converts epochs to steps for an epoch-driven loop, so otherwise "
                "the interval would be ignored. Use training.evo_steps instead."
            )
            raise ValueError(msg)
        if (
            self.eps_start is not None
            and self.eps_end is not None
            and self.eps_start < self.eps_end
        ):
            msg = f"eps_start ({self.eps_start}) must be greater than or equal to eps_end ({self.eps_end})."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_async_rollout(self) -> Self:
        if not self.async_rollout:
            return self

        engines = self.rollout_engines_per_agent
        if engines != "auto" and int(engines) < 1:
            msg = (
                "async rollout requires training.rollout_engines_per_agent >= 1 "
                f"or 'auto'; got {engines}"
            )
            raise ValueError(msg)

        if self.weight_sync_backend is None:
            self.weight_sync_backend = "nccl"
        if self.rollout_batch_size is None:
            self.rollout_batch_size = 1
        if self.weight_sync_interval is None:
            self.weight_sync_interval = 1

        # An engine publishes group_index in [0, rollout_batch_size) and the
        # buffer shards on group_index % trainers, so a remainder leaves the
        # trailing learner shards permanently empty.
        if self.rollout_batch_size % self.training_gpus_per_agent:
            msg = (
                "async rollout requires training.rollout_batch_size to be a "
                "multiple of training.training_gpus_per_agent so every "
                "data-parallel learner shard receives groups; got "
                f"rollout_batch_size={self.rollout_batch_size} and "
                f"training_gpus_per_agent={self.training_gpus_per_agent}"
            )
            raise ValueError(msg)

        if self.experience_sharing:
            msg = "training.experience_sharing is not supported under async rollout."
            raise ValueError(msg)

        return self
