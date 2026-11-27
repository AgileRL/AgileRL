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
    model_serializer,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import Self

from agilerl.models.visibility import AUTO, HIDDEN

RolloutMode = Literal["colocated", "async"]
RolloutVersionStamp = Literal["publish", "oldest_turn"]
CheckpointExportFormat = Literal["adapter", "merged"]
CheckpointExportTrigger = Literal["final", "best", "on_demand", "every"]

DEFAULT_DROP_UNINFORMATIVE_GROUPS = True

# Starting values for a new LLM run; the schema and the default manifest share them.
LLM_REPORTING_INTERVAL = 1
LLM_EVALUATION_INTERVAL = 10
ROLLOUT_LLM_MAX_STEPS = 200

# 1.x manifests carry these; validation drops them.
CLASSIC_ONLY_TRAINING_KEYS = (
    "learning_delay",
    "episode_steps",
    "eps_start",
    "eps_end",
    "eps_decay",
    "sum_scores",
)


def ignored_keys_warning(keys: list[str]) -> str:
    """The warning for classic-only training keys that validation drops.

    :param keys: The dropped keys, in manifest order.
    :type keys: list[str]
    :returns: The warning text.
    :rtype: str
    """
    return (
        f"training.{', training.'.join(keys)} ignored; removed in a future "
        "agilerl release."
    )


def classic_only_key_warnings(document: dict[str, Any]) -> list[str]:
    """Warn when the raw manifest's training section names a classic-only key.

    :param document: A raw manifest document that passed validation.
    :type document: dict[str, Any]
    :returns: Zero or one warning.
    :rtype: list[str]
    """
    training = document.get("training", {})
    present = [key for key in CLASSIC_ONLY_TRAINING_KEYS if key in training]
    return [ignored_keys_warning(present)] if present else []


class CheckpointExportSpec(BaseModel):
    """When to write a merged Hugging Face package next to adapter checkpoints.

    :param format: ``adapter`` writes LoRA weights only; ``merged`` also
        writes ``merged/`` after a durable adapter save when ``trigger`` matches.
    :param trigger: ``final`` (last durable save of the run), ``best`` (best
        checkpoint if the trainer marks one, otherwise ``final``), ``on_demand``
        (no merge during train), or ``every`` (after every adapter save).
    """

    model_config = ConfigDict(extra="forbid")

    format: CheckpointExportFormat = Field(
        default="adapter",
        description=(
            "adapter writes LoRA weights only; merged also writes merged/ after "
            "a durable adapter save when trigger matches."
        ),
    )
    trigger: CheckpointExportTrigger = Field(
        default="final",
        description=(
            "final (last durable save of the run), best (best checkpoint if the "
            "trainer marks one, otherwise final), on_demand (no merge during "
            "train), or every (after every adapter save)."
        ),
    )

    def should_merge(self, is_final: bool, is_best: bool) -> bool:
        """Return whether this adapter save should write merged HF weights."""
        if self.format != "merged":
            return False
        match self.trigger:
            case "every":
                return True
            case "on_demand":
                return False
            case "best":
                return is_best or is_final
            case "final":
                return is_final


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
    # Groups up to two weight versions old stay learnable, so the trainer rarely
    # waits on generation; the importance-ratio correction absorbs the gap.
    max_rollout_version_lag: int = Field(
        default=2,
        ge=0,
        description=(
            "Weight-sync versions a group may fall behind before it is dropped "
            "as too off-policy: the sync to version v keeps groups generated at "
            "version v - lag or later. 0 is strictly on-policy; the default 2 "
            "also keeps groups started under the two previous weight versions."
        ),
    )
    drop_uninformative_groups: bool = Field(
        default=DEFAULT_DROP_UNINFORMATIVE_GROUPS,
        description=(
            "Discard groups whose completions all got identical rewards before "
            "they reach a learner, and wait until each learner's batch is full "
            "of groups whose rewards differ. Tied groups have zero advantage, so "
            "training on them only costs compute, but each trained step then "
            "generates more groups than it trains. No effect when group_size is 1."
        ),
    )
    group_replay_max_age: int = Field(
        default=20,
        ge=0,
        description=(
            "Policy versions (weight syncs) a stored completion stays eligible "
            "for group replay; 0 turns group replay off. Group replay keeps "
            "each task's most recent completion for each episode return seen "
            "(up to 4 returns per task). When every completion in a new group "
            "for that task gets the same return, swap one of them for the "
            "stored completion with the highest other return, so the group "
            "still has a reward difference to learn from: a group that all "
            "failed gets a past success, one that all succeeded gets a past "
            "failure. The replayed completion keeps the token log-probabilities "
            "it was sampled with, and the algorithm's ratio clipping bounds its "
            "update. Applies to async rollout over a dataset-backed task list; "
            "no effect when group_size is 1. The store is in memory only and "
            "refills from new rollouts after a resume."
        ),
    )

    @model_serializer(mode="wrap")
    def _dump_kind_tag(
        self, handler: Callable[[Self], dict[str, Any]]
    ) -> dict[str, Any]:
        """Tag the dumped section so reload selects this spec."""
        return {"kind": "llm", **handler(self)}


def _parse_buffer_section(value: object) -> object:
    """Strip the ``kind: llm`` tag, which selects the spec but is not a field on it."""
    if not isinstance(value, dict) or "kind" not in value:
        return value
    kind = value["kind"]
    if kind != "llm":
        msg = f"replay_buffer.kind must be 'llm', got {kind!r}"
        raise ValueError(msg)
    return {k: v for k, v in value.items() if k != "kind"}


BufferSpec = Annotated[
    LLMRolloutBufferSpec,
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
        description=(
            "Total steps to train for. The run stops here. Unset on an LLM run "
            "over dataset rows trains one pass over the dataset."
        ),
    )
    evo_steps: int | None = Field(
        default=None,
        json_schema_extra=AUTO,
        ge=1,
        validation_alias=AliasChoices("evo_steps", "metrics_interval"),
        description=(
            "Steps between evaluation rounds, also spelled metrics_interval. "
            "Unset resolves to the algorithm's own cadence."
        ),
    )
    pop_size: SkipJsonSchema[int] = Field(
        default=1,
        json_schema_extra=HIDDEN,
        ge=1,
        validation_alias=AliasChoices("pop_size", "population_size"),
        description=(
            "Agents trained in parallel. Population-based HPO needs more than "
            "one; a single agent just trains normally."
        ),
    )
    eval_steps: SkipJsonSchema[int | None] = Field(
        default=None,
        ge=1,
        description="Steps per evaluation episode. Unset runs a full episode.",
    )
    eval_loop: int = Field(
        default=1,
        ge=1,
        description=(
            "Evaluation passes averaged into each eval score. More passes "
            "give a steadier score."
        ),
    )
    hpo: SkipJsonSchema[bool] = Field(
        default=False,
        json_schema_extra=HIDDEN,
        description=(
            "Mutate hyperparameters and architecture between evolution rounds. "
            "When false, the population trains without hyperparameter mutation."
        ),
    )
    target_score: float | None = Field(
        default=None,
        description="Stop early once the eval score reaches this. Unset trains to max_steps.",
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
    checkpoint_optimizer: bool = Field(
        default=True,
        description=(
            "Store the optimizer state in LLM checkpoints so a resumed run "
            "keeps its Adam moments. Adds about twice the adapter's size."
        ),
    )

    evaluation_interval: int | None = Field(
        default=None,
        ge=1,
        description="Steps between evaluations during LLM fine-tuning.",
    )
    eval_samples_per_task: int = Field(
        default=0,
        ge=0,
        description=(
            "Held-out episodes per task sampled at the algorithm's temperature "
            "and top_p under async rollout. Their mean is eval_score and "
            "eval_accuracy. 0 reports the near-greedy pass there instead."
        ),
    )
    eval_greedy: bool = Field(
        default=True,
        description=(
            "Also run eval_loop near-greedy passes over the held-out tasks "
            "under async rollout. With eval_samples_per_task set they report "
            "as eval_score_greedy and eval_accuracy_greedy."
        ),
    )
    eval_max_concurrent_episodes: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Held-out episodes one rollout engine runs at once; the rest wait "
            "in a queue. They share the engine's episode slots with training. "
            "Unset lets them take every free slot."
        ),
    )
    num_epochs: SkipJsonSchema[int | None] = Field(
        default=None,
        ge=1,
        description=(
            "Passes over the dataset, when the loop is epoch-driven rather than "
            "step-driven. Not for LLM runs: they set max_steps."
        ),
    )
    evo_epochs: SkipJsonSchema[int | None] = Field(
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

    reporting_interval: int = Field(
        default=1024, ge=1, description="Steps between metric reports."
    )
    experience_sharing: SkipJsonSchema[bool | None] = Field(
        default=None,
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
            "learning on the same GPU, for local runs; 'async' moves "
            "generation onto dedicated rollout engines that run continuously. "
            "Arena runs every rollout LLM job async. "
            "replay_buffer.max_rollout_version_lag bounds staleness."
        ),
    )
    rollout_batch_size: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Rollout groups per cycle, in total across all of a member's "
            "rollout engines, rounded up to a multiple of the engine count; "
            "each engine runs an equal share. Must divide evenly by "
            "training_gpus_per_agent so every learner shard gets data. Unset "
            "is the algorithm's batch_size rounded up to a multiple of both "
            "training_gpus_per_agent and the engine count, so the engines keep "
            "one learn step of groups in flight."
        ),
    )
    rollout_engines_per_agent: int | Literal["auto"] = Field(
        default=0,
        description=(
            "Generation engines per trainer, one GPU each. 'auto' "
            "takes every GPU the trainers leave over. Unset under async "
            "rollout with training_gpus_per_agent also unset, the trainer / "
            "engine split is picked by the throughput model; unset otherwise "
            "is 'auto'."
        ),
    )
    training_gpus_per_agent: int = Field(
        default=1,
        ge=1,
        description=(
            "Data-parallel trainer GPUs. The replay buffer shards rollouts across them."
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
    rollout_version_stamp: RolloutVersionStamp | None = Field(
        default=None,
        json_schema_extra=HIDDEN,
        description=(
            "Which weight version a published group is fenced by. "
            "'oldest_turn' ages a group by how long it ran, so long episodes "
            "are dropped first; 'publish' makes buffer residency independent "
            "of episode duration."
        ),
    )
    checkpoint_export: CheckpointExportSpec | None = Field(
        default=None,
        description=(
            "Adapter-only save by default, or a merged Hugging Face export "
            "when format is merged and trigger matches."
        ),
    )

    def effective_checkpoint_export(self) -> CheckpointExportSpec:
        """LLM checkpoint export defaults when the training section omits it."""
        if self.checkpoint_export is not None:
            return self.checkpoint_export
        return CheckpointExportSpec()

    def effective_rollout_version_stamp(self) -> RolloutVersionStamp | None:
        """Async rollout stamping when the training section omits it."""
        if self.rollout_version_stamp is not None:
            return self.rollout_version_stamp
        if self.rollout_mode == "async":
            return "oldest_turn"
        return None

    @property
    def async_rollout(self) -> bool:
        """Whether generation runs on rollout engines rather than the trainer."""
        return self.rollout_mode == "async"

    @model_validator(mode="before")
    @classmethod
    def _drop_classic_only_keys(cls, data: object) -> object:
        """Drop the classic-only training keys; callers report them with :func:`classic_only_key_warnings`."""
        if not isinstance(data, dict):
            return data
        # A subclass that declares these fields (the classic legacy spec) keeps them.
        present = [
            key
            for key in CLASSIC_ONLY_TRAINING_KEYS
            if key in data and key not in cls.model_fields
        ]
        return {key: value for key, value in data.items() if key not in present}

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
        if (
            self.num_epochs is not None
            and "num_epochs" in self.model_fields_set
            and "max_steps" in self.model_fields_set
        ):
            msg = (
                "training.num_epochs and training.max_steps are both set; set one: "
                "num_epochs for an epoch-driven run, max_steps for a step-driven run."
            )
            raise ValueError(msg)
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
        if not self.eval_greedy and self.eval_samples_per_task == 0:
            msg = (
                "training.eval_greedy false needs training.eval_samples_per_task "
                ">= 1; otherwise held-out eval runs no episodes."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_async_rollout(self) -> Self:
        if not self.async_rollout:
            return self

        # Unset stays 0, which async runs read as ``auto``.
        engines = self.rollout_engines_per_agent
        if (
            "rollout_engines_per_agent" in self.model_fields_set
            and engines != "auto"
            and int(engines) < 1
        ):
            msg = (
                "async rollout requires training.rollout_engines_per_agent >= 1 "
                f"or 'auto'; got {engines}"
            )
            raise ValueError(msg)

        if self.weight_sync_interval is None:
            self.weight_sync_interval = 1

        # The buffer deals each group to the emptiest learner shard, so a cycle
        # fills every shard evenly only when the trainer count divides it. The
        # driver fills an unset size once it knows the GPU split.
        if (
            self.rollout_batch_size is not None
            and self.rollout_batch_size % self.training_gpus_per_agent
        ):
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
