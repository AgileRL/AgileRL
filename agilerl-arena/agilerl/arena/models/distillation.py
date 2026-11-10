# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""On-policy distillation teacher settings (torch-free)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from agilerl.arena.models.networks import VLLMConfig

DEFAULT_TEACHER_TEMPLATE = (
    "A reference solution to the task that follows is given below. Solve the "
    "task as that reference does.\n\n{reference}"
)
DEFAULT_TEACHER_CONTEXT_KEY = "reference_trace"


class DistillationTeacherModel(BaseModel):
    """A separately hosted model that scores the student's rollouts."""

    model_config = ConfigDict(extra="forbid")

    pretrained_model_name_or_path: str = Field(
        min_length=1,
        description=(
            "Hub id of the teacher, from the model catalog. It must share the "
            "student's tokenizer vocabulary, since it scores the student's token ids."
        ),
    )
    vllm_config: VLLMConfig = Field(
        default_factory=VLLMConfig,
        description=(
            "vLLM settings of the teacher engine. tensor_parallel_size is the "
            "GPUs each rollout engine's teacher adds to the run; unset is 1."
        ),
    )

    @property
    def gpus(self) -> int:
        """GPUs one teacher engine occupies."""
        return self.vllm_config.tensor_parallel_size or 1


class DistillationTeacherConfig(BaseModel):
    """The model that scores each rollout; the per-token gap is the distillation advantage.

    By default the teacher is the live policy, given each task's reference
    trace in a leading system message. With ``adapter`` it is the student's base
    model under that frozen LoRA adapter, served beside the policy's adapter in
    the rollout engines. With ``model`` it is that model, in its own vLLM engine
    beside each rollout engine. A ``context_key`` gives either the reference.
    """

    model_config = ConfigDict(extra="forbid")

    model: DistillationTeacherModel | None = Field(
        default=None,
        description=(
            "A separate, usually larger, teacher model. Unset scores in the "
            "rollout engines, with teacher.adapter or the live policy."
        ),
    )
    adapter: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "LoRA adapter of the student's base model that scores rollouts, such "
            "as a GRPO run's checkpoint: a local or s3:// directory holding "
            "adapter_config.json, or a checkpoint holding it under actor/. Its "
            "rank must fit the student's network.lora_config.lora_r. The "
            "rollout engines serve it frozen beside the policy's adapter, so "
            "weight sync always sends the adapter."
        ),
    )
    context_key: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Key of the env reset info holding the task's reference trace. A "
            "dataset-backed rollout reads the dataset column of that name. Unset "
            f"is {DEFAULT_TEACHER_CONTEXT_KEY!r} for the live policy, and no "
            "reference with a teacher model or adapter."
        ),
    )
    template: str = Field(
        default=DEFAULT_TEACHER_TEMPLATE,
        description=(
            "System message the teacher reads before the rollout; '{reference}' "
            "is replaced with the reference trace."
        ),
    )
    max_context_tokens: int = Field(
        default=8192,
        ge=1,
        description=(
            "Token budget of the teacher's system message. A longer reference is "
            "cut to fit, and rollout engines reserve this much extra context."
        ),
    )
    coef: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Weight of the per-token teacher-minus-student log-prob on the advantage."
        ),
    )
    max_token_gap: float | None = Field(
        default=10.0,
        gt=0.0,
        description=(
            "Bound, in nats, on each token's teacher-minus-student log-prob "
            "before coef scales it. Tokens the teacher finds near impossible "
            "otherwise outweigh the rest of the batch. Null leaves it unbounded."
        ),
    )

    @field_validator("template")
    @classmethod
    def _template_names_the_reference(cls, template: str) -> str:
        if "{reference}" not in template:
            msg = "teacher.template must contain '{reference}'."
            raise ValueError(msg)
        return template

    @model_validator(mode="after")
    def _one_teacher_source(self) -> Self:
        if self.model is not None and self.adapter is not None:
            msg = (
                "Set one of teacher.model and teacher.adapter: a teacher model "
                "runs its own engine, a teacher adapter runs in the student's."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _self_distillation_reads_a_reference(self) -> Self:
        if self.model is None and self.adapter is None and self.context_key is None:
            self.context_key = DEFAULT_TEACHER_CONTEXT_KEY
        return self

    @property
    def context_tokens(self) -> int:
        """Context the scoring engine reserves ahead of an episode for the reference."""
        return 0 if self.context_key is None else self.max_context_tokens

    @property
    def rollout_engine_context_tokens(self) -> int:
        """Context the rollout engine itself reserves: none when a teacher model scores."""
        return 0 if self.model is not None else self.context_tokens


class CheckpointPoolConfig(BaseModel):
    """Distil one earlier checkpoint of a finished GRPO run at a time.

    The stage reads ``checkpoint_<step>`` directories that run already wrote.
    It does not write checkpoints. ``every_n_steps`` only drops saved steps
    (keep a step when ``step % every_n_steps == 0``). Each teacher is that
    step's LoRA adapter, loaded without optimizer state. The source run must
    leave ``overwrite_checkpoints`` false, or only one checkpoint exists.
    Disk is one adapter directory per kept step.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Run the checkpoint-pool stage after GRPO. Unset leaves GRPO as it is.",
    )
    run: str | None = Field(
        default=None,
        min_length=1,
        description="Local or s3:// directory of the finished run's checkpoint_<step> dirs.",
    )
    val_set: str | None = Field(
        default=None,
        min_length=1,
        description="Held-out set the checkpoints were scored on. A local path must exist.",
    )
    scores: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "JSON of greedy validation scores: held_out_score, and checkpoints "
            "mapping a step to the task ids it solved. student_solved is optional."
        ),
    )
    every_n_steps: int | None = Field(
        default=None,
        ge=1,
        description="Keep a saved checkpoint only when its step is a multiple of this.",
    )
    schedule_interval: int = Field(
        default=10,
        ge=1,
        description="Learn steps between teacher choices.",
    )
    priority_increment: float = Field(
        default=1.0,
        gt=0.0,
        description="Priority added to a validation task neither the student nor the teacher solves.",
    )
    priority_cap: float = Field(
        default=3.0,
        ge=1.0,
        description="Highest priority a validation task can reach.",
    )

    @model_validator(mode="after")
    def _enabled_pool_names_its_inputs(self) -> Self:
        if not self.enabled:
            return self
        missing = [
            name for name in ("run", "val_set", "scores") if not getattr(self, name)
        ]
        if missing:
            msg = "An enabled checkpoint pool needs " + ", ".join(missing) + "."
            raise ValueError(msg)
        return self
