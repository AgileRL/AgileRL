# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""On-policy distillation teacher settings (torch-free)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
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


class DistillationTeacherRoute(BaseModel):
    """One frozen teacher checkpoint and the domains it scores."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1, description="Name used in metrics and as the fallback."
    )
    model: DistillationTeacherModel
    domains: list[str] = Field(
        default_factory=list,
        description=(
            "Routing values that this teacher scores. The default teacher may own none."
        ),
    )


class DistillationTeacherConfig(BaseModel):
    """The model that scores each rollout; the per-token gap is the distillation advantage.

    Without ``model`` or ``teachers`` the teacher is the live policy, given each
    task's reference trace in a leading system message. With ``model`` it is that
    model, in its own vLLM engine beside each rollout engine. ``teachers`` is a
    pool of frozen checkpoints: each row's ``routing_key`` picks one, and
    ``default_teacher`` scores the rest. A ``context_key`` also gives a teacher
    the reference.
    """

    model_config = ConfigDict(extra="forbid")

    model: DistillationTeacherModel | None = Field(
        default=None,
        description=(
            "A separate, usually larger, teacher model. Unset scores with the "
            "live policy in the rollout engines."
        ),
    )
    context_key: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Key of the env reset info holding the task's reference trace. A "
            "dataset-backed rollout reads the dataset column of that name. Unset "
            f"is {DEFAULT_TEACHER_CONTEXT_KEY!r} without a teacher model, and no "
            "reference with one."
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
    teachers: list[DistillationTeacherRoute] = Field(
        default_factory=list,
        description=(
            "Frozen teacher checkpoints. Each training row is scored by the "
            "teacher whose domains contain its routing value. Empty keeps the "
            "single teacher above."
        ),
    )
    default_teacher: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Name of the teacher that scores a row whose routing value no "
            "teacher owns. Required when teachers is set."
        ),
    )
    routing_key: str = Field(
        default="domain",
        min_length=1,
        description=(
            "Reset-info key, and dataset column, that selects the teacher. "
            "A missing or unknown value uses default_teacher."
        ),
    )
    max_response_tokens: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Warn when a scored response is longer than this. Length growth is "
            "the known failure mode of a teacher the student cannot match. "
            "Null does not warn."
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
    def _self_distillation_reads_a_reference(self) -> Self:
        if self.model is None and not self.teachers and self.context_key is None:
            self.context_key = DEFAULT_TEACHER_CONTEXT_KEY
        return self

    @property
    def context_tokens(self) -> int:
        """Context the scoring engine reserves ahead of an episode for the reference."""
        return 0 if self.context_key is None else self.max_context_tokens

    @property
    def separate_model(self) -> bool:
        """Whether a frozen checkpoint scores, rather than the live policy."""
        return self.model is not None or bool(self.teachers)

    @property
    def rollout_engine_context_tokens(self) -> int:
        """Context the rollout engine itself reserves: none when a teacher model scores."""
        return 0 if self.separate_model else self.context_tokens


def teacher_deployments(
    config: DistillationTeacherConfig,
) -> list[DistillationTeacherModel]:
    """Frozen checkpoints to host, one engine per distinct Hub id.

    :param config: Teacher settings.
    :type config: DistillationTeacherConfig
    :return: Models in first-seen order. A pool hides ``model``.
    :rtype: list[DistillationTeacherModel]
    """
    if config.teachers:
        seen: dict[str, DistillationTeacherModel] = {}
        for route in config.teachers:
            seen.setdefault(route.model.pretrained_model_name_or_path, route.model)
        return list(seen.values())
    if config.model is not None:
        return [config.model]
    return []


def route_teacher(
    config: DistillationTeacherConfig, domain: object
) -> DistillationTeacherRoute:
    """The teacher that scores one row.

    A domain listed on a teacher selects it. Any other value, including a
    missing one, selects ``default_teacher``.

    :param config: Teacher settings. ``teachers`` must be set.
    :type config: DistillationTeacherConfig
    :param domain: The row's routing value.
    :type domain: object
    :return: The teacher route.
    :rtype: DistillationTeacherRoute
    :raises ValueError: If there is no pool, or no fallback for this domain.
    """
    if not config.teachers:
        msg = "route_teacher needs algorithm.teacher.teachers."
        raise ValueError(msg)
    text = domain if isinstance(domain, str) and domain else None
    if text is not None:
        for route in config.teachers:
            if text in route.domains:
                return route
    shown = "<missing>" if text is None else text
    if config.default_teacher is None:
        msg = (
            f"domain {shown!r} has no teacher and algorithm.teacher.default_teacher "
            "is unset."
        )
        raise ValueError(msg)
    for route in config.teachers:
        if route.name == config.default_teacher:
            return route
    msg = (
        f"domain {shown!r} has no teacher and default_teacher "
        f"{config.default_teacher!r} is not one of "
        f"{[route.name for route in config.teachers]}."
    )
    raise ValueError(msg)


def teacher_pool_problems(
    config: DistillationTeacherConfig,
    student_id: str | None,
    colocated: bool,
    layout_written: bool,
) -> list[str]:
    """Everything wrong with a multi-teacher pool, in one list.

    An empty pool is the single-teacher path and has no problems here.

    :param config: Teacher settings.
    :type config: DistillationTeacherConfig
    :param student_id: Student Hub id, for the vocabulary check.
    :type student_id: str | None
    :param colocated: Generation shares the trainer GPU.
    :type colocated: bool
    :param layout_written: The manifest sets the student width and a GPU count.
    :type layout_written: bool
    :return: Problem sentences. Empty when the pool is absent or valid.
    :rtype: list[str]
    """
    if not config.teachers:
        return []
    problems: list[str] = []
    if config.model is not None:
        problems.append(
            "set algorithm.teacher.teachers or algorithm.teacher.model, not both"
        )
    names = [route.name for route in config.teachers]
    if len(names) != len(set(names)):
        problems.append(f"teacher names must be unique, got {names}")
    owned: dict[str, str] = {}
    for route in config.teachers:
        if not route.domains and route.name != config.default_teacher:
            problems.append(
                f"teacher {route.name} owns no domain and is not default_teacher"
            )
        for domain in route.domains:
            if not domain:
                problems.append(f"teacher {route.name} has an empty domain")
            elif domain in owned:
                problems.append(
                    f"domain {domain!r} is owned by {owned[domain]} and {route.name}"
                )
            else:
                owned[domain] = route.name
    if config.default_teacher is None:
        problems.append(
            "default_teacher is required; an unknown domain has no fallback"
        )
    elif config.default_teacher not in names:
        problems.append(
            f"default_teacher {config.default_teacher!r} is not one of {names}"
        )
    if colocated:
        problems.append("multi-teacher OPD needs rollout_mode async, not colocated")
    if not layout_written:
        problems.append(
            "multi-teacher OPD needs its GPU layout written out: set "
            "algorithm.vllm_config.tensor_parallel_size and "
            "training.training_gpus_per_agent or training.rollout_engines_per_agent"
        )
    student = SUPPORTED_MODEL_INFO.get(student_id) if student_id else None
    student_vocab = None if student is None else student.vocab_size
    for model in teacher_deployments(config):
        hub_id = model.pretrained_model_name_or_path
        info = SUPPORTED_MODEL_INFO.get(hub_id)
        if info is None:
            problems.append(
                f"teacher {hub_id!r} is not in the model catalog, so its engine "
                "memory cannot be sized"
            )
            continue
        if student_vocab is not None and info.vocab_size != student_vocab:
            problems.append(
                f"teacher {hub_id} scores the student's token ids, so both need "
                f"one vocabulary; {student_id} has {student_vocab} tokens and "
                f"{hub_id} has {info.vocab_size}"
            )
    return problems
