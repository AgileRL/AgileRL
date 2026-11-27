# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The training manifest: the whole contract, in one place."""

from __future__ import annotations

import functools
import logging
from difflib import get_close_matches
from pathlib import Path
from types import UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    overload,
)

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    PlainSerializer,
    PrivateAttr,
    ValidationInfo,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import Self

from agilerl.models.algorithms.base import AlgoSpec, LLMAlgorithmSpec
from agilerl.models.algorithms.opd import OPDSpec
from agilerl.models.algorithms.rollout_llm import (
    RolloutLLMSpec,
    records_sampling_supports,
)
from agilerl.models.bundled_envs import (
    apply_bundled_run_settings,
    arena_context_length,
    lax,
    manifest_model,
)
from agilerl.models.env import EnvSpec, LLMEnvSpec
from agilerl.models.hpo import (
    MultiFrequencySelectionSpec,
    MutationSpec,
    SelectionStrategySpec,
    TournamentSelectionSpec,
    no_op_mutation_probabilities,
    require_mutation_range_inside_field,
)
from agilerl.models.manifest_rules import (
    apply_run_rollout_mode,
    async_rollout_algorithm,
    cpu_offload_blocker,
    rollout_engines_blocker,
)
from agilerl.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.models.networks import (
    FinetuningNetworkSpec,
    check_lora_config_supported,
    default_rollout_vllm_config,
    target_modules_warnings,
)
from agilerl.models.registry import MANIFEST_REGISTRY, did_you_mean
from agilerl.models.training import (
    BufferSpec,
    LLMRolloutBufferSpec,
    TrainingSpec,
)
from agilerl.models.visibility import HIDDEN_PLATFORM_WRITTEN, schema_skipped_fields

logger = logging.getLogger(__name__)


@functools.cache
def warn_once(message: str) -> None:
    """Log *message* as a warning the first time this process sees it.

    A CLI submit validates the manifest for the quote and again for the
    submit; the person reads each warning once.

    :param message: The warning.
    :type message: str
    """
    logger.warning(message)


API_VERSION = "agilerl.com/v1"

# Algorithm fields the run picks from the memory estimate, fastest that fits.
RESOLVED_MEMORY_FIELDS = (
    ("micro_batch_size_per_gpu",),
    ("micro_batch_tokens_per_gpu",),
    ("gradient_checkpointing",),
    ("activation_offload",),
    ("routed_expert_chunk_mib",),
    ("fsdp", "reshard_after_forward"),
    ("fsdp", "optim_cpu_offload"),
    ("fsdp", "routed_expert_chunk_mib"),
    ("fsdp", "activation_offload"),
    ("fsdp", "cpu_offload"),
    ("fsdp", "shard_frozen_tp"),
    ("fsdp", "save_moe_collectives"),
    ("fsdp", "checkpoint_every_n_blocks"),
)


def reject_resolved_memory_fields(manifest: dict[str, Any]) -> None:
    """Raise when a raw manifest sets a field the run picks from the memory estimate.

    :param manifest: Raw manifest document, as written.
    :raises ValueError: Naming every such field the algorithm section sets.
    """
    algorithm = manifest.get("algorithm")
    if not isinstance(algorithm, dict):
        return
    pinned = []
    for path in RESOLVED_MEMORY_FIELDS:
        section = algorithm.get(path[0]) if len(path) > 1 else algorithm
        if isinstance(section, dict) and path[-1] in section:
            pinned.append("algorithm." + ".".join(path))
    if pinned:
        msg = (
            f"{', '.join(pinned)} cannot be set: the run picks the fastest "
            "value that fits from the memory estimate. Remove it from the manifest."
        )
        raise ValueError(msg)


def strips_multimodal_towers(algorithm: LLMAlgorithmSpec, text_only: bool) -> bool:
    """Whether the vLLM engine skips the model's vision and audio towers.

    True for a vision-language catalog model whose environment sends text only
    (modality text, or dataset rows).

    :param algorithm: The run's LLM algorithm spec.
    :param text_only: Whether the environment never sends images.
    :return: Whether the engine serves the language model only.
    """
    if not text_only:
        return False
    hub_id = algorithm.pretrained_model_name_or_path
    model = SUPPORTED_MODEL_INFO.get(hub_id) if hub_id is not None else None
    return model is not None and "vision" in model.modalities


def copy_when_unset(
    target: BaseModel, field: str, value: object, *, source_set: bool
) -> None:
    """Copy ``value`` onto ``target.field`` when the target omitted it; raise when both set it differently.

    :param target: The section that receives the value, such as the algorithm.
    :param field: Field name on ``target``; a field ``target`` lacks is skipped.
    :param value: The value another section implies.
    :param source_set: Whether the other section set ``value`` explicitly.
    :raises ValueError: When ``target`` set ``field`` to a different value.
    """
    if field not in type(target).model_fields:
        return
    if field not in target.model_fields_set:
        if source_set:
            setattr(target, field, value)
        return
    if source_set and getattr(target, field) != value:
        msg = (
            f"algorithm.{field}={getattr(target, field)!r} disagrees with "
            f"{value!r} from another section."
        )
        raise ValueError(msg)


# Validation context key: the reward file is added at submit, so its absence is not an error.
REWARD_FILE_AT_SUBMIT = "reward_file_at_submit"
# Validation context key: the Snowflake app validates, and it refuses rubric files.
IN_SNOWFLAKE_APP = "in_snowflake_app"


def fill_default_context_length(manifest: dict[str, Any]) -> bool:
    """Fill an unset LLM context with 2048, or the next allowed length above max_output_tokens (twice that with answer_continuation), capped at the catalog window.

    :param manifest: Raw manifest document, updated in place.
    :return: Whether the context was filled.
    """
    algorithm = manifest.get("algorithm")
    network = manifest.get("network")
    if not isinstance(network, dict) or not names_llm_algorithm(algorithm):
        return False
    if (
        network.get("max_context_length") is not None
        or algorithm.get("max_model_len") is not None
    ):
        return False
    entry = SUPPORTED_MODEL_INFO.get(manifest_model(manifest))
    if entry is None:
        return False
    # A value pydantic cannot read stays at 2048; pydantic reports it.
    output_cap = lax(int, algorithm.get("max_output_tokens"))
    # The continuation is a second generation of the same cap.
    if output_cap is not None and lax(bool, algorithm.get("answer_continuation")):
        output_cap *= 2
    prompt_room = output_cap + 1 if output_cap is not None else 0
    fill = arena_context_length(max(2048, prompt_room))
    network["max_context_length"] = min(fill, entry.max_context_length)
    return True


def _registered_algorithms() -> str:
    """The sentence listing every registered algorithm once."""
    return f"Registered: {', '.join(MANIFEST_REGISTRY.alias_groups())}."


def names_llm_algorithm(section: object) -> bool:
    """Whether an algorithm section, raw or validated, is a registered LLM algorithm."""
    if isinstance(section, BaseModel):
        return isinstance(section, LLMAlgorithmSpec)
    name = section.get("name") if isinstance(section, dict) else None
    if not isinstance(name, str):
        return False
    try:
        spec_cls = MANIFEST_REGISTRY.get(name)
    except KeyError:
        return False
    return issubclass(spec_cls, LLMAlgorithmSpec)


def resolve_algorithm(data: object) -> object:
    """Dispatch the algorithm section to its registered spec class via ``name``.

    :param data: The raw algorithm section, or an already-built spec.
    :return: The spec, built from the registry when ``data`` is a mapping.
    :raises ValueError: When the section is not a mapping or names no registered algorithm.
    """
    if isinstance(data, BaseModel):
        return data
    if not isinstance(data, dict):
        # ValueError, not TypeError: pydantic only converts ValueError /
        # AssertionError into a ValidationError; a TypeError would escape the
        # validate CLI and submit paths as a crash.
        msg = f"The algorithm section must be a mapping, got {type(data).__name__}."
        raise ValueError(msg)

    payload = dict(data)
    name = payload.pop("name", None)
    if name is None:
        msg = (
            "The algorithm section must include a 'name' field naming a registered "
            f"algorithm. {_registered_algorithms()}"
        )
        raise ValueError(msg)
    try:
        return MANIFEST_REGISTRY.create(name, **payload)
    except KeyError as err:
        # Surface an unknown name as a validation error like any other bad
        # field, rather than a KeyError a caller cannot tell from a crash.
        msg = (
            f"'name' is not a registered algorithm: {name!r}."
            f"{did_you_mean(str(name), MANIFEST_REGISTRY.names())} "
            f"{_registered_algorithms()}"
        )
        raise ValueError(msg) from err


def _serialize_algorithm(spec: AlgoSpec) -> dict[str, Any]:
    """Serialize the algorithm section, restoring the ``name`` discriminator."""
    dumped = spec.model_dump(mode="json", exclude_none=True)
    dumped["name"] = spec.name
    return dumped


def _default_selection_strategy(data: object) -> object:
    """Tag a selection section that omits ``strategy`` as the tournament branch."""
    if isinstance(data, dict) and "strategy" not in data:
        return {**data, "strategy": "tournament"}
    return data


def _is_numeric(annotation: object) -> bool:
    """Whether mutation can move this field between a numeric min and max.

    ``bool`` is deliberately excluded: it subclasses ``int``, but a grow factor
    applied to a flag means nothing. Every member of a union must be a scalar
    number, so ``float | list[float]`` (asymmetric clip bounds) is not mutable.
    """
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        members = [arg for arg in get_args(annotation) if arg is not type(None)]
        return bool(members) and all(_is_numeric(arg) for arg in members)
    return annotation in (int, float)


def _mutable_names(algorithm: type[AlgoSpec]) -> dict[str, str]:
    """Map every spelling of a mutable hyperparameter to its canonical field.

    Aliases have to be in here: ``lr`` is how most manifests spell LLMPPO's
    ``lr_actor``, and rejecting it would fail documents that run today.

    :param algorithm: The resolved algorithm's spec class.
    :type algorithm: type[AlgoSpec]
    :returns: Accepted spelling -> field name.
    :rtype: dict[str, str]
    """
    names: dict[str, str] = {}
    for field, info in algorithm.model_fields.items():
        names[field] = field
        alias = info.validation_alias
        if isinstance(alias, AliasChoices):
            names.update({c: field for c in alias.choices if isinstance(c, str)})
    return names


AlgorithmSection = Annotated[
    AlgoSpec,
    BeforeValidator(resolve_algorithm),
    PlainSerializer(_serialize_algorithm, return_type=dict[str, Any]),
]
SelectionSection = Annotated[
    SelectionStrategySpec,
    BeforeValidator(_default_selection_strategy),
]


# Validators without a leading underscore are public: a subclass may override them by name.
class TrainingManifest(BaseModel):
    """A whole training run, described once."""

    model_config = ConfigDict(extra="forbid")

    api_version: str = Field(
        default=API_VERSION,
        validation_alias=AliasChoices("apiVersion", "api_version"),
        serialization_alias="apiVersion",
        description="Contract version this manifest is written against.",
    )
    algorithm: AlgorithmSection = Field(
        description="What to train, and the hyperparameters it trains with."
    )
    environment: EnvSpec = Field(
        description="What to train against: an environment, or a dataset."
    )
    training: TrainingSpec = Field(
        default_factory=TrainingSpec,
        description="How long to train for, and how often to evaluate and checkpoint.",
    )
    network: FinetuningNetworkSpec | None = Field(
        default=None,
        title="Network",
        description="The pretrained model to fine-tune.",
    )
    mutation: SkipJsonSchema[MutationSpec | None] = Field(
        default=None,
        json_schema_extra=HIDDEN_PLATFORM_WRITTEN,
        description="What population-based HPO may change between evolution rounds.",
    )
    replay_buffer: BufferSpec | None = Field(
        default=None,
        title="Replay Buffer",
        description="Where collected experience is held before it is learned from.",
    )
    selection_strategy: SkipJsonSchema[SelectionSection | None] = Field(
        default=None,
        json_schema_extra=HIDDEN_PLATFORM_WRITTEN,
        validation_alias=AliasChoices("selection_strategy", "tournament_selection"),
        serialization_alias="tournament_selection",
        description="How the population is ranked and replaced at each evolution round.",
    )
    # Whether Arena filled an unset network.max_context_length with its default.
    _context_defaulted: bool = PrivateAttr(default=False)

    @model_validator(mode="wrap")
    @classmethod
    def _fill_run_settings(
        cls, data: object, handler: ModelWrapValidatorHandler[Self]
    ) -> Self:
        """Fill what a run leaves unset: bundled environment settings, then the default context."""
        if not isinstance(data, dict):
            return handler(data)
        filled: dict[str, Any] = {str(key): value for key, value in data.items()}
        for section in ("algorithm", "network"):
            if isinstance(filled.get(section), dict):
                filled[section] = {**filled[section]}
        apply_bundled_run_settings(filled)
        defaulted = fill_default_context_length(filled)
        manifest = handler(filled)
        manifest._context_defaulted = defaulted
        return manifest

    @property
    def context_defaulted(self) -> bool:
        """Whether network.max_context_length is the default Arena filled in."""
        return self._context_defaulted

    @model_validator(mode="before")
    @classmethod
    def _tag_environment(cls, data: object) -> object:
        """Tag the environment with the branch the algorithm implies, when unset.

        The environment union discriminates on ``env_type``, but a manifest for
        an algorithm that only ever trains against one kind of environment does
        not have to repeat it.
        """
        if not isinstance(data, dict):
            return data
        environment = data.get("environment")
        algorithm = data.get("algorithm")
        if not isinstance(environment, dict) or "env_type" in environment:
            return data
        if not isinstance(algorithm, dict) or "name" not in algorithm:
            return data
        try:
            spec_cls = MANIFEST_REGISTRY.get(algorithm["name"])
        except KeyError:
            # Inferring env_type is all this does. Resolving the algorithm
            # section reports an unknown name as a validation error; raising
            # here would beat it to it with a traceback instead.
            return data
        tagged = {**environment, "env_type": str(spec_cls.env_type)}
        if spec_cls.objective is not None and "objective" not in tagged:
            tagged["objective"] = spec_cls.objective
        return {**data, "environment": tagged}

    @model_validator(mode="after")
    def _check_api_version(self) -> Self:
        if self.api_version != API_VERSION:
            msg = (
                f"Unsupported manifest apiVersion {self.api_version!r}; "
                f"this build of the manifest contract speaks {API_VERSION!r}."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_mutated_hyperparameters_exist(self) -> Self:
        """Reject HPO mutating a name the algorithm does not have, or a non-numeric field."""
        if self.mutation is None or not self.mutation.rl_hp_selection:
            return self

        spec = type(self.algorithm)
        accepted = _mutable_names(spec)
        for spelling in self.mutation.rl_hp_selection:
            field = accepted.get(spelling)
            if field is None:
                near = get_close_matches(spelling, sorted(accepted), n=1)
                hint = f" Did you mean {near[0]!r}?" if near else ""
                msg = (
                    f"mutation.rl_hp_selection cannot mutate {spelling!r}: "
                    f"{self.algorithm.name} has no such hyperparameter.{hint}"
                )
                raise ValueError(msg)
            if not _is_numeric(spec.model_fields[field].annotation):
                msg = (
                    f"mutation.rl_hp_selection cannot mutate {spelling!r}: "
                    f"{self.algorithm.name} declares it, but mutation moves a "
                    "value between a numeric min and max and this is not a number."
                )
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_mutation_ranges_inside_field_limits(self) -> Self:
        """Reject a mutation window outside the hyperparameter it changes."""
        if self.mutation is None or not self.mutation.rl_hp_selection:
            return self
        spec = type(self.algorithm)
        accepted = _mutable_names(spec)
        for spelling, bounds in self.mutation.rl_hp_selection.items():
            field = spec.model_fields[accepted[spelling]]
            try:
                require_mutation_range_inside_field(bounds, field)
            except ValueError as exc:
                msg = f"mutation.rl_hp_selection cannot mutate {spelling!r}: {exc}"
                raise ValueError(msg) from exc
        return self

    @model_validator(mode="after")
    def _check_mutated_sampling_support(self) -> Self:
        """Reject HPO moving ``temperature`` or ``top_p`` to values rollout cannot score."""
        if (
            not isinstance(self.algorithm, RolloutLLMSpec)
            or self.mutation is None
            or not self.mutation.rl_hp_selection
        ):
            return self
        accepted = _mutable_names(type(self.algorithm))
        lowest = {
            "temperature": self.algorithm.temperature,
            "top_p": self.algorithm.top_p,
        }
        for spelling, bounds in self.mutation.rl_hp_selection.items():
            field = accepted[spelling]
            if field in lowest:
                lowest[field] = min(lowest[field], bounds.min)
        prefix = "mutation.rl_hp_selection can reach sampling rollout cannot score"
        try:
            records_sampling_supports(
                lowest["temperature"],
                self.algorithm.top_k,
                lowest["top_p"],
                self.algorithm.min_p,
                self.algorithm.constrain_answer_pattern,
            )
        except ValueError as exc:
            msg = f"{prefix}: {exc}"
            raise ValueError(msg) from exc
        return self

    @model_validator(mode="after")
    def _resolve_selection_brackets(self) -> Self:
        if isinstance(self.selection_strategy, MultiFrequencySelectionSpec):
            self.selection_strategy.resolve_brackets(self.training.pop_size)
        return self

    @model_validator(mode="after")
    def _empty_mutation_is_a_noop(self) -> Self:
        """An empty ``mutation: {}`` is the same no-op as omitting the section."""
        if self.mutation is not None and not self.mutation.model_fields_set:
            self.mutation = MutationSpec(probabilities=no_op_mutation_probabilities())
        return self

    @model_validator(mode="before")
    @classmethod
    def _reject_llm_epochs(cls, data: object) -> object:
        """An LLM run's length is training.max_steps; unset over dataset rows it trains one pass.

        It runs before the training section validates, so the section's own
        epoch checks do not answer an LLM run with advice it cannot follow.
        """
        if not isinstance(data, dict) or not names_llm_algorithm(data.get("algorithm")):
            return data
        training = data.get("training")
        if isinstance(training, BaseModel):
            training = training.model_dump()
        if not isinstance(training, dict):
            return data
        named = [
            f"training.{field}"
            for field in ("num_epochs", "evo_epochs")
            if training.get(field) is not None
        ]
        if not named:
            return data
        msg = (
            f"{' and '.join(named)} are not used by LLM runs; set "
            "training.max_steps and training.evo_steps instead. Leave max_steps "
            "unset to train one pass over a dataset."
        )
        raise ValueError(msg)

    @model_validator(mode="after")
    def resolve_cross_section_defaults(self) -> Self:
        """Fill in the values one section implies for another."""
        if self.training.evo_steps is None and self.training.evo_epochs is None:
            # An epoch-driven run states the interval as evo_epochs; the trainer
            # turns it into steps once it knows the dataset size, so defaulting
            # evo_steps here would put a second, contradictory answer in the payload.
            self.training.evo_steps = min(
                type(self.algorithm).default_evo_steps, self.training.max_steps
            )

        if isinstance(self.network, FinetuningNetworkSpec):
            copy_when_unset(
                self.algorithm,
                "pretrained_model_name_or_path",
                self.network.pretrained_model_name_or_path,
                source_set=True,
            )
            copy_when_unset(
                self.algorithm,
                "weights_path",
                self.network.weights_path,
                source_set="weights_path" in self.network.model_fields_set,
            )
            # The payload keeps the model fields on the network section only.
            if isinstance(self.algorithm, LLMAlgorithmSpec):
                self.network.weights_path = self.algorithm.weights_path
            copy_when_unset(
                self.algorithm,
                "max_model_len",
                self.network.max_context_length,
                source_set=True,
            )
            if "lora_config" in type(self.algorithm).model_fields:
                self._resolve_lora_config(self.network)

        if (
            isinstance(self.algorithm, RolloutLLMSpec)
            and self.environment.segment_prompt_tokens is not None
            and self.algorithm.max_output_tokens is not None
        ):
            # The harness keeps every prompt within segment_prompt_tokens, so a
            # segment's row is at most that plus one turn's generation. An
            # answer continuation generates up to max_output_tokens more.
            turn_tokens = self.algorithm.max_output_tokens * (
                2 if self.algorithm.answer_continuation else 1
            )
            copy_when_unset(
                self.algorithm,
                "max_row_tokens",
                min(
                    self.environment.segment_prompt_tokens + turn_tokens,
                    self.algorithm.max_model_len,
                ),
                source_set=True,
            )

        if isinstance(self.algorithm, RolloutLLMSpec):
            # Ray validates with training.rollout_mode dropped, so an async run
            # starts the same default engine.
            if self.algorithm.vllm_config is None:
                self.algorithm.vllm_config = default_rollout_vllm_config()
            # A share the catalogue names for the model wins over the manifest's.
            info = SUPPORTED_MODEL_INFO.get(
                self.algorithm.pretrained_model_name_or_path or ""
            )
            if info is not None and info.rollout_gpu_memory_utilization is not None:
                self.algorithm.vllm_config.gpu_memory_utilization = (
                    info.rollout_gpu_memory_utilization
                )
        return self

    @model_validator(mode="after")
    def _check_max_output_tokens(self) -> Self:
        """A completion cap has to fit inside the model's context length."""
        if isinstance(self.algorithm, RolloutLLMSpec):
            self.algorithm.check_output_fits_context()
        return self

    @model_validator(mode="after")
    def check_async_rollout_buffer(self) -> Self:
        """Async rollout queues generated groups in the LLM buffer; default it when unset."""
        if not self.training.async_rollout:
            return self
        if self.replay_buffer is None:
            self.replay_buffer = LLMRolloutBufferSpec()
        return self

    @model_validator(mode="after")
    def _default_async_rollout_fields(self) -> Self:
        """A run Ray rolls out async gets the LLM rollout buffer and the oldest-turn stamp unless set."""
        if not async_rollout(self) or explicit_colocated_rollout(self):
            return self
        if self.replay_buffer is None:
            self.replay_buffer = LLMRolloutBufferSpec()
        if self.training.rollout_version_stamp is None:
            self.training.rollout_version_stamp = "oldest_turn"
        return self

    @model_validator(mode="after")
    def _check_rollout_engines(self) -> Self:
        """Only algorithms that generate their own rollouts start rollout engines."""
        blocker = rollout_engines_blocker(
            async_rollout_algorithm(type(self.algorithm)),
            self.training.rollout_engines_per_agent,
        )
        if blocker is not None:
            raise ValueError(blocker)
        return self

    @model_validator(mode="after")
    def check_cpu_offload_colocated(self) -> Self:
        """Full FSDP CPU offload cannot run beside async rollout engines."""
        blocker = cpu_offload_blocker(
            self.algorithm.fsdp,
            async_rollout_algorithm(type(self.algorithm))
            and self.training.async_rollout,
        )
        if blocker is not None:
            raise ValueError(blocker)
        return self

    @model_validator(mode="after")
    def _check_async_stop_sequences(self) -> Self:
        """Async turns generate without detokenizing, so vLLM cannot match stop strings."""
        if not (
            self.training.async_rollout
            and isinstance(self.algorithm, RolloutLLMSpec)
            and self.algorithm.vllm_config is not None
            and self.algorithm.vllm_config.stop_sequences
        ):
            return self
        msg = (
            "vllm_config.stop_sequences is not supported with async rollout; end "
            "turns with stop tokens or the env's turn protocol."
        )
        raise ValueError(msg)

    @model_validator(mode="after")
    def check_dataset_rollout_rubric(self, info: ValidationInfo) -> Self:
        """A rollout over dataset rows needs a rubric, unless OPD trains on its teacher alone."""
        environment = self.environment
        if (
            (info.context or {}).get(REWARD_FILE_AT_SUBMIT)
            or not environment.dataset_backed_rollout
            or environment.rubric_file_path is not None
            or environment.reward is not None
            or isinstance(self.algorithm, OPDSpec)
        ):
            return self
        if (info.context or {}).get(IN_SNOWFLAKE_APP):
            msg = (
                "A dataset-backed rollout needs environment.reward; the Snowflake "
                "app takes no rubric file. Only OPD may leave it unset, and then "
                "every turn scores 0."
            )
            raise ValueError(msg)
        msg = (
            "A dataset-backed rollout needs reward or rubric_file_path; only "
            "OPD may leave both unset, and then every turn scores 0."
        )
        raise ValueError(msg)

    @model_validator(mode="after")
    def check_distillation_teacher(self) -> Self:
        """The teacher rescores each rollout's text, so the env must be text-only."""
        if not isinstance(self.algorithm, OPDSpec):
            return self
        if not self.environment.text_only:
            msg = (
                "algorithm.teacher rescores each rollout's text, so the "
                "environment must be a text-only LLM rollout env (modality text "
                "or a dataset-backed rollout)."
            )
            raise ValueError(msg)
        model = self.algorithm.teacher.model
        if model is None:
            return self
        teacher_info = SUPPORTED_MODEL_INFO.get(model.pretrained_model_name_or_path)
        if teacher_info is None:
            msg = (
                f"algorithm.teacher.model {model.pretrained_model_name_or_path!r} is "
                "not in the model catalog, so its engine memory cannot be sized."
            )
            raise ValueError(msg)
        student_id = self.algorithm.pretrained_model_name_or_path
        student_info = SUPPORTED_MODEL_INFO.get(student_id) if student_id else None
        if (
            student_info is not None
            and student_info.vocab_size != teacher_info.vocab_size
        ):
            msg = (
                "algorithm.teacher.model scores the student's token ids, so both "
                f"need one vocabulary; {student_id} has {student_info.vocab_size} "
                f"tokens and {model.pretrained_model_name_or_path} has "
                f"{teacher_info.vocab_size}."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def check_pretrained_model_resolved(self) -> Self:
        """An LLM run has to name a model, from the algorithm or network section."""
        if self.algorithm.pretrained_model_name_or_path is None:
            msg = (
                "Required field 'pretrained_model_name_or_path' wasn't found in "
                "the manifest. This is required for LLM finetuning algorithms, "
                "and can be added under either the 'algorithm' or 'network' "
                "sections."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def check_thinking_budget_parser(self) -> Self:
        """vLLM enforces thinking_token_budget only through a reasoning parser."""
        model = SUPPORTED_MODEL_INFO.get(self.algorithm.pretrained_model_name_or_path)
        # Only supported ids have a model_type offline; the Ray driver checks the rest.
        if model is not None:
            self.algorithm.check_thinking_budget_parser(model.model_type)
        return self

    @model_validator(mode="after")
    def _check_speculative_logit_bias(self) -> Self:
        """vLLM rejects ``logit_bias`` requests on a drafting engine."""
        if not isinstance(self.algorithm, RolloutLLMSpec):
            return self
        model = SUPPORTED_MODEL_INFO.get(self.algorithm.pretrained_model_name_or_path)
        # Only supported ids are known offline; the rollout engine checks the rest.
        if model is not None:
            self.algorithm.check_speculative_sampling(model.biases_image_token)
        return self

    @model_validator(mode="after")
    def check_expert_parallel_covers_tensor_parallel(self) -> Self:
        """MoE models need ep >= tp; tensor parallel leaves routed experts whole."""
        fsdp = self.algorithm.fsdp
        if fsdp is None or fsdp.ep >= fsdp.tp:
            return self
        model = SUPPORTED_MODEL_INFO.get(self.algorithm.pretrained_model_name_or_path)
        # Only supported ids are known offline; the trainer checks the rest.
        if model is None or not model.supports_expert_parallel:
            return self
        # Without EP, experts sit on tp times fewer FSDP ranks, so each GPU
        # holds tp times the expert memory. The trainer raises this at startup too.
        msg = (
            f"algorithm.fsdp.ep={fsdp.ep} is below tp={fsdp.tp}: "
            f"{model.hub_id} is a mixture-of-experts model, and tensor parallel "
            "does not split routed experts. Set ep >= tp."
        )
        raise ValueError(msg)

    @staticmethod
    def load(source: str | Path | dict[str, Any]) -> dict[str, Any]:
        """Read a YAML/JSON manifest into a dict, or pass a dict through.

        :param source: Path to a YAML/JSON file, or an already-parsed manifest.
        :type source: str | Path | dict[str, Any]
        :returns: The raw manifest document.
        :rtype: dict[str, Any]
        """
        if isinstance(source, dict):
            return source
        with open(source) as fh:
            return yaml.safe_load(fh)

    @overload
    @classmethod
    def get_validated(
        cls,
        manifest: str | Path | dict[str, Any],
        *,
        mode: Literal["json"] = ...,
    ) -> dict[str, Any]: ...

    @overload
    @classmethod
    def get_validated(
        cls,
        manifest: str | Path | dict[str, Any],
        *,
        mode: Literal["python"],
    ) -> TrainingManifest: ...

    @classmethod
    def get_validated(
        cls,
        manifest: str | Path | dict[str, Any],
        *,
        mode: Literal["json", "python"] = "json",
    ) -> dict[str, Any] | TrainingManifest:
        """Validate a manifest and return it as a payload dict or a model.

        :param manifest: Path to a YAML/JSON file, or a raw manifest dict.
        :type manifest: str | Path | dict[str, Any]
        :param mode: ``"json"`` for the submission payload, ``"python"`` for the
            validated model.
        :type mode: Literal["json", "python"]
        :returns: The submission payload or the validated manifest.
        :rtype: dict[str, Any] | TrainingManifest
        :raises ValueError: When the manifest sets a field in
            :data:`RESOLVED_MEMORY_FIELDS`.
        """
        raw = cls.load(manifest)
        reject_resolved_memory_fields(raw)
        validated = cls.model_validate(raw)
        for warning in target_modules_warnings(raw):
            warn_once(warning)
        return validated if mode == "python" else validated.to_payload()

    def to_payload(self) -> dict[str, Any]:
        """Serialize to the JSON submission payload.

        :returns: A JSON-safe manifest document.
        :rtype: dict[str, Any]
        """
        payload = self.model_dump(mode="json", exclude_none=True, by_alias=True)
        self._set_population_sections(payload)
        if "training" in payload:
            self._drop_unset_training_fields(payload["training"])
            if isinstance(self.algorithm, LLMAlgorithmSpec):
                self._drop_default_hidden_training_fields(payload["training"])
        algorithm = payload.get("algorithm")
        if isinstance(algorithm, dict):
            if isinstance(self.algorithm, RolloutLLMSpec):
                vllm_config = self.algorithm.vllm_config
                vllm_payload = algorithm.get("vllm_config")
                if (
                    isinstance(vllm_payload, dict)
                    and vllm_config is not None
                    and "tensor_parallel_size" not in vllm_config.model_fields_set
                ):
                    vllm_payload.pop("tensor_parallel_size", None)
        if isinstance(self.network, FinetuningNetworkSpec) and isinstance(
            algorithm, dict
        ):
            self._move_model_fields_to_network(algorithm)
        environment = payload.get("environment")
        if isinstance(environment, dict):
            if environment.get("screenshot_field") is None:
                environment.pop("screenshot_field", None)
            if environment.get("goal_images_field") is None:
                environment.pop("goal_images_field", None)
            # Classic env specs reuse this model.
            if (
                isinstance(self.environment, LLMEnvSpec)
                and self.environment.name_only_built_in
            ):
                self._drop_unset_built_in_environment_fields(environment)
        return payload

    def _set_population_sections(self, payload: dict[str, Any]) -> None:
        """Give *payload* the mutation and selection sections its run reads.

        :param payload: The dumped manifest, edited in place.
        :type payload: dict[str, Any]
        """
        if isinstance(self.algorithm, LLMAlgorithmSpec):
            # An LLM run without HPO or a population carries neither section.
            if self.mutation is None or not mutates(self.mutation):
                payload.pop("mutation", None)
            if self.training.pop_size == 1:
                payload.pop("tournament_selection", None)
            return
        # Optional sections dump away under exclude_none. Mutation is filled as
        # a no-op (architecture mutation off) plus rl_hp_selection, and
        # tournament_selection with tournament_size / elitism, so those keys
        # are present for later readers.
        payload.setdefault(
            "mutation",
            MutationSpec(probabilities=no_op_mutation_probabilities()).model_dump(
                mode="json", exclude_none=True
            ),
        )
        payload.setdefault(
            "tournament_selection",
            TournamentSelectionSpec().model_dump(mode="json", exclude_none=True),
        )

    def _resolve_lora_config(self, network: FinetuningNetworkSpec) -> None:
        """Give the algorithm and network one LoRA config, the catalog default when neither sets one.

        A null on either section counts as unset.

        :param network: The manifest's finetuning network section.
        :type network: FinetuningNetworkSpec
        """
        algorithm_lora = self.algorithm.lora_config
        if algorithm_lora is None:
            self.algorithm.model_fields_set.discard("lora_config")
        else:
            check_lora_config_supported(
                network.pretrained_model_name_or_path, algorithm_lora
            )
            if "lora_config" not in network.model_fields_set:
                network.lora_config = algorithm_lora
                return
        copy_when_unset(
            self.algorithm, "lora_config", network.lora_config, source_set=True
        )

    @staticmethod
    def _move_model_fields_to_network(algorithm: dict[str, Any]) -> None:
        """Drop the algorithm's copies of the model, weights, context length and LoRA; ``network`` holds them.

        :param algorithm: The dumped ``algorithm`` section, edited in place.
        :type algorithm: dict[str, Any]
        """
        algorithm.pop("pretrained_model_name_or_path", None)
        algorithm.pop("weights_path", None)
        algorithm.pop("max_model_len", None)
        algorithm.pop("lora_config", None)

    def _drop_unset_built_in_environment_fields(
        self, environment: dict[str, Any]
    ) -> None:
        """Omit environment fields a name-only built-in left unset.

        Ray deep-merges this document over the bundled profile, and a present
        key replaces the profile value.

        :param environment: The dumped ``environment`` section, edited in place.
        :type environment: dict[str, Any]
        """
        authored = self.environment.model_fields_set
        for field_name in type(self.environment).model_fields:
            if field_name not in authored:
                environment.pop(field_name, None)

    def _drop_unset_training_fields(self, training: dict[str, Any]) -> None:
        """Omit dumped training defaults that parsers treat as explicit choices.

        An unset engine count leaves the trainer / engine split to the driver,
        and an unset step cap would dump the default million.

        :param training: The dumped ``training`` section, edited in place.
        :type training: dict[str, Any]
        """
        authored = self.training.model_fields_set
        for field in (
            "training_gpus_per_agent",
            "rollout_mode",
            "rollout_engines_per_agent",
            "max_steps",
        ):
            if field not in authored:
                training.pop(field, None)

    def _drop_default_hidden_training_fields(self, training: dict[str, Any]) -> None:
        """Omit hidden training fields an LLM manifest left unset at their defaults.

        The published schema marks these keys deprecated, so the payload
        carries one only when the manifest set it.

        :param training: The dumped ``training`` section, edited in place.
        :type training: dict[str, Any]
        """
        authored = self.training.model_fields_set
        for name, field in schema_skipped_fields(type(self.training)).items():
            if name not in authored and getattr(self.training, name) == field.default:
                training.pop(field.serialization_alias or name, None)


def async_rollout(manifest: TrainingManifest) -> bool:
    """Whether the run generates on async rollout engines, as :func:`async_rollout_algorithm` decides."""
    return async_rollout_algorithm(type(manifest.algorithm))


def mutates(mutation: MutationSpec) -> bool:
    """Whether a mutation section changes anything between evolution rounds."""
    return bool(
        mutation.rl_hp_selection
        or mutation.probabilities != no_op_mutation_probabilities()
    )


def explicit_colocated_rollout(manifest: TrainingManifest) -> bool:
    """Whether a rollout LLM manifest names ``rollout_mode: colocated``, which only local runs take."""
    return (
        async_rollout(manifest)
        and "rollout_mode" in manifest.training.model_fields_set
        and manifest.training.rollout_mode == "colocated"
    )


def validated_run_manifest(manifest: TrainingManifest) -> TrainingManifest:
    """The manifest validated as Ray runs it, from :func:`apply_run_rollout_mode`.

    :param manifest: Training manifest, validated as written.
    :return: ``manifest`` itself when Ray reads it as written, else the run Ray starts.
    :raises pydantic.ValidationError: When the run cannot start as Ray runs it.
    """
    run_async = async_rollout(manifest)
    if not run_async and "rollout_mode" not in manifest.training.model_fields_set:
        return manifest
    payload = manifest.to_payload()
    apply_run_rollout_mode(payload["training"], run_async)
    return TrainingManifest.model_validate(payload)
