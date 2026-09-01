# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The training manifest: the whole contract, in one place."""

from __future__ import annotations

from collections.abc import Callable
from difflib import get_close_matches
from pathlib import Path
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin, overload

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    field_serializer,
    model_validator,
)
from typing_extensions import Self

from agilerl.arena.models.algorithms.base import AlgoSpec, LLMAlgorithmSpec
from agilerl.arena.models.algorithms.ppo import PPOSpec
from agilerl.arena.models.algorithms.rainbow_dqn import RainbowDQNSpec
from agilerl.arena.models.env import EnvSpec
from agilerl.arena.models.hpo import (
    MultiFrequencySelectionSpec,
    MutationSpec,
    SelectionStrategySpec,
    TournamentSelectionSpec,
    no_op_mutation_probabilities,
)
from agilerl.arena.models.networks import (
    FinetuningNetworkSpec,
    MlpSpec,
    NetworkSpec,
    default_colocated_vllm_config,
    dump_network_section,
    network_arch_is_resolvable,
    normalize_manifest_network,
    validate_vllm_lora_target_modules,
)
from agilerl.arena.models.registry import MANIFEST_REGISTRY
from agilerl.arena.models.training import (
    BufferSpec,
    LLMRolloutBufferSpec,
    ReplayBufferSpec,
    TrainingSpec,
)

API_VERSION = "agilerl.com/v1"


def _copy_when_unset(
    target: BaseModel, field: str, value: object, *, source_set: bool
) -> None:
    """Copy *value* onto *target.field* when the target omitted it.

    If both sides were set and they disagree, that is an error rather than a
    silent overwrite.
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


class DeferredNetworkSpec(BaseModel):
    """A network section whose encoder arch is inferred from the observation space.

    Manifests for the classic RL algorithms may omit ``arch``; the trainer picks
    the encoder once it knows the observation space, so ``encoder_config`` cannot
    be validated against a concrete encoder spec here.
    """

    model_config = ConfigDict(extra="forbid")

    latent_dim: int = Field(
        default=32,
        gt=0,
        description="Width of the latent the encoder produces and the head consumes.",
    )
    min_latent_dim: int = Field(
        default=8, gt=0, description="Narrowest the latent may be mutated to."
    )
    max_latent_dim: int = Field(
        default=128, gt=1, description="Widest the latent may be mutated to."
    )
    encoder_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Encoder settings. Left unvalidated here because the architecture "
            "is only known once the observation space is."
        ),
    )
    head_config: MlpSpec | None = Field(
        default=None,
        description="Head mapping the latent onto the algorithm's outputs.",
    )
    random_seed: int | None = Field(
        default=None, description="Seed for weight initialization."
    )
    simba: bool = Field(default=False, description="Use a SimBa encoder.")
    squash_output: bool = Field(
        default=False,
        description="Squash actions through tanh into the action space's bounds.",
    )


def _resolve_algorithm(data: object) -> object:
    """Dispatch the algorithm section to its registered spec class via ``name``."""
    if isinstance(data, BaseModel):
        return data
    if not isinstance(data, dict):
        msg = f"The algorithm section must be a mapping, got {type(data).__name__}."
        raise TypeError(msg)

    payload = dict(data)
    name = payload.pop("name", None)
    if name is None:
        msg = (
            "The algorithm section must include a 'name' field naming a registered "
            f"algorithm. Registered: {', '.join(MANIFEST_REGISTRY.names())}."
        )
        raise ValueError(msg)
    try:
        return MANIFEST_REGISTRY.create(name, **payload)
    except KeyError as err:
        # Surface an unknown name as a validation error like any other bad
        # field, rather than a KeyError a caller cannot tell from a crash.
        msg = (
            f"'name' is not a registered algorithm: {name!r}. "
            f"Registered: {', '.join(MANIFEST_REGISTRY.names())}."
        )
        raise ValueError(msg) from err


def _serialize_algorithm(spec: AlgoSpec) -> dict[str, Any]:
    """Serialize the algorithm section, restoring the ``name`` discriminator.

    ``net_config`` is resolved onto the spec from the network section, which
    is what the payload carries, so it is left out rather than written twice.
    """
    dumped = spec.model_dump(mode="json", exclude_none=True, exclude={"net_config"})
    dumped["name"] = spec.name
    return dumped


def normalize_network_section(data: object) -> object:
    """Nest a top-level ``arch`` and pin a concrete NetworkSpec when it is set.

    Without this, an unknown encoder key fails ``NetworkSpec`` then matches
    ``DeferredNetworkSpec``, whose ``encoder_config`` is an untyped dict.
    """
    if isinstance(data, BaseModel) or not isinstance(data, dict):
        return data
    if "pretrained_model_name_or_path" in data:
        return data
    data = normalize_manifest_network(data)
    if isinstance(data, dict) and network_arch_is_resolvable(data):
        return NetworkSpec.model_validate(data)
    return data


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
    BeforeValidator(_resolve_algorithm),
    PlainSerializer(_serialize_algorithm, return_type=dict[str, Any]),
]
NetworkSection = Annotated[
    FinetuningNetworkSpec | NetworkSpec | DeferredNetworkSpec,
    BeforeValidator(normalize_network_section),
]
SelectionSection = Annotated[
    SelectionStrategySpec,
    BeforeValidator(_default_selection_strategy),
]


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
        description="How long to train for, and how the population evolves.",
    )
    network: NetworkSection | None = Field(
        default=None,
        description=(
            "The model. Either a network to build from scratch, or a "
            "pretrained model to fine-tune."
        ),
    )
    mutation: MutationSpec | None = Field(
        default=None,
        description="What population-based HPO may change between evolution rounds.",
    )
    replay_buffer: BufferSpec | None = Field(
        default=None,
        description="Where collected experience is held before it is learned from.",
    )
    selection_strategy: SelectionSection | None = Field(
        default=None,
        validation_alias=AliasChoices("selection_strategy", "tournament_selection"),
        serialization_alias="tournament_selection",
        description="How the population is ranked and replaced at each evolution round.",
    )

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

    @model_validator(mode="before")
    @classmethod
    def _fold_network_mutation_ranges(cls, data: object) -> object:
        """Move ``mutation.network`` bounds onto the network they constrain.

        They bound the same encoder and head the network section configures, so
        keeping them apart lets the two disagree — a head sized outside its own
        mutation range validates only because nothing compares them.
        """
        if not isinstance(data, dict):
            return data
        # The section arrives as a spec when a caller builds the manifest from
        # objects rather than parsing a document -- from_trainer_specs does.
        mutation = data.get("mutation")
        if isinstance(mutation, BaseModel):
            mutation = mutation.model_dump(exclude_none=True)
        ranges = (mutation or {}).get("network")
        network = data.get("network")
        if not isinstance(ranges, dict) or not isinstance(network, dict):
            return data

        folded = dict(network)
        for key in ("min_latent_dim", "max_latent_dim"):
            if key in ranges:
                folded.setdefault(key, ranges[key])
        for section, bounds in (
            ("encoder_config", ranges.get("encoder")),
            ("head_config", ranges.get("head_net")),
        ):
            if not isinstance(bounds, dict):
                continue
            folded[section] = {**bounds, **(folded.get(section) or {})}
        return {**data, "network": folded}

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
    def _check_network_matches_algorithm(self) -> Self:
        is_llm = isinstance(self.algorithm, LLMAlgorithmSpec)
        if is_llm and isinstance(self.network, (NetworkSpec, DeferredNetworkSpec)):
            msg = (
                f"{self.algorithm.name} fine-tunes a pretrained model, so the "
                "network section must declare 'pretrained_model_name_or_path'."
            )
            raise ValueError(msg)
        if not is_llm and isinstance(self.network, FinetuningNetworkSpec):
            msg = (
                f"{self.algorithm.name} builds its own networks, so the network "
                "section must not declare 'pretrained_model_name_or_path'."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_encoder_compatibility(self) -> Self:
        simba = (
            isinstance(self.network, (NetworkSpec, DeferredNetworkSpec))
            and self.network.simba
        )
        if isinstance(self.algorithm, PPOSpec) and self.algorithm.recurrent and simba:
            msg = (
                "`simba` and `recurrent` cannot both be set: a network cannot use "
                "both a SimBa and a recurrent (LSTM) encoder. Enable only one."
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

    @model_validator(mode="after")
    def _resolve_cross_section_defaults(self) -> Self:
        """Fill in the values one section implies for another."""
        if self.training.evo_steps is None and self.training.evo_epochs is None:
            # An epoch-driven run states the interval as evo_epochs; the trainer
            # turns it into steps once it knows the dataset size, so defaulting
            # evo_steps here would put a second, contradictory answer in the payload.
            self.training.evo_steps = min(
                type(self.algorithm).default_evo_steps, self.training.max_steps
            )

        num_envs = self.environment.num_envs
        env_set_num_envs = "num_envs" in self.environment.model_fields_set
        for field_name in ("num_envs", "vect_noise_dim"):
            _copy_when_unset(
                self.algorithm,
                field_name,
                num_envs,
                source_set=env_set_num_envs,
            )

        if isinstance(self.network, FinetuningNetworkSpec):
            _copy_when_unset(
                self.algorithm,
                "pretrained_model_name_or_path",
                self.network.pretrained_model_name_or_path,
                source_set=True,
            )
            _copy_when_unset(
                self.algorithm,
                "max_model_len",
                self.network.max_context_length,
                source_set=True,
            )
            _copy_when_unset(
                self.algorithm,
                "lora_config",
                self.network.lora_config,
                source_set="lora_config" in self.network.model_fields_set,
            )

        if "use_vllm" in type(self.algorithm).model_fields:
            # Async rollout owns generation on its own engines, so the trainer's
            # in-process vLLM must be off. An omitted use_vllm follows
            # rollout_mode; both sides set and disagree is an error.
            implied = self.training.rollout_mode == "colocated"
            if "use_vllm" not in self.algorithm.model_fields_set:
                self.algorithm.use_vllm = implied
            elif (
                "rollout_mode" in self.training.model_fields_set
                and self.algorithm.use_vllm != implied
            ):
                msg = (
                    f"algorithm.use_vllm={self.algorithm.use_vllm!r} disagrees with "
                    f"training.rollout_mode={self.training.rollout_mode!r}."
                )
                raise ValueError(msg)
            if self.algorithm.use_vllm:
                if self.algorithm.vllm_config is None:
                    self.algorithm.vllm_config = default_colocated_vllm_config()
                else:
                    self.algorithm.vllm_config.sleep_mode = True

        if (
            isinstance(self.algorithm, LLMAlgorithmSpec)
            and self.algorithm.lora_config is not None
        ):
            served_by_vllm = False
            if "use_vllm" in type(self.algorithm).model_fields:
                served_by_vllm = bool(self.algorithm.use_vllm) or (
                    self.training.rollout_mode == "async"
                )
            validate_vllm_lora_target_modules(
                self.algorithm.lora_config.target_modules,
                served_by_vllm=served_by_vllm,
            )

        if (
            isinstance(self.algorithm, RainbowDQNSpec)
            and isinstance(self.replay_buffer, ReplayBufferSpec)
            and self.replay_buffer.n_step_buffer
        ):
            _copy_when_unset(
                self.algorithm,
                "n_step",
                self.replay_buffer.n_step_buffer_args.n_step,
                source_set=True,
            )
        return self

    @model_validator(mode="after")
    def _check_async_rollout_buffer(self) -> Self:
        """Async rollout queues generated groups, so it needs the LLM buffer."""
        if not self.training.async_rollout:
            return self
        if not isinstance(self.replay_buffer, LLMRolloutBufferSpec):
            got = (
                type(self.replay_buffer).__name__
                if self.replay_buffer is not None
                else None
            )
            msg = f"Async rollout requires an LLMRolloutBufferSpec, got {got}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_pretrained_model_resolved(self) -> Self:
        """An LLM run has to name a model, from the algorithm or network section."""
        if (
            isinstance(self.algorithm, LLMAlgorithmSpec)
            and self.algorithm.pretrained_model_name_or_path is None
        ):
            msg = (
                "Required field 'pretrained_model_name_or_path' wasn't found in "
                "the manifest. This is required for LLM finetuning algorithms, "
                "and can be added under either the 'algorithm' or 'network' "
                "sections."
            )
            raise ValueError(msg)
        return self

    @field_serializer("network", mode="wrap")
    def _serialize_network(
        self, value: object, handler: Callable[[object], object], info: object
    ) -> object:
        """Keep ``arch``, the encoder's discriminator, in the dumped section.

        The field is ``exclude=True`` so it never reaches the network kwargs
        the algorithms are built with. A dump without ``arch`` cannot be
        re-validated against the encoder union.
        """
        dumped = handler(value)
        if isinstance(value, NetworkSpec) and isinstance(dumped, dict):
            encoder_config = dumped.get("encoder_config")
            if isinstance(encoder_config, dict):
                encoder_config["arch"] = value.encoder_config.arch
        return dumped

    @property
    def network_arch_is_resolvable(self) -> bool:
        """Whether the encoder architecture is declared rather than inferred."""
        return not isinstance(self.network, DeferredNetworkSpec)

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
        """
        validated = cls.model_validate(cls.load(manifest))
        return validated if mode == "python" else validated.to_payload()

    def algorithm_with_network(self) -> AlgoSpec:
        """Copy the network section onto ``algorithm.net_config`` when the spec has that field."""
        algorithm = self.algorithm
        if self.network is None:
            return algorithm
        net_config_field = type(algorithm).model_fields.get("net_config")
        if net_config_field is None:
            return algorithm
        spec_cls: type[NetworkSpec] | None = next(
            (t for t in get_args(net_config_field.annotation) if t is not type(None)),
            None,
        )
        if spec_cls is None:
            return algorithm
        network = dump_network_section(self.network, exclude_none=True)
        if network_arch_is_resolvable(network):
            resolved: NetworkSpec | dict[str, Any] = spec_cls.model_validate(network)
        else:
            resolved = network
        return algorithm.model_copy(update={"net_config": resolved})

    def to_payload(self) -> dict[str, Any]:
        """Serialize to the JSON submission payload.

        :returns: A JSON-safe manifest document.
        :rtype: dict[str, Any]
        """
        payload = self.model_dump(mode="json", exclude_none=True, by_alias=True)
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
        # The network has no safe default: the trainer infers the encoder from
        # the observation space, and inventing one here would pick an
        # architecture the run never asked for.
        if (
            isinstance(self.network, DeferredNetworkSpec)
            and not self.network.model_fields_set
        ):
            payload.pop("network", None)
        # Omit dumped defaults that parsers treat as explicit choices.
        if (
            "rollout_mode" not in self.training.model_fields_set
            and "training" in payload
        ):
            payload["training"].pop("rollout_mode", None)
        algorithm = payload.get("algorithm")
        if isinstance(algorithm, dict):
            if "num_envs" not in self.algorithm.model_fields_set:
                algorithm.pop("num_envs", None)
            if "vect_noise_dim" not in self.algorithm.model_fields_set:
                algorithm.pop("vect_noise_dim", None)
        return payload
