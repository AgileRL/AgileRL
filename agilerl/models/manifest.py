# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Training manifest model for local training configurations."""

from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, get_args, overload

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    Field,
    PlainSerializer,
    model_validator,
)
from typing_extensions import Self

from agilerl import HAS_ARENA_DEPENDENCIES, HAS_LLM_DEPENDENCIES
from agilerl.models.algo import (
    ALGO_REGISTRY,
    AlgoSpec,
    LLMAlgorithmSpec,
)
from agilerl.models.hpo import (
    MultiFrequencySelectionSpec,
    MutationSpec,
    SelectionStrategySpec,
    TournamentSelectionSpec,
)
from agilerl.models.networks import (
    FinetuningNetworkSpec,
    LoraConfigDict,
    NetworkSpec,
    network_arch_is_resolvable,
    normalize_manifest_network,
)
from agilerl.models.training import ReplayBufferSpec, TrainingSpec

if TYPE_CHECKING:
    from agilerl.arena.models import TrainingManifest as ArenaTrainingManifest
    from agilerl.models.env import GymEnvSpec, LLMEnvSpec, OfflineEnvSpec, PzEnvSpec

    EnvSpecType = GymEnvSpec | PzEnvSpec | OfflineEnvSpec | LLMEnvSpec


def _resolve_algorithm(data: dict[str, Any] | AlgoSpec) -> AlgoSpec:
    """Dispatch to the concrete algorithm spec using ``ALGO_REGISTRY``.

    Reads the ``name`` key from the raw dict (e.g. ``"DQN"``), looks up
    the corresponding spec class in the registry, and instantiates it
    with the remaining fields.

    :param data: The raw dict or AlgorithmSpec to resolve.
    :type data: dict[str, Any] | AlgoSpec
    :returns: The resolved AlgorithmSpec.
    :rtype: AlgoSpec
    :raises TypeError: If the input is not a dict or AlgorithmSpec.
    :raises ValueError: If the 'name' field is not present.
    """
    if isinstance(data, AlgoSpec):
        return data
    if isinstance(data, BaseModel):
        # Foreign spec (e.g. an arena algorithm spec) — re-resolve via the core
        # ``ALGO_REGISTRY`` using its serialized form and registered name.
        payload = data.model_dump(
            mode="json",
            exclude_none=True,
            exclude=_ALGO_NON_SERIALIZABLE_FIELDS,
        )
        payload["name"] = getattr(data, "name", None)
    elif isinstance(data, dict):
        payload = dict(data)
    else:
        msg = f"Expected a dict or AlgorithmSpec, got {type(data).__name__}"
        raise TypeError(msg)

    name = payload.pop("name", None)
    if name is None:
        msg = "Algorithm section must include a 'name' field, corresponding to the name of the algorithm class."
        raise ValueError(msg)

    entry = ALGO_REGISTRY.get(name)

    if not HAS_LLM_DEPENDENCIES and issubclass(entry.spec_cls, LLMAlgorithmSpec):
        msg = "LLM dependencies are not installed. Please install them using `pip install agilerl[llm]`."
        raise ImportError(msg)

    return entry.spec_cls(**payload)


def _coerce_environment(data: dict[str, Any] | BaseModel) -> dict[str, Any]:
    """Accept environment spec objects or raw dicts.

    If *data* is a Pydantic ``BaseModel`` (e.g. :class:`ArenaEnvSpec`,
    :class:`GymEnvSpec`), it is serialized via ``model_dump()``.
    Plain dicts are passed through unchanged.

    :param data: An environment spec or raw dict.
    :type data: Any
    :returns: A plain dictionary suitable for manifest serialization.
    :rtype: dict[str, Any]
    """
    if isinstance(data, dict):
        return data
    if isinstance(data, BaseModel):
        return data.model_dump()

    msg = f"Expected a dict or environment spec (BaseModel), got {type(data).__name__}"
    raise TypeError(msg)


def _resolve_network(data: dict[str, Any] | BaseModel) -> dict[str, Any]:
    """Normalise the network section to a validated dict with all defaults filled in.

    Raw dicts are validated through :class:`NetworkSpec` so that default values
    (``layer_norm``, ``init_layers``, ``output_vanish``, ``min_latent_dim``, etc.)
    are included in the serialized output.

    :param data: Network config dict or spec instance.
    :type data: dict[str, Any] | BaseModel
    :returns: A plain dictionary suitable for manifest storage.
    :rtype: dict[str, Any]
    """
    if isinstance(data, FinetuningNetworkSpec):
        return data.model_dump(mode="json")
    if isinstance(data, BaseModel):
        data_dict = data.model_dump()
        encoder_config = getattr(data, "encoder_config", None)
        encoder_dump = data_dict.get("encoder_config")
        if encoder_config is not None and isinstance(encoder_dump, dict):
            encoder_dump.setdefault("arch", encoder_config.arch)
        head_config = getattr(data, "head_config", None)
        head_dump = data_dict.get("head_config")
        if (
            head_config is not None
            and isinstance(head_dump, dict)
            and hasattr(head_config, "arch")
        ):
            head_dump.setdefault("arch", head_config.arch)
        return data_dict

    if isinstance(data, dict) and "pretrained_model_name_or_path" in data:
        return FinetuningNetworkSpec.model_validate(data).model_dump(mode="json")

    normalized = normalize_manifest_network(data)
    if not network_arch_is_resolvable(normalized):
        # Deferred: keep the raw network dict; the trainer resolves the arch
        # from the observation space in Trainer.__init__.
        return normalized
    spec = NetworkSpec.model_validate(normalized)
    data_dict = spec.model_dump()
    data_dict["encoder_config"]["arch"] = spec.encoder_config.arch
    return data_dict


_ALGO_NON_SERIALIZABLE_FIELDS: set[str] = {
    "hp_config",
    "net_config",
    "actor_network",
    "critic_network",
    "critic_networks",
    "actor_networks",
}


def _serialize_algorithm(spec: AlgoSpec) -> dict[str, Any]:
    """Serialize an algorithm spec to a JSON-safe dict for manifest storage.

    Runtime-only fields (PyTorch modules, HP configs, network specs) are
    excluded here rather than on the Pydantic model so that
    ``model_dump(mode="python")`` still returns them for internal use.
    """
    dumped = spec.model_dump(
        mode="json",
        exclude_none=True,
        exclude=_ALGO_NON_SERIALIZABLE_FIELDS,
    )
    dumped["name"] = spec.name
    return dumped


# NOTE: Use of PlainSerializer here I believe results in not being able to serialize a
# TrainingManifest's algorithm section in "python" mode i.e. the non-serializable fields
# are lost. Not really an issue because we serialize the algo spec directly (not through
# TrainingManifest) when building the algorithm instance - but could lead to confusion?
AlgorithmFromManifest = Annotated[
    AlgoSpec,
    BeforeValidator(_resolve_algorithm),
    PlainSerializer(_serialize_algorithm, return_type=dict[str, Any]),
]
EnvironmentFromManifest = Annotated[
    dict[str, Any], BeforeValidator(_coerce_environment)
]
NetworkFromManifest = Annotated[dict[str, Any], BeforeValidator(_resolve_network)]


def _known_field_names(model: BaseModel) -> set[str]:
    """Return every accepted input key for *model*: field names plus aliases.

    Includes declared-but-excluded fields (e.g. ``cudagraphs``) and every
    :class:`~pydantic.AliasChoices` option so aliased keys are not mistaken
    for unknown fields.
    """
    names: set[str] = set()
    for field_name, field in type(model).model_fields.items():
        names.add(field_name)
        if isinstance(field.alias, str):
            names.add(field.alias)
        validation_alias = field.validation_alias
        if isinstance(validation_alias, str):
            names.add(validation_alias)
        elif isinstance(validation_alias, AliasChoices):
            names.update(c for c in validation_alias.choices if isinstance(c, str))
    return names


def _collect_unknown_fields(
    raw: dict[str, Any], validated: TrainingManifest
) -> list[str]:
    """Collect dotted paths of manifest keys dropped during validation.

    Only sections that resolve to a concrete Pydantic model are inspected;
    ``environment`` (a raw passthrough dict) and ``network`` (normalized
    before validation) are skipped to avoid false positives.
    """
    if not isinstance(raw, dict):
        return []

    dumped = validated.model_dump(mode="json", exclude_none=True)

    unknown: list[str] = []
    top_known = _known_field_names(validated) | set(dumped)
    unknown.extend(str(key) for key in raw if key not in top_known)

    sections: dict[str, Any] = {
        "algorithm": validated.algorithm,
        "training": validated.training,
        "mutation": validated.mutation,
        "replay_buffer": validated.replay_buffer,
        "selection_strategy": validated.selection_strategy,
    }
    for section, model in sections.items():
        # selection_strategy still accepts its legacy tournament_selection
        # spelling, so report unknown keys under whichever one was written.
        raw_key = (
            "tournament_selection"
            if section == "selection_strategy" and section not in raw
            else section
        )
        raw_section = raw.get(raw_key)
        if not isinstance(raw_section, dict) or not isinstance(model, BaseModel):
            continue
        known = _known_field_names(model)
        dumped_section = dumped.get(section)
        if isinstance(dumped_section, dict):
            known |= {str(key) for key in dumped_section}
        unknown.extend(f"{raw_key}.{key}" for key in raw_section if key not in known)

    return unknown


def _default_selection_strategy(
    value: dict[str, Any] | BaseModel | None,
) -> dict[str, Any] | BaseModel | None:
    """Inject the default strategy discriminator for config dicts that omit it.

    :param value: Raw selection_strategy value from the manifest: a config dict, an
        already-built selection spec, or None.
    :type value: dict[str, Any] | BaseModel | None
    :returns: The value with strategy defaulted to "tournament" when it was a
        discriminator-less dict; otherwise the value unchanged.
    :rtype: dict[str, Any] | BaseModel | None
    """
    if isinstance(value, dict) and "strategy" not in value:
        return {**value, "strategy": "tournament"}
    return value


_ARENA_DIVERGENT_MUTATION_FIELDS = frozenset({"dormant_threshold"})


def _drop_unset_arena_divergent_mutation_fields(
    manifest: TrainingManifest, data: dict[str, Any]
) -> None:
    """Strip unset :data:`_ARENA_DIVERGENT_MUTATION_FIELDS` keys from an Arena payload.

    ``model_dump`` fills every field with its resolved value regardless of
    whether the caller set it, so a raw dict/YAML manifest that never mentions
    one of these fields reaches Arena without the key at all (Arena's own
    default then applies), while a manifest built as a :class:`TrainingManifest`
    would otherwise forward the core default explicitly. Dropping the keys the
    user never set makes the two submission paths behave the same way.

    :param manifest: The core manifest ``data`` was dumped from.
    :type manifest: TrainingManifest
    :param data: The dumped Arena payload, mutated in place.
    :type data: dict[str, Any]
    """
    mutation_data = data.get("mutation")
    if manifest.mutation is None or mutation_data is None:
        return
    unset = _ARENA_DIVERGENT_MUTATION_FIELDS - manifest.mutation.model_fields_set
    for field in unset:
        mutation_data.pop(field, None)


class TrainingManifest(BaseModel):
    """Pydantic model that validates a full training manifest.

    Handles discriminated parsing of the algorithm section via
    :data:`ALGO_REGISTRY` and pre-processes the network section so
    the encoder discriminator works correctly.

    The ``environment`` section is stored as a raw dict.  Callers may
    pass either a plain dict **or** an environment spec (any
    :class:`~pydantic.BaseModel` subclass such as :class:`ArenaEnvSpec`);
    spec objects are automatically serialized to dicts on validation.
    """

    algorithm: AlgorithmFromManifest
    environment: EnvironmentFromManifest
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    network: NetworkFromManifest | None = Field(default=None)
    mutation: MutationSpec | None = Field(default=None)
    replay_buffer: ReplayBufferSpec | None = Field(default=None)
    selection_strategy: Annotated[
        SelectionStrategySpec | None,
        BeforeValidator(_default_selection_strategy),
    ] = Field(
        default=None,
        validation_alias=AliasChoices("selection_strategy", "tournament_selection"),
        serialization_alias="tournament_selection",
    )

    @property
    def tournament_selection(self) -> TournamentSelectionSpec | None:
        """The configured tournament-selection spec.

        .. deprecated::
            Superseded by :attr:`selection_strategy`, the field this section is now
            declared under. Manifests may still be written with either key, and the
            serialized output is unaffected.

        :returns: The tournament-selection spec, or None when MF-PBT or no strategy
            is configured.
        :rtype: TournamentSelectionSpec | None
        """
        warnings.warn(
            "TrainingManifest.tournament_selection is deprecated and will be removed "
            "in a future release; use TrainingManifest.selection_strategy instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        spec = self.selection_strategy
        return spec if isinstance(spec, TournamentSelectionSpec) else None

    @model_validator(mode="after")
    def _resolve_and_validate_compatibility_with_multi_frequency(self) -> Self:
        """Check the training block is compatible with multi-frequency selection.

        :raises ValueError: If pop_size is unset or the MF-PBT layout is invalid.
        :return: The validated manifest.
        :rtype: Self
        """
        spec = self.selection_strategy
        if not isinstance(spec, MultiFrequencySelectionSpec):
            return self

        from agilerl.hpo.multi_frequency import MultiFrequencySelection

        if "pop_size" not in self.training.model_fields_set:
            msg = "pop_size is required in the training block."
            raise ValueError(msg)

        # Only the bracket sizes are written back: the spec's own validator already
        # resolved the rest
        (
            _,
            _,
            _,
            spec.n_winners,
            spec.n_survivors,
            spec.n_open_for_migration,
            spec.n_losers,
        ) = MultiFrequencySelection._resolve_and_validate(
            self.training.pop_size,
            spec.n_subpopulations,
            spec.evolution_frequency_ratios,
            spec.n_winners,
            spec.n_survivors,
            spec.n_open_for_migration,
            spec.n_losers,
        )
        return self

    @model_validator(mode="after")
    def _process_manifest(self) -> Self:
        """Resolve network fields into algorithm-specific objects."""
        # 'network' component of manifest corresponds to algorithm's underlying networks
        algo_spec_cls = type(self.algorithm)
        if self.network is not None:
            # Resolve the raw dict into the algorithm's concrete NetworkSpec
            # if `net_config` field is present in the algorithm spec.
            net_config_field = algo_spec_cls.model_fields.get("net_config")
            if net_config_field is not None:
                # get the NetworkSpec class from the type annotation and validate
                spec_cls: type[NetworkSpec] | None = next(
                    (
                        t
                        for t in get_args(net_config_field.annotation)
                        if t is not type(None)
                    ),
                    None,
                )
                if spec_cls is not None:
                    if network_arch_is_resolvable(self.network):
                        resolved_net_config = spec_cls.model_validate(self.network)
                    else:
                        # Deferred: leave the raw dict for the trainer to resolve
                        # once the observation space is known.
                        resolved_net_config = dict(self.network)
                    # The concrete `net_config` type is only resolved dynamically
                    # here, so write it back through model_copy rather than a
                    # statically-bound attribute assignment.
                    self.algorithm = self.algorithm.model_copy(
                        update={"net_config": resolved_net_config}
                    )
            # LLM algorithms expect a pretrained model
            elif isinstance(self.algorithm, LLMAlgorithmSpec):
                llm_network = FinetuningNetworkSpec.model_validate(self.network)
                self.algorithm.pretrained_model_name_or_path = (
                    llm_network.pretrained_model_name_or_path
                )
                self.algorithm.max_model_len = llm_network.max_context_length
                # After validation the network spec's lora_config is the resolved
                # peft LoraConfig (or None); LoraConfigDict is only the transient
                # pre-validation input type.
                resolved_lora = llm_network.lora_config
                if not isinstance(resolved_lora, LoraConfigDict):
                    self.algorithm.lora_config = resolved_lora

        if (
            isinstance(self.algorithm, LLMAlgorithmSpec)
            and self.algorithm.pretrained_model_name_or_path is None
        ):
            msg = (
                "Required field 'pretrained_model_name_or_path' wasn't found in the manifest. "
                "This is required for LLM finetuning algorithms, and can be added under either the "
                "'algorithm' or 'network' sections."
            )
            raise ValueError(msg)

        recurrent = bool(getattr(self.algorithm, "recurrent", False))
        net_config = getattr(self.algorithm, "net_config", None)
        if isinstance(net_config, dict):
            simba = bool(net_config.get("simba", False))
        else:
            simba = bool(getattr(net_config, "simba", False))
        if recurrent and simba:
            msg = (
                "`simba` and `recurrent` cannot both be set: a network cannot use "
                "both a SimBa and a recurrent (LSTM) encoder. Enable only one."
            )
            raise ValueError(msg)

        return self

    @staticmethod
    def _network_from_algorithm(algorithm: AlgoSpec) -> Any | None:  # noqa: ANN401 -- value coerced to a dict by the network field's BeforeValidator
        """Resolve the manifest ``network`` section from an algorithm spec."""
        if isinstance(algorithm, LLMAlgorithmSpec):
            return FinetuningNetworkSpec.model_validate(
                {
                    "pretrained_model_name_or_path": (
                        algorithm.pretrained_model_name_or_path
                    ),
                    "max_context_length": algorithm.max_model_len,
                    "lora_config": algorithm.lora_config,
                }
            )
        return getattr(algorithm, "net_config", None)

    @classmethod
    def from_trainer_specs(
        cls,
        *,
        algorithm: AlgoSpec,
        environment: BaseModel,
        training: TrainingSpec,
        mutation: MutationSpec | None = None,
        replay_buffer: ReplayBufferSpec | None = None,
        selection_strategy: SelectionStrategySpec | None = None,
        **kwargs: Any,
    ) -> TrainingManifest:
        """Build a validated core manifest from trainer component specs.

        :param algorithm: Core algorithm spec or registered algorithm name dict.
        :type algorithm: AlgoSpec
        :param environment: Environment spec instance held on the trainer.
        :type environment: BaseModel
        :param training: Training loop parameters.
        :type training: TrainingSpec
        :param mutation: Optional mutation spec.
        :type mutation: MutationSpec | None
        :param replay_buffer: Optional replay-buffer spec.
        :type replay_buffer: ReplayBufferSpec | None
        :param selection_strategy: Selection-strategy spec: tournament or
            multi-frequency selection.
        :type selection_strategy: SelectionStrategySpec | None
        :param kwargs: Accepts the deprecated tournament_selection alias for
            selection_strategy.
        :returns: A validated :class:`TrainingManifest`.
        :rtype: TrainingManifest
        """
        from agilerl.utils.trainer_utils import resolve_deprecated_selection_kwargs

        selection_strategy = resolve_deprecated_selection_kwargs(
            selection_strategy,
            kwargs,
            deprecated_key="tournament_selection",
            caller="TrainingManifest.from_trainer_specs",
        )

        def _coerce(value: BaseModel | None, core_cls: type) -> Any:  # noqa: ANN401 -- returns a dict or spec for a field whose BeforeValidator accepts foreign specs
            """Dump foreign BaseModel inputs (e.g. arena specs) to plain dicts."""
            if value is None or isinstance(value, core_cls):
                return value
            if isinstance(value, BaseModel):
                return value.model_dump(mode="json", exclude_none=True)
            return value

        # A foreign (e.g. arena) spec is dumped to a dict and re-validated against the
        # matching core class; a core MF spec must pass through untouched, since the
        # trainer resolves its bracket sizes onto that same object.
        selection_cls = (
            MultiFrequencySelectionSpec
            if isinstance(selection_strategy, MultiFrequencySelectionSpec)
            else TournamentSelectionSpec
        )

        return cls(
            algorithm=algorithm,
            environment=_coerce_environment(environment),
            training=_coerce(training, TrainingSpec),
            network=cls._network_from_algorithm(algorithm),
            mutation=_coerce(mutation, MutationSpec),
            replay_buffer=_coerce(replay_buffer, ReplayBufferSpec),
            selection_strategy=_coerce(selection_strategy, selection_cls),
        )

    @overload
    @classmethod
    def to_arena_manifest(
        cls,
        manifest: str | Path | dict[str, Any] | TrainingManifest,
        *,
        mode: Literal["json"] = ...,
    ) -> dict[str, Any]: ...

    @overload
    @classmethod
    def to_arena_manifest(
        cls,
        manifest: str | Path | dict[str, Any] | TrainingManifest,
        *,
        mode: Literal["python"],
    ) -> ArenaTrainingManifest: ...

    @classmethod
    def to_arena_manifest(
        cls,
        manifest: str | Path | dict[str, Any] | TrainingManifest,
        *,
        mode: Literal["json", "python"] = "json",
    ) -> dict[str, Any] | ArenaTrainingManifest:
        """Validate a manifest for Arena submission.

        Accepts a core :class:`TrainingManifest`, a raw manifest dict, or a
        YAML/JSON path.

        :param manifest: Manifest source to validate for the Arena platform.
        :type manifest: str | Path | dict[str, Any] | TrainingManifest
        :param mode: ``"json"`` for the submission payload, ``"python"`` for the
            validated arena manifest model.
        :type mode: Literal["json", "python"]
        :returns: Arena submission dict or validated arena manifest model.
        :rtype: dict[str, Any] | Any
        :raises ImportError: If ``agilerl-arena`` is not installed.
        """
        if not HAS_ARENA_DEPENDENCIES:
            msg = (
                "Arena dependencies are not installed. "
                "Please install them using: pip install agilerl-arena"
            )
            raise ImportError(msg)

        from agilerl.arena.models import TrainingManifest as ArenaManifest

        if isinstance(manifest, TrainingManifest):
            data = manifest.model_dump(mode="json", exclude_none=True, by_alias=True)
            _drop_unset_arena_divergent_mutation_fields(manifest, data)
        elif isinstance(manifest, dict):
            data = manifest
        else:
            data = cls._load_yaml(manifest)

        return ArenaManifest.get_validated(data, mode=mode)

    @staticmethod
    def _load_yaml(manifest: str | Path | dict[str, Any]) -> dict[str, Any]:
        """Read a YAML/JSON file or pass through a raw dict."""
        if isinstance(manifest, (str, Path)):
            with open(manifest) as fh:
                return yaml.safe_load(fh)
        return manifest

    @classmethod
    def get_validated(
        cls,
        manifest: str | Path | dict[str, Any],
        *,
        mode: Literal["json", "python"] = "json",
    ) -> dict[str, Any] | TrainingManifest:
        """Validate a YAML file and return a JSON-serializable dict or TrainingManifest.

        :param manifest: Path to a YAML/JSON file, or a raw dict.
        :type manifest: str | Path | dict[str, Any]
        :param mode: The mode to validate the manifest in.
        :type mode: Literal["json", "python"]
        :returns: A JSON-serializable dict or TrainingManifest.
        :rtype: dict[str, Any] | TrainingManifest
        """
        data = TrainingManifest._load_yaml(manifest)
        raw_snapshot = copy.deepcopy(data) if isinstance(data, dict) else {}
        validated = cls.model_validate(data)

        unknown = _collect_unknown_fields(raw_snapshot, validated)
        if unknown:
            formatted = "[" + ", ".join(f'"{name}"' for name in unknown) + "]"
            msg = (
                f"Unrecognized manifest field(s): {formatted}. Manifest keys "
                "must match spec fields exactly; remove or rename them."
            )
            raise ValueError(msg)

        if mode == "json":
            return validated.model_dump(mode="json", exclude_none=True)

        return validated
