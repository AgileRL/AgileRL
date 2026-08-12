# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Training manifest model for deserializing YAML/JSON training configurations."""

from __future__ import annotations

import copy
import logging
import warnings
from pathlib import Path
from typing import Annotated, Any, Literal, overload

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

from agilerl.arena.models.algo import (
    ARENA_REGISTRY,
    AlgoSpec,
    LLMAlgorithmSpec,
)
from agilerl.arena.models.hpo import MutationSpec, TournamentSelectionSpec
from agilerl.arena.models.networks import FinetuningNetworkSpec
from agilerl.arena.models.training import ReplayBufferSpec, TrainingSpec

logger = logging.getLogger(__name__)


def _ensure_platform_run_spec_keys(data: dict[str, Any]) -> None:
    """Mutate *data* for Arena run-spec JSON.

    Optional manifest sections use ``model_dump(..., exclude_none=True)``, so keys
    for unset models are omitted. The keys below are the payload names, which the
    dump produces via ``by_alias=True``: the selection section is declared as
    ``selection_strategy`` but reaches the platform as ``tournament_selection``.
    """
    for key in ("mutation", "tournament_selection", "network"):
        data.setdefault(key, {})


def _resolve_algorithm(data: dict[str, Any] | AlgoSpec) -> AlgoSpec:
    """Dispatch to the concrete algorithm spec using ``ARENA_REGISTRY``.

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
    if not isinstance(data, dict):
        msg = f"Expected a dict or AlgorithmSpec, got {type(data).__name__}"
        raise TypeError(msg)

    payload = dict(data)
    name = payload.pop("name", None)
    if name is None:
        msg = "Algorithm section must include a 'name' field, corresponding to the name of the algorithm class."
        raise ValueError(msg)

    entry = ARENA_REGISTRY.get(name)
    return entry.spec_cls(**payload)


def _coerce_environment(data: dict[str, Any] | BaseModel) -> dict[str, Any]:
    """Accept environment spec objects or raw dicts.

    If *data* is a Pydantic ``BaseModel`` (e.g. :class:`ArenaEnvSpec`,
    :class:`GymEnvSpec`), it is serialized via ``model_dump()``.
    Plain dicts are passed through unchanged.

    :param data: An environment spec or raw dict.
    :type data: dict[str, Any] | BaseModel
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
    """Normalise the network section to a plain dict for manifest storage.

    The encoder architecture is resolved server-side, so the network section is
    passed through untouched. Only the LLM finetuning spec is materialised
    client-side, since its fields feed the algorithm section.

    :param data: Network config dict or spec instance.
    :type data: Any
    :returns: A plain dictionary suitable for manifest storage.
    :rtype: dict[str, Any]
    """
    if isinstance(data, FinetuningNetworkSpec):
        return data.model_dump(mode="json")
    if isinstance(data, dict) and "pretrained_model_name_or_path" in data:
        return FinetuningNetworkSpec.model_validate(data).model_dump(mode="json")
    if isinstance(data, BaseModel):
        return data.model_dump()
    return data


def _serialize_algorithm(spec: AlgoSpec) -> dict[str, Any]:
    """Serialize an algorithm spec to a JSON-safe dict for manifest storage."""
    dumped = spec.model_dump(mode="json", exclude_none=True)
    dumped["name"] = spec.name
    return dumped


# NOTE: Use of PlainSerializer here results in not being able to serialize the
# TrainingManifest's algorithm section in "python" mode, which is fine since we only serialize
# when submitting jobs to Arena.
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

    Includes declared-but-excluded fields and every
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
        # `dumped` is serialized without by_alias, so it is keyed on field names.
        dumped_section = dumped.get(section)
        if isinstance(dumped_section, dict):
            known.update(dumped_section)
        unknown.extend(f"{raw_key}.{key}" for key in raw_section if key not in known)

    return unknown


class TrainingManifest(BaseModel):
    """Pydantic model that validates a full training manifest.

    Handles discriminated parsing of the algorithm section via
    :data:`ARENA_REGISTRY` and pre-processes the network section so
    the encoder discriminator works correctly.

    The ``environment`` section is stored as a raw dict.  Callers may
    pass either a plain dict **or** an environment spec (any
    :class:`~pydantic.BaseModel` subclass such as :class:`EnvSpec`);
    spec objects are automatically serialized to dicts on validation.
    """

    algorithm: AlgorithmFromManifest
    environment: EnvironmentFromManifest
    training: TrainingSpec
    network: NetworkFromManifest | None = Field(default=None)
    mutation: MutationSpec | None = Field(default=None)
    replay_buffer: ReplayBufferSpec | None = Field(default=None)
    selection_strategy: TournamentSelectionSpec | None = Field(
        default=None,
        validation_alias=AliasChoices("selection_strategy", "tournament_selection"),
        serialization_alias="tournament_selection",
    )

    @property
    def tournament_selection(self) -> TournamentSelectionSpec | None:
        """The configured selection strategy.

        .. deprecated::
            Superseded by :attr:`selection_strategy`, the field this section is now
            declared under. Manifests may still be written with either key, and the
            platform payload is unaffected.

        :returns: The selection spec, or None when the section is omitted.
        :rtype: TournamentSelectionSpec | None
        """
        warnings.warn(
            "TrainingManifest.tournament_selection is deprecated and will be removed "
            "in a future release; use TrainingManifest.selection_strategy instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.selection_strategy

    @model_validator(mode="after")
    def _process_manifest(self) -> Self:
        """Process the manifest for submission to Arena."""
        # 'network' component of manifest corresponds to algorithm's underlying
        # networks. `arch` is resolved server-side, so `net_config` is never
        # populated on the algorithm spec here (the platform rejects a set
        # `net_config`).
        # LLM algorithms expect a pretrained model
        if self.network is not None and isinstance(self.algorithm, LLMAlgorithmSpec):
            llm_network = FinetuningNetworkSpec.model_validate(self.network)
            self.algorithm.pretrained_model_name_or_path = (
                llm_network.pretrained_model_name_or_path
            )
            self.algorithm.max_model_len = llm_network.max_context_length

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
        simba = isinstance(self.network, dict) and bool(self.network.get("simba"))
        if recurrent and simba:
            msg = (
                "`simba` and `recurrent` cannot both be set: a network cannot use "
                "both a SimBa and a recurrent (LSTM) encoder. Enable only one."
            )
            raise ValueError(msg)

        return self

    @staticmethod
    def _load_yaml(path: str | Path) -> dict[str, Any]:
        """Read a YAML/JSON file into a dict."""
        with open(path) as fh:
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
        """Validate a YAML file and return a JSON dict or :class:`TrainingManifest`.

        :param manifest: Path to a YAML/JSON file, or a raw dict.
        :type manifest: str | Path | dict[str, Any]
        :param mode: ``"json"`` for Arena submission payload, ``"python"`` for the
            validated Pydantic model.
        :type mode: Literal["json", "python"]
        :returns: A JSON-serializable dict or :class:`TrainingManifest`.
        :rtype: dict[str, Any] | TrainingManifest
        """
        if isinstance(manifest, dict):
            data: dict[str, Any] = manifest
        else:
            data = cls._load_yaml(manifest)
        raw_snapshot = copy.deepcopy(data)
        validated = cls.model_validate(data)

        unknown = _collect_unknown_fields(raw_snapshot, validated)
        if unknown:
            formatted = "[" + ", ".join(f'"{name}"' for name in unknown) + "]"
            logger.warning("Ignoring unrecognized manifest field(s): %s", formatted)

        if mode == "python":
            return validated

        payload = validated.model_dump(mode="json", exclude_none=True, by_alias=True)
        _ensure_platform_run_spec_keys(payload)
        return payload
