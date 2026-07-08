"""Training manifest model for deserializing YAML/JSON training configurations."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Annotated, Any, Literal, get_args

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

import agilerl.arena.models.algorithms as _arena_algorithms  # noqa: F401
from agilerl.arena.models.algo import (
    ARENA_REGISTRY,
    AlgoSpecT,
    LLMAlgorithmSpec,
)
from agilerl.arena.models.hpo import MutationSpec, TournamentSelectionSpec
from agilerl.arena.models.networks import (
    FinetuningNetworkSpec,
    NetworkSpec,
)
from agilerl.arena.models.training import ReplayBufferSpec, TrainingSpec

logger = logging.getLogger(__name__)

_MANIFEST_ENCODER_ARCHS = ("mlp", "cnn", "lstm", "simba", "multiinput")

_ARCH_TO_NAME: dict[str, str] = {
    "mlp": "EvolvableMLP",
    "cnn": "EvolvableCNN",
    "simba": "EvolvableSimBa",
    "multiinput": "EvolvableMultiInput",
    "lstm": "EvolvableLSTM",
}


def _normalize_network_for_platform(network: dict[str, Any]) -> None:
    """Move ``arch`` from encoder_config to network top level and add ``name``."""
    encoder = network.get("encoder_config")
    if isinstance(encoder, dict):
        arch = encoder.pop("arch", None)
        if arch:
            network.setdefault("arch", arch)

    arch = network.get("arch")
    if arch and "name" not in network:
        network["name"] = _ARCH_TO_NAME.get(arch, "EvolvableMLP")


def _ensure_platform_run_spec_keys(data: dict[str, Any]) -> None:
    """Mutate *data* for Arena run-spec JSON.

    Optional manifest sections use ``model_dump(..., exclude_none=True)``, so keys
    for unset models are omitted.
    """
    for key in ("mutation", "tournament_selection", "network"):
        data.setdefault(key, {})

    if data["network"]:
        _normalize_network_for_platform(data["network"])


def _normalize_network_arch(
    data: dict[str, Any] | BaseModel,
) -> dict[str, Any] | BaseModel:
    """Move a top-level ``arch`` key into ``encoder_config.arch``.

    Raw YAML/JSON manifests place ``arch`` at the network section root,
    but :class:`NetworkSpec` (a discriminated union) expects it nested
    under ``encoder_config``.  This helper bridges the two representations.

    If ``arch`` is missing from both the network root and ``encoder_config``,
    the data is returned unchanged: the Arena backend performs its own
    obs-space-aware ``arch`` resolution server-side, so the client defers to
    it instead of rejecting the manifest.
    """
    # If passing a network spec we already have the correct structure
    if not isinstance(data, dict):
        return data

    data = dict(data)
    top_level_arch: str | None = data.pop("arch", None)
    encoder_config = data.get("encoder_config")
    nested_arch: str | None = (
        encoder_config.get("arch") if isinstance(encoder_config, dict) else None
    )
    arch = top_level_arch or nested_arch

    # If arch is not found, defer resolution to the server (obs-space-aware).
    if arch is None:
        return data

    if encoder_config is None:
        data["encoder_config"] = {"arch": arch}
    else:
        data["encoder_config"] = dict(encoder_config)
        data["encoder_config"].setdefault("arch", arch)

    return data


def _network_has_arch(network: dict[str, Any]) -> bool:
    """Whether a raw network dict declares a resolvable ``arch``.

    Checks both the network root and ``encoder_config`` (the two places a
    manifest may place it). Used to decide whether the network section can be
    validated client-side, or must be deferred to the Arena backend's own
    obs-space-aware resolution.

    :param network: Raw network section dict.
    :type network: dict[str, Any]
    :returns: ``True`` if ``arch`` is present at either location.
    :rtype: bool
    """
    if network.get("arch"):
        return True
    encoder_config = network.get("encoder_config")
    return isinstance(encoder_config, dict) and bool(encoder_config.get("arch"))


def _resolve_algorithm(data: Any) -> AlgoSpecT:
    """Dispatch to the concrete algorithm spec using ``ARENA_REGISTRY``.

    Reads the ``name`` key from the raw dict (e.g. ``"DQN"``), looks up
    the corresponding spec class in the registry, and instantiates it
    with the remaining fields.

    :param data: The raw dict or AlgorithmSpec to resolve.
    :type data: Any
    :returns: The resolved AlgorithmSpec.
    :rtype: AlgoSpecT
    :raises TypeError: If the input is not a dict or AlgorithmSpec.
    :raises ValueError: If the 'name' field is not present.
    """
    if isinstance(data, AlgoSpecT):
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


def _coerce_environment(data: Any) -> dict[str, Any]:
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

    Raw dicts are validated through :class:`NetworkSpec` so that default values are included in the serialized output.

    If the raw dict has no resolvable ``arch`` (at the network root or nested
    in ``encoder_config``), the network section is returned unchanged: the
    Arena backend performs its own obs-space-aware ``arch`` resolution
    server-side, so the client passes the raw section through rather than
    validating (and rejecting) it.

    :param data: Network config dict or spec instance.
    :type data: Any
    :returns: A plain dictionary suitable for manifest storage.
    :rtype: dict[str, Any]
    """
    if isinstance(data, FinetuningNetworkSpec):
        return data.model_dump(mode="json")
    if isinstance(data, BaseModel):
        data_dict = data.model_dump()
        if isinstance(data, NetworkSpec):
            data_dict["encoder_config"]["arch"] = data.encoder_config.arch
        return data_dict

    if isinstance(data, dict) and "pretrained_model_name_or_path" in data:
        return FinetuningNetworkSpec.model_validate(data).model_dump(mode="json")

    if isinstance(data, dict) and not _network_has_arch(data):
        # Deferred: hand the raw network section to the server to validate.
        return data

    normalized = _normalize_network_arch(data)
    spec = NetworkSpec.model_validate(normalized)
    data_dict = spec.model_dump()
    data_dict["encoder_config"]["arch"] = spec.encoder_config.arch
    return data_dict


def _serialize_algorithm(spec: AlgoSpecT) -> dict[str, Any]:
    """Serialize an algorithm spec to a JSON-safe dict for manifest storage."""
    dumped = spec.model_dump(mode="json", exclude_none=True)
    dumped["name"] = spec.name
    return dumped


# NOTE: Use of PlainSerializer here I believe results in not being able to serialize a
# TrainingManifest's algorithm section in "python" mode, which is fine since we only serialize
# when submitting jobs to Arena.
AlgorithmFromManifest = Annotated[
    AlgoSpecT,
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
        "tournament_selection": validated.tournament_selection,
    }
    for section, model in sections.items():
        raw_section = raw.get(section)
        if not isinstance(raw_section, dict) or not isinstance(model, BaseModel):
            continue
        known = _known_field_names(model)
        dumped_section = dumped.get(section)
        if isinstance(dumped_section, dict):
            known |= set(dumped_section)
        unknown.extend(f"{section}.{key}" for key in raw_section if key not in known)

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
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    network: NetworkFromManifest | None = Field(default=None)
    mutation: MutationSpec | None = Field(default=None)
    replay_buffer: ReplayBufferSpec | None = Field(default=None)
    tournament_selection: TournamentSelectionSpec | None = Field(default=None)

    @model_validator(mode="after")
    def _process_manifest(self) -> Self:
        """Process the manifest for submission to Arena."""
        # 'network' component of manifest corresponds to algorithm's underlying networks
        algo_spec_cls = type(self.algorithm)
        if self.network is not None:
            # Resolve the raw dict into the algorithm's concrete NetworkSpec
            # if `net_config` field is present in the algorithm spec.
            net_config_field = algo_spec_cls.model_fields.get("net_config")
            if net_config_field is not None:
                # get the NetworkSpec class from the type annotation and validate
                spec_cls: NetworkSpec = next(
                    (
                        t
                        for t in get_args(net_config_field.annotation)
                        if t is not type(None)
                    ),
                    None,
                )
                # Skip when the network has no resolvable `arch`: the client
                # defers to the Arena backend's obs-space-aware resolution
                # rather than validating (and rejecting) the raw section.
                if spec_cls is not None and _network_has_arch(self.network):
                    self.algorithm.net_config = spec_cls.model_validate(self.network)
            # LLM algorithms expect a pretrained model
            elif issubclass(algo_spec_cls, LLMAlgorithmSpec):
                llm_network = FinetuningNetworkSpec.model_validate(self.network)
                self.algorithm.pretrained_model_name_or_path = (
                    llm_network.pretrained_model_name_or_path
                )
                self.algorithm.max_model_len = llm_network.max_context_length
                self.algorithm.lora_config = llm_network.lora_config

        if (
            issubclass(algo_spec_cls, LLMAlgorithmSpec)
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
        if net_config is not None:
            simba = bool(getattr(net_config, "simba", False))
        elif isinstance(self.network, dict):
            simba = bool(self.network.get("simba", False))
        else:
            simba = False
        if recurrent and simba:
            msg = (
                "`simba` and `recurrent` cannot both be set: a network cannot use "
                "both a SimBa and a recurrent (LSTM) encoder. Enable only one."
            )
            raise ValueError(msg)

        return self

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
        """Validate a YAML file and return a JSON dict or :class:`TrainingManifest`.

        :param manifest: Path to a YAML/JSON file, or a raw dict.
        :type manifest: str | Path | dict[str, Any]
        :param mode: ``"json"`` for Arena submission payload, ``"python"`` for the
            validated Pydantic model.
        :type mode: Literal["json", "python"]
        :returns: A JSON-serializable dict or :class:`TrainingManifest`.
        :rtype: dict[str, Any] | TrainingManifest
        """
        data = TrainingManifest._load_yaml(manifest)
        raw_snapshot = copy.deepcopy(data) if isinstance(data, dict) else {}
        validated = cls.model_validate(data)

        unknown = _collect_unknown_fields(raw_snapshot, validated)
        if unknown:
            formatted = "[" + ", ".join(f'"{name}"' for name in unknown) + "]"
            logger.warning("Ignoring unrecognized manifest field(s): %s", formatted)

        if mode == "python":
            return validated

        payload = validated.model_dump(mode="json", exclude_none=True)
        _ensure_platform_run_spec_keys(payload)
        return payload
