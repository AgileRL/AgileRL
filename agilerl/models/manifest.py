"""Training manifest model for deserializing YAML/JSON training configurations."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, get_args

import yaml
from pydantic import BaseModel, BeforeValidator, Field, PlainSerializer, model_validator

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.models.algo import (
    ALGO_REGISTRY,
    AlgoSpecT,
    LLMAlgorithmSpec,
)
from agilerl.models.hpo import MutationSpec, TournamentSelectionSpec
from agilerl.models.networks import (
    FinetuningNetworkSpec,
    NetworkSpec,
    normalize_manifest_network,
)
from agilerl.models.training import ReplayBufferSpec, TrainingSpec

if TYPE_CHECKING:
    from agilerl.models.env import (
        ArenaEnvSpec,
        GymEnvSpec,
        LLMEnvSpec,
        OfflineEnvSpec,
        PzEnvSpec,
    )

    EnvSpecT = GymEnvSpec | PzEnvSpec | OfflineEnvSpec | LLMEnvSpec | ArenaEnvSpec


def _resolve_algorithm(data: Any, *, arena_only: bool = False) -> AlgoSpecT:
    """Dispatch to the concrete algorithm spec using ``ALGO_REGISTRY``.

    Reads the ``name`` key from the raw dict (e.g. ``"DQN"``), looks up
    the corresponding spec class in the registry, and instantiates it
    with the remaining fields.

    :param data: The raw dict or AlgorithmSpec to resolve.
    :type data: Any
    :param arena_only: If True, reject algorithms not eligible for Arena.
    :type arena_only: bool
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

    data = dict(data)
    name = data.pop("name", None)
    if name is None:
        msg = "Algorithm section must include a 'name' field, corresponding to the name of the algorithm class."
        raise ValueError(msg)

    entry = ALGO_REGISTRY.get(name)

    if arena_only and not entry.arena:
        supported = ", ".join(sorted(ALGO_REGISTRY.arena_algorithms()))
        msg = f"Algorithm {name!r} is not available on Arena. Available: {supported}"
        raise ValueError(msg)

    if not HAS_LLM_DEPENDENCIES and issubclass(entry.spec_cls, LLMAlgorithmSpec):
        msg = "LLM dependencies are not installed. Please install them using `pip install agilerl[llm]`."
        raise ImportError(msg)

    return entry.spec_cls(**data)


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


def _resolve_network(data: Any) -> dict[str, Any]:
    """Normalise the network section to a plain dict.

    The top-level arch discriminator is moved into the encoder_config.arch field for
    proper dispatching to the correct encoder type.

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

    return normalize_manifest_network(data)


_ALGO_NON_SERIALIZABLE_FIELDS: set[str] = {
    "hp_config",
    "net_config",
    "actor_network",
    "critic_network",
    "critic_networks",
    "actor_networks",
}


def _serialize_algorithm(spec: AlgoSpecT) -> dict[str, Any]:
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
    AlgoSpecT,
    BeforeValidator(_resolve_algorithm),
    PlainSerializer(_serialize_algorithm, return_type=dict[str, Any]),
]
ArenaAlgorithmFromManifest = Annotated[
    AlgoSpecT,
    BeforeValidator(functools.partial(_resolve_algorithm, arena_only=True)),
    PlainSerializer(_serialize_algorithm, return_type=dict[str, Any]),
]
EnvironmentFromManifest = Annotated[
    dict[str, Any], BeforeValidator(_coerce_environment)
]
NetworkFromManifest = Annotated[dict[str, Any], BeforeValidator(_resolve_network)]


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
    training: TrainingSpec
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
                if spec_cls is not None:
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
        """Validate a YAML file and return a JSON-serializable dict or TrainingManifest.

        :param manifest: Path to a YAML/JSON file, or a raw dict.
        :type manifest: str | Path | dict[str, Any]
        :param mode: The mode to validate the manifest in.
        :type mode: Literal["json", "python"]
        :returns: A JSON-serializable dict or TrainingManifest.
        :rtype: dict[str, Any] | TrainingManifest
        """
        data = TrainingManifest._load_yaml(manifest)
        validated = cls.model_validate(data)
        return (
            validated.model_dump(mode="json", exclude_none=True)
            if mode == "json"
            else validated
        )


class ArenaManifest(TrainingManifest):
    """Manifest variant that restricts algorithms to Arena-eligible ones."""

    algorithm: ArenaAlgorithmFromManifest
