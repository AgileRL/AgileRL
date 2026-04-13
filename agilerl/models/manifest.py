"""Training manifest model for deserializing YAML/JSON training configurations."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field, PlainSerializer

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.models.algo import (
    ALGO_REGISTRY,
    LLMAlgorithmSpec,
    MultiAgentRLAlgorithmSpec,
    RLAlgorithmSpec,
)
from agilerl.models.env import (
    ArenaEnvSpec,
    GymEnvSpec,
    LLMEnvSpec,
    OfflineEnvSpec,
    PzEnvSpec,
)
from agilerl.models.hpo import MutationSpec, TournamentSelectionSpec
from agilerl.models.networks import FinetuningNetworkSpec, NetworkSpec
from agilerl.models.training import ReplayBufferSpec, TrainingSpec
from agilerl.typing import ConfigType

AlgoSpecT = RLAlgorithmSpec | MultiAgentRLAlgorithmSpec | LLMAlgorithmSpec
EnvSpecT = GymEnvSpec | PzEnvSpec | OfflineEnvSpec | LLMEnvSpec | ArenaEnvSpec


def _resolve_algorithm(data: ConfigType | AlgoSpecT) -> AlgoSpecT:
    """Dispatch to the concrete algorithm spec using ``ALGO_REGISTRY``.

    Reads the ``name`` key from the raw dict (e.g. ``"DQN"``), looks up
    the corresponding spec class in the registry, and instantiates it
    with the remaining fields.

    :param data: The raw dict or AlgorithmSpec to resolve.
    :type v: ConfigType | AlgoSpecT
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
    if not HAS_LLM_DEPENDENCIES and issubclass(entry.spec_cls, LLMAlgorithmSpec):
        msg = "LLM dependencies are not installed. Please install them using `pip install agilerl[llm]`."
        raise ImportError(msg)

    return entry.spec_cls(**data)


def _coerce_environment(data: ConfigType | BaseModel) -> dict[str, Any]:
    """Accept environment spec objects or raw dicts.

    If *data* is a Pydantic ``BaseModel`` (e.g. :class:`ArenaEnvSpec`,
    :class:`GymEnvSpec`), it is serialized via ``model_dump()``.
    Plain dicts are passed through unchanged.

    :param data: An environment spec or raw dict.
    :type data: ConfigType
    :returns: A plain dictionary suitable for manifest serialization.
    :rtype: dict[str, Any]
    """
    if isinstance(data, dict):
        return data
    if isinstance(data, BaseModel):
        return data.model_dump()

    msg = f"Expected a dict or environment spec (BaseModel), got {type(data).__name__}"
    raise TypeError(msg)


def _resolve_network(
    data: ConfigType | NetworkSpec | FinetuningNetworkSpec,
) -> dict[str, Any]:
    """Normalise the network section to a plain dict.

    The top-level arch discriminator is moved into the encoder_config.arch field for
    proper dispatching to the correct encoder type.

    :param data: Network config dict or spec instance.
    :type data: ConfigType | NetworkSpec | FinetuningNetworkSpec
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

    data = dict(data)
    arch = data.pop("arch", None)
    if arch and "encoder_config" in data:
        data["encoder_config"] = dict(data["encoder_config"])
        data["encoder_config"].setdefault("arch", arch)
    return data


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


# NOTE: Use of PlainSerializer here I believe results in not being able to
# serialize a TrainingManifest's algorithm section in "python" mode i.e. the non-serializable fields
# are lost. Not really an issue because we serialize the algo spec directly (not through TrainingManifest)
# to build the algorithm instance - but could lead to confusion?
AlgorithmFromManifest = Annotated[
    AlgoSpecT,
    BeforeValidator(_resolve_algorithm),
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
