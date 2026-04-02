"""Training manifest model for deserializing YAML/JSON training configurations."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field, PlainSerializer

from agilerl.models.algo import (
    ALGO_REGISTRY,
    LLMAlgorithmSpec,
    MultiAgentRLAlgorithmSpec,
    RLAlgorithmSpec,
)
from agilerl.models.hpo import MutationSpec, TournamentSelectionSpec
from agilerl.models.networks import NetworkSpec
from agilerl.models.training import ReplayBufferSpec, TrainingSpec

AlgoSpecT = RLAlgorithmSpec | MultiAgentRLAlgorithmSpec | LLMAlgorithmSpec


def _resolve_algorithm(v: Any) -> AlgoSpecT:
    """Dispatch to the concrete algorithm spec using ``ALGO_REGISTRY``.

    Reads the ``name`` key from the raw dict (e.g. ``"DQN"``), looks up
    the corresponding spec class in the registry, and instantiates it
    with the remaining fields.

    :param v: The raw dict or AlgorithmSpec to resolve.
    :type v: Any
    :returns: The resolved AlgorithmSpec.
    :rtype: AlgoSpecT
    :raises TypeError: If the input is not a dict or AlgorithmSpec.
    :raises ValueError: If the 'name' field is not present.
    """
    if isinstance(v, AlgoSpecT):
        return v
    if not isinstance(v, dict):
        msg = f"Expected a dict or AlgorithmSpec, got {type(v).__name__}"
        raise TypeError(msg)

    data = dict(v)
    name = data.pop("name", None)
    if name is None:
        msg = "Algorithm section must include a 'name' field, corresponding to the name of the algorithm class."
        raise ValueError(msg)

    entry = ALGO_REGISTRY.get(name)
    return entry.spec_cls(**data)


def _coerce_environment(v: Any) -> dict[str, Any]:
    """Accept environment spec objects or raw dicts.

    If *v* is a Pydantic ``BaseModel`` (e.g. :class:`ArenaEnvSpec`,
    :class:`GymEnvSpec`), it is serialized via ``model_dump()``.
    Plain dicts are passed through unchanged.

    :param v: An environment spec or raw dict.
    :type v: Any
    :returns: A plain dictionary suitable for manifest serialization.
    :rtype: dict[str, Any]
    """
    if isinstance(v, dict):
        return v
    if isinstance(v, BaseModel):
        return v.model_dump()
    msg = f"Expected a dict or environment spec (BaseModel), got {type(v).__name__}"
    raise TypeError(msg)


def _resolve_network(v: Any) -> Any:
    """Pre-process the network section for :class:`NetworkSpec` parsing.

    Moves the top-level ``arch`` value into ``encoder_config.arch`` so
    the existing Pydantic discriminator can dispatch the encoder union,
    and strips the manifest-only ``simba`` convenience field.
    """
    if isinstance(v, dict):
        v = dict(v)
        arch = v.pop("arch", None)
        v.pop("simba", None)
        if arch and "encoder_config" in v:
            v["encoder_config"] = dict(v["encoder_config"])
            v["encoder_config"].setdefault("arch", arch)
        return v
    return v


AlgorithmFromManifest = Annotated[
    AlgoSpecT,
    BeforeValidator(_resolve_algorithm),
    PlainSerializer(lambda v: v.to_manifest(), return_type=dict[str, Any]),
]
EnvironmentFromManifest = Annotated[
    dict[str, Any], BeforeValidator(_coerce_environment)
]
NetworkFromManifest = Annotated[NetworkSpec, BeforeValidator(_resolve_network)]


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
