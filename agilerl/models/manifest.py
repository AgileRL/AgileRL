"""Training manifest model for local training configurations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, get_args

import yaml
from pydantic import BaseModel, BeforeValidator, Field, PlainSerializer, model_validator
from typing_extensions import Self

from agilerl import HAS_ARENA_DEPENDENCIES, HAS_LLM_DEPENDENCIES, AgentType
from agilerl.models.algo import (
    ALGO_REGISTRY,
    AlgoSpecT,
    LLMAlgorithmSpec,
)
from agilerl.models.hpo import MutationSpec, TournamentSelectionSpec
from agilerl.models.networks import (
    FinetuningNetworkSpec,
    NetworkSpec,
    network_arch_is_resolvable,
    normalize_manifest_network,
)
from agilerl.models.training import ReplayBufferSpec, TrainingSpec

if TYPE_CHECKING:
    from agilerl.models.env import GymEnvSpec, LLMEnvSpec, OfflineEnvSpec, PzEnvSpec

    EnvSpecT = GymEnvSpec | PzEnvSpec | OfflineEnvSpec | LLMEnvSpec


def _resolve_algorithm(data: dict[str, Any] | AlgoSpecT) -> AlgoSpecT:
    """Dispatch to the concrete algorithm spec using ``ALGO_REGISTRY``.

    Reads the ``name`` key from the raw dict (e.g. ``"DQN"``), looks up
    the corresponding spec class in the registry, and instantiates it
    with the remaining fields.

    :param data: The raw dict or AlgorithmSpec to resolve.
    :type data: dict[str, Any] | AlgoSpecT
    :returns: The resolved AlgorithmSpec.
    :rtype: AlgoSpecT
    :raises TypeError: If the input is not a dict or AlgorithmSpec.
    :raises ValueError: If the 'name' field is not present.
    """
    if isinstance(data, AlgoSpecT):
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


def _resolve_network(data: Any) -> dict[str, Any]:
    """Normalise the network section to a validated dict with all defaults filled in.

    Raw dicts are validated through :class:`NetworkSpec` so that default values
    (``layer_norm``, ``init_layers``, ``output_vanish``, ``min_latent_dim``, etc.)
    are included in the serialized output.

    :param data: Network config dict or spec instance.
    :type data: Any
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
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    network: NetworkFromManifest | None = Field(default=None)
    mutation: MutationSpec | None = Field(default=None)
    replay_buffer: ReplayBufferSpec | None = Field(default=None)
    tournament_selection: TournamentSelectionSpec | None = Field(default=None)

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
                spec_cls: NetworkSpec = next(
                    (
                        t
                        for t in get_args(net_config_field.annotation)
                        if t is not type(None)
                    ),
                    None,
                )
                if spec_cls is not None:
                    if network_arch_is_resolvable(self.network):
                        self.algorithm.net_config = spec_cls.model_validate(
                            self.network
                        )
                    else:
                        # Deferred: leave the raw dict for the trainer to resolve
                        # once the observation space is known.
                        self.algorithm.net_config = dict(self.network)
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
    def _network_from_algorithm(algorithm: AlgoSpecT) -> Any | None:
        """Resolve the manifest ``network`` section from an algorithm spec."""
        if algorithm.agent_type == AgentType.LLMAgent:
            return FinetuningNetworkSpec(
                pretrained_model_name_or_path=algorithm.pretrained_model_name_or_path,
                max_context_length=algorithm.max_model_len,
                lora_config=algorithm.lora_config,
            )
        return getattr(algorithm, "net_config", None)

    @classmethod
    def from_trainer_specs(
        cls,
        *,
        algorithm: AlgoSpecT,
        environment: BaseModel,
        training: TrainingSpec,
        mutation: MutationSpec | None = None,
        replay_buffer: ReplayBufferSpec | None = None,
        tournament_selection: TournamentSelectionSpec | None = None,
    ) -> TrainingManifest:
        """Build a validated core manifest from trainer component specs.

        :param algorithm: Core algorithm spec or registered algorithm name dict.
        :type algorithm: AlgoSpecT
        :param environment: Environment spec instance held on the trainer.
        :type environment: BaseModel
        :param training: Training loop parameters.
        :type training: TrainingSpec
        :param mutation: Optional mutation spec.
        :type mutation: MutationSpec | None
        :param replay_buffer: Optional replay-buffer spec.
        :type replay_buffer: ReplayBufferSpec | None
        :param tournament_selection: Optional tournament-selection spec.
        :type tournament_selection: TournamentSelectionSpec | None
        :returns: A validated :class:`TrainingManifest`.
        :rtype: TrainingManifest
        """

        def _coerce(value: Any, core_cls: type) -> Any:
            """Dump foreign BaseModel inputs (e.g. arena specs) to plain dicts."""
            if value is None or isinstance(value, core_cls):
                return value
            if isinstance(value, BaseModel):
                return value.model_dump(mode="json", exclude_none=True)
            return value

        return cls(
            algorithm=algorithm,
            environment=environment,
            training=_coerce(training, TrainingSpec),
            network=cls._network_from_algorithm(algorithm),
            mutation=_coerce(mutation, MutationSpec),
            replay_buffer=_coerce(replay_buffer, ReplayBufferSpec),
            tournament_selection=_coerce(tournament_selection, TournamentSelectionSpec),
        )

    @classmethod
    def to_arena_manifest(
        cls,
        manifest: str | Path | dict[str, Any] | TrainingManifest,
        *,
        mode: Literal["json", "python"] = "json",
    ) -> dict[str, Any]:
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

        if isinstance(manifest, cls):
            data = manifest.model_dump(mode="json", exclude_none=True)
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
        validated = cls.model_validate(data)

        if mode == "json":
            return validated.model_dump(mode="json", exclude_none=True)

        return validated
