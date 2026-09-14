# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Network specs: arena field models, plus trainer-side encoder helpers and PEFT LoRA."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, get_args, overload

from gymnasium import spaces
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    field_serializer,
    model_validator,
)
from typing_extensions import Self

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.arena.models.networks import (
    CnnSpec,
    ContinuousQNetworkSpec,
    DeterministicActorSpec,
    EncoderType,
    LoraConfigDict,
    LstmSpec,
    MlpActivation,
    MlpSpec,
    MultiInputSpec,
    NetworkSpec,
    QNetworkSpec,
    RainbowQNetworkSpec,
    SimbaSpec,
    StochasticActorSpec,
    ValueNetworkSpec,
    min_max_validator,
)
from agilerl.arena.models.networks import (
    FinetuningNetworkSpec as ArenaFinetuningNetworkSpec,
)

if TYPE_CHECKING:
    from peft import LoraConfig
else:
    # peft is optional and LoraConfig is not a pydantic type; at runtime it is
    # Any so the field annotation stays resolvable without peft installed.
    LoraConfig = Any

__all__ = [
    "CnnSpec",
    "ContinuousQNetworkSpec",
    "DeterministicActorSpec",
    "EncoderType",
    "FinetuningNetworkSpec",
    "LoraConfigDict",
    "LstmSpec",
    "MlpActivation",
    "MlpSpec",
    "MultiInputSpec",
    "NetworkSpec",
    "QNetworkSpec",
    "RainbowQNetworkSpec",
    "SimbaSpec",
    "StochasticActorSpec",
    "ValueNetworkSpec",
    "encoder_spec_for_arch",
    "infer_encoder_arch",
    "min_max_validator",
    "network_arch_is_resolvable",
    "normalize_manifest_network",
]


def infer_encoder_arch(
    observation_space: spaces.Space,
    *,
    recurrent: bool = False,
    simba: bool = False,
) -> Literal["mlp", "cnn", "lstm", "simba", "multiinput"]:
    """Infer the encoder architecture from an observation space.

    Mirrors the branch order in
    :func:`agilerl.utils.evolvable_networks.get_default_encoder_config` and
    :meth:`agilerl.networks.base.EvolvableNetwork._build_encoder` so the schema
    used to validate ``encoder_config`` always matches the encoder that will be
    built. ``simba`` takes precedence over ``recurrent``.

    :param observation_space: The (single-agent or per-agent) observation space.
    :param recurrent: Whether the algorithm requests a recurrent encoder.
    :param simba: Whether the network requests a SimBa encoder.
    :returns: One of ``"mlp"``, ``"cnn"``, ``"lstm"``, ``"simba"``, ``"multiinput"``.
    """
    if isinstance(observation_space, (spaces.Dict, spaces.Tuple)):
        return "multiinput"
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        return "cnn"
    if simba:
        return "simba"
    if recurrent:
        return "lstm"
    return "mlp"


def network_arch_is_resolvable(network: dict) -> bool:
    """Return True if the manifest network section declares an ``arch``.

    Checks the top level and the nested ``encoder_config``. When False, the
    architecture must be inferred from the observation space at build time.
    """
    if not isinstance(network, dict):
        return False
    if network.get("arch"):
        return True
    encoder_config = network.get("encoder_config")
    return isinstance(encoder_config, dict) and bool(encoder_config.get("arch"))


@overload
def normalize_manifest_network(data: dict[str, Any]) -> dict[str, Any]: ...
@overload
def normalize_manifest_network(data: object) -> object: ...
def normalize_manifest_network(data: object) -> object:
    """Move a top-level ``arch`` key into ``encoder_config.arch`` when present.

    Raw YAML/JSON manifests place ``arch`` at the network section root, but
    :class:`NetworkSpec` (a discriminated union) expects it nested under
    ``encoder_config``. When ``arch`` is absent it is inferred later from the
    observation space, so this helper leaves the data unchanged rather than
    raising.
    """
    if not isinstance(data, dict):
        return data

    data = dict(data)
    top_level_arch = data.pop("arch", None)
    encoder_config = data.get("encoder_config")
    nested_arch = (
        encoder_config.get("arch") if isinstance(encoder_config, dict) else None
    )
    arch = top_level_arch or nested_arch

    if arch is None:
        # Deferred: architecture inferred from the observation space later.
        return data

    if encoder_config is None:
        data["encoder_config"] = {"arch": arch}
    else:
        data["encoder_config"] = dict(encoder_config)
        data["encoder_config"].setdefault("arch", arch)

    return data


def encoder_spec_for_arch(arch: str) -> type[BaseModel]:
    """Return the encoder spec class (``MlpSpec``, ``CnnSpec``, ...) for an arch literal.

    Single source of truth mapping an ``arch`` string (as produced by
    :func:`infer_encoder_arch`) to the concrete pydantic spec that validates
    that encoder's ``encoder_config``.

    :param arch: The encoder architecture literal (e.g. ``"mlp"``, ``"cnn"``).
    :type arch: str
    :returns: The encoder spec class whose ``arch`` field matches.
    :rtype: type[BaseModel]
    """
    for member in get_args(EncoderType):
        if member.model_fields["arch"].default == arch:
            return member
    msg = f"Unknown encoder arch: {arch!r}"
    raise ValueError(msg)


def _peft_lora_config_to_manifest_dict(cfg: LoraConfig) -> dict[str, Any]:
    """Map a PEFT :class:`~peft.LoraConfig` to manifest / :class:`LoraConfigDict` keys.

    :param cfg: The PEFT :class:`~peft.LoraConfig` to convert.
    :type cfg: LoraConfig
    :returns: The manifest / :class:`LoraConfigDict` keys.
    :rtype: dict[str, Any]
    """
    task_type = cfg.task_type
    if hasattr(task_type, "value"):
        task_type = task_type.value
    task_type = str(task_type)

    tm = cfg.target_modules
    if tm is None:
        tm_out: list[str] | str | set[str] = "all-linear"
    elif isinstance(tm, set):
        tm_out = sorted(tm)
    else:
        tm_out = tm

    target_parameters = getattr(cfg, "target_parameters", None)
    return {
        "lora_r": cfg.r,
        "lora_alpha": cfg.lora_alpha,
        "target_modules": tm_out,
        "target_parameters": (sorted(target_parameters) if target_parameters else None),
        "task_type": task_type,
        "lora_dropout": cfg.lora_dropout,
    }


class FinetuningNetworkSpec(ArenaFinetuningNetworkSpec):
    """LLM finetuning network spec: arena fields plus PEFT ``LoraConfig`` coerce."""

    # Allow arbitrary types so the resolved peft ``LoraConfig`` (Any at runtime,
    # the real type under TYPE_CHECKING) can live in the ``lora_config`` field.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Validated from a manifest ``LoraConfigDict``; ``_resolve_lora_config``
    # replaces it in place with the peft ``LoraConfig`` after validation.
    lora_config: LoraConfigDict | LoraConfig | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def _coerce_peft_lora(cls, data: object) -> object:
        """Accept a peft ``LoraConfig`` instance and convert it to a dict
        that Pydantic can validate as :class:`LoraConfigDict`.
        """
        if not isinstance(data, dict):
            return data
        lc = data.get("lora_config")
        if lc is not None and not isinstance(lc, (dict, LoraConfigDict)):
            data = dict(data)
            data["lora_config"] = _peft_lora_config_to_manifest_dict(
                cast("LoraConfig", lc)
            )
        return data

    @model_validator(mode="after")
    def _resolve_lora_config(self) -> Self:
        """Convert :class:`LoraConfigDict` to a peft ``LoraConfig`` at runtime."""
        if isinstance(self.lora_config, LoraConfigDict):
            if not HAS_LLM_DEPENDENCIES:
                msg = "LLM dependencies are required to resolve LoRA configuration."
                raise ImportError(msg)
            from peft import LoraConfig as _LoraConfig  # optional extra: llm

            peft_lora = self.lora_config.model_dump()
            peft_lora["r"] = peft_lora.pop("lora_r")
            self.lora_config = _LoraConfig(**peft_lora)
        return self

    @field_serializer("lora_config")
    def _serialize_lora_config(
        self, value: LoraConfigDict | LoraConfig | None, info: SerializationInfo
    ) -> LoraConfigDict | LoraConfig | dict[str, Any] | None:
        if info.mode != "json":
            return value
        if value is None:
            return None
        if isinstance(value, LoraConfigDict):
            return value.model_dump(mode="json")
        return _peft_lora_config_to_manifest_dict(value)
