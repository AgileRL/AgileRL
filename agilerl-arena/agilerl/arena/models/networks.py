# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The manifest's ``network`` section: encoder, head and LLM fine-tuning specs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeVar, get_args, overload

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    field_serializer,
    model_validator,
)
from typing_extensions import Self

T = TypeVar("T", bound=BaseModel)

MlpActivation = Literal[
    "Tanh",
    "GELU",
    "ReLU",
    "ELU",
    "LeakyReLU",
    "Sigmoid",
    "Softplus",
    "Softmax",
    "GumbelSoftmax",
    "PReLU",
    "Identity",
]


def min_max_validator(min_field: str, max_field: str) -> Callable[[T], T]:
    """Validate that a field is less than or equal to another field.

    :param min_field: The name of the minimum field
    :type min_field: str
    :param max_field: The name of the maximum field
    :type max_field: str
    :return: A validator function
    :rtype: Callable[[T], T]
    """

    def validator(model: T) -> T:
        min_value = getattr(model, min_field)
        max_value = getattr(model, max_field)
        if min_value > max_value:
            msg = f"{min_field} ({min_value}) must be less than or equal to {max_field} ({max_value})."
            raise ValueError(
                msg,
            )
        return model

    return validator


class MlpSpec(BaseModel):
    """Model specification for evolvable MLP networks."""

    model_config = ConfigDict(extra="forbid")

    hidden_size: list[int] = Field(
        min_length=1, description="Width of each hidden layer, in order."
    )
    activation: MlpActivation = Field(
        default="ReLU", description="Activation applied after each hidden layer."
    )
    output_activation: MlpActivation | None = Field(
        default=None,
        description="Activation applied to the output. Unset leaves it linear.",
    )
    layer_norm: bool = Field(
        default=True, description="Apply layer normalization after each hidden layer."
    )
    init_layers: bool = Field(
        default=True, description="Apply the library's weight initialization scheme."
    )
    output_vanish: bool = Field(
        default=True,
        description="Scale the output layer's initial weights towards zero, so the network starts near-neutral.",
    )
    min_hidden_layers: int = Field(
        default=1,
        gt=0,
        description="Fewest hidden layers architecture mutation may shrink to.",
    )
    max_hidden_layers: int = Field(
        default=6,
        gt=1,
        description="Most hidden layers architecture mutation may grow to.",
    )
    min_mlp_nodes: int = Field(
        default=8, gt=0, description="Narrowest a hidden layer may be mutated to."
    )
    max_mlp_nodes: int = Field(
        default=256, gt=1, description="Widest a hidden layer may be mutated to."
    )

    arch: Literal["mlp"] = Field(
        default="mlp",
        exclude=True,
        description="Selects a multi-layer perceptron encoder.",
    )

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Dump the model to a dictionary. Remove the ``output_activation`` field if it is ``None``.

        :param kwargs: Keyword arguments to pass to the parent method.
        :type kwargs: Any
        :returns: The model as a dictionary.
        :rtype: dict[str, Any]
        """
        d = super().model_dump(**kwargs)
        if d.get("output_activation") is None:
            d.pop("output_activation", None)
        return d

    @model_validator(mode="after")
    def _check_hidden_layers(self) -> Self:
        min_max_validator("min_hidden_layers", "max_hidden_layers")(self)

        if len(self.hidden_size) < self.min_hidden_layers:
            msg = f"hidden_size must be greater than or equal to min_hidden_layers ({self.min_hidden_layers})."
            raise ValueError(
                msg,
            )

        if len(self.hidden_size) > self.max_hidden_layers:
            msg = f"hidden_size must be less than or equal to max_hidden_layers ({self.max_hidden_layers})."
            raise ValueError(
                msg,
            )

        return self

    @model_validator(mode="after")
    def _check_mlp_nodes(self) -> Self:
        min_max_validator("min_mlp_nodes", "max_mlp_nodes")(self)

        if any(node < self.min_mlp_nodes for node in self.hidden_size):
            msg = f"hidden_size elements must be greater than or equal to min_mlp_nodes ({self.min_mlp_nodes})."
            raise ValueError(
                msg,
            )

        if any(node > self.max_mlp_nodes for node in self.hidden_size):
            msg = f"hidden_size elements must be less than or equal to max_mlp_nodes ({self.max_mlp_nodes})."
            raise ValueError(
                msg,
            )

        return self


class SimbaSpec(BaseModel):
    """Model specification for evolvable SimBA networks.

    Reference: https://arxiv.org/abs/2410.09754
    """

    model_config = ConfigDict(extra="forbid")

    hidden_size: int = Field(gt=0, description="Width of each residual block.")
    num_blocks: int = Field(gt=0, description="Residual blocks stacked in the encoder.")
    output_activation: MlpActivation | None = Field(
        default=None,
        description="Activation applied to the output. Unset leaves it linear.",
    )
    scale_factor: int = Field(
        default=4, gt=0, description="Width multiplier inside each block's bottleneck."
    )
    min_blocks: int = Field(
        default=1,
        gt=0,
        description="Fewest blocks architecture mutation may shrink to.",
    )
    max_blocks: int = Field(
        default=4, gt=1, description="Most blocks architecture mutation may grow to."
    )
    min_mlp_nodes: int = Field(
        default=8, gt=0, description="Narrowest a block may be mutated to."
    )
    max_mlp_nodes: int = Field(
        default=256, gt=1, description="Widest a block may be mutated to."
    )

    arch: Literal["simba"] = Field(
        default="simba",
        exclude=True,
        description="Selects a SimBa residual encoder, which scales better with width than a plain MLP.",
    )

    @model_validator(mode="after")
    def _check_blocks(self) -> Self:
        min_max_validator("min_blocks", "max_blocks")(self)
        min_max_validator("min_blocks", "num_blocks")(self)
        return min_max_validator("num_blocks", "max_blocks")(self)

    @model_validator(mode="after")
    def _check_mlp_nodes(self) -> Self:
        min_max_validator("min_mlp_nodes", "max_mlp_nodes")(self)

        if self.hidden_size < self.min_mlp_nodes:
            msg = f"hidden_size must be greater than or equal to min_mlp_nodes ({self.min_mlp_nodes})."
            raise ValueError(msg)

        if self.hidden_size > self.max_mlp_nodes:
            msg = f"hidden_size must be less than or equal to max_mlp_nodes ({self.max_mlp_nodes})."
            raise ValueError(msg)

        return self


class CnnSpec(BaseModel):
    """Model specification for evolvable convolutional networks."""

    model_config = ConfigDict(extra="forbid")

    channel_size: list[int] = Field(
        min_length=1,
        description="Output channels of each convolutional layer, in order.",
    )
    kernel_size: list[int] = Field(
        min_length=1,
        description="Kernel size of each convolutional layer. Same length as channel_size.",
    )
    stride_size: list[int] = Field(
        min_length=1,
        description="Stride of each convolutional layer. Same length as channel_size.",
    )
    min_hidden_layers: int = Field(
        default=1,
        gt=0,
        description="Fewest convolutional layers architecture mutation may shrink to.",
    )
    max_hidden_layers: int = Field(
        default=6,
        gt=1,
        description="Most convolutional layers architecture mutation may grow to.",
    )
    min_channel_size: int = Field(
        default=8, gt=0, description="Fewest channels a layer may be mutated to."
    )
    max_channel_size: int = Field(
        default=256, gt=1, description="Most channels a layer may be mutated to."
    )
    layer_norm: bool = Field(
        default=False, description="Apply layer normalization after each convolution."
    )
    init_layers: bool = Field(
        default=True, description="Apply the library's weight initialization scheme."
    )
    activation: MlpActivation = Field(
        default="ReLU", description="Activation applied after each convolution."
    )

    arch: Literal["cnn"] = Field(
        default="cnn",
        exclude=True,
        description="Selects a convolutional encoder, for image observations.",
    )

    @model_validator(mode="after")
    def _check_hidden_layers(self) -> Self:
        min_max_validator("min_hidden_layers", "max_hidden_layers")(self)

        if (
            len(self.channel_size) < self.min_hidden_layers
            or len(self.channel_size) > self.max_hidden_layers
        ):
            msg = f"hidden_layers must be between min_hidden_layers ({self.min_hidden_layers}) and max_hidden_layers ({self.max_hidden_layers})."
            raise ValueError(
                msg,
            )

        return self

    @model_validator(mode="after")
    def _check_channel_size(self) -> Self:
        min_max_validator("min_channel_size", "max_channel_size")(self)

        if any(size < self.min_channel_size for size in self.channel_size):
            msg = f"channel_size must be greater than or equal to min_channel_size ({self.min_channel_size})."
            raise ValueError(
                msg,
            )

        if any(size > self.max_channel_size for size in self.channel_size):
            msg = f"channel_size must be less than or equal to max_channel_size ({self.max_channel_size})."
            raise ValueError(
                msg,
            )

        return self

    @model_validator(mode="after")
    def _check_sizes(self) -> Self:
        if len(self.channel_size) != len(self.kernel_size) or len(
            self.channel_size,
        ) != len(self.stride_size):
            msg = f"channel_size, kernel_size, and stride_size must have the same length ({len(self.channel_size)})."
            raise ValueError(
                msg,
            )
        return self


class MultiInputSpec(BaseModel):
    """Model specification for evolvable multi-input networks."""

    model_config = ConfigDict(extra="forbid")

    latent_dim: int = Field(
        default=32,
        gt=0,
        description="Width of the latent each sub-encoder projects into.",
    )
    vector_space_mlp: bool = Field(
        default=False,
        description="Pass vector observations through an MLP before concatenating, rather than using them raw.",
    )
    min_latent_dim: int = Field(
        default=8, gt=0, description="Narrowest the latent may be mutated to."
    )
    max_latent_dim: int = Field(
        default=128, gt=1, description="Widest the latent may be mutated to."
    )
    mlp_config: MlpSpec | None = Field(
        default=None, description="Encoder for the vector parts of the observation."
    )
    cnn_config: CnnSpec | None = Field(
        default=None, description="Encoder for the image parts of the observation."
    )

    arch: Literal["multiinput"] = Field(
        default="multiinput",
        exclude=True,
        description="Selects a multi-input encoder, for dict or tuple observation spaces.",
    )

    @model_validator(mode="after")
    def _check_latent_dim(self) -> Self:
        min_max_validator("min_latent_dim", "max_latent_dim")(self)
        min_max_validator("min_latent_dim", "latent_dim")(self)
        return min_max_validator("latent_dim", "max_latent_dim")(self)


class LstmSpec(BaseModel):
    """Model specification for evolvable LSTM networks."""

    model_config = ConfigDict(extra="forbid")

    hidden_state_size: int = Field(
        gt=0, description="Width of the recurrent hidden state."
    )
    num_layers: int = Field(default=1, gt=0, description="Stacked LSTM layers.")
    min_hidden_state_size: int = Field(
        default=8, gt=0, description="Narrowest the hidden state may be mutated to."
    )
    max_hidden_state_size: int = Field(
        default=256, gt=1, description="Widest the hidden state may be mutated to."
    )
    min_layers: int = Field(
        default=1,
        ge=0,
        description="Fewest LSTM layers architecture mutation may shrink to.",
    )
    max_layers: int = Field(
        default=6,
        ge=1,
        description="Most LSTM layers architecture mutation may grow to.",
    )
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Dropout applied between stacked LSTM layers.",
    )

    arch: Literal["lstm"] = Field(
        default="lstm",
        exclude=True,
        description="Selects a recurrent encoder, for partially-observable environments.",
    )

    @model_validator(mode="after")
    def _check_hidden_state_size(self) -> Self:
        min_max_validator("min_hidden_state_size", "max_hidden_state_size")(self)
        min_max_validator("min_hidden_state_size", "hidden_state_size")(self)
        return min_max_validator("hidden_state_size", "max_hidden_state_size")(self)

    @model_validator(mode="after")
    def _check_layers(self) -> Self:
        min_max_validator("min_layers", "max_layers")(self)
        min_max_validator("min_layers", "num_layers")(self)
        return min_max_validator("num_layers", "max_layers")(self)


EncoderType = MlpSpec | CnnSpec | LstmSpec | MultiInputSpec | SimbaSpec

ENCODER_ARCHS = ("mlp", "cnn", "lstm", "simba", "multiinput")


def encoder_spec_for_arch(arch: str) -> type[BaseModel]:
    """Return the encoder spec class (``MlpSpec``, ``CnnSpec``, ...) for an arch literal.

    :param arch: The encoder architecture literal (e.g. ``"mlp"``, ``"cnn"``).
    :type arch: str
    :returns: The encoder spec class whose ``arch`` field matches.
    :rtype: type[BaseModel]
    :raises ValueError: If *arch* is not a known encoder architecture.
    """
    for member in get_args(EncoderType):
        if member.model_fields["arch"].default == arch:
            return member
    msg = f"Unknown encoder arch: {arch!r}"
    raise ValueError(msg)


def network_arch_is_resolvable(network: object) -> bool:
    """Return True if the manifest network section declares an ``arch``.

    Checks the top level and the nested ``encoder_config``. When False, the
    architecture must be inferred from the observation space at build time.

    :param network: The manifest's network section.
    :type network: object
    :returns: Whether the architecture is declared.
    :rtype: bool
    """
    if not isinstance(network, dict):
        return False
    if network.get("arch"):
        return True
    encoder_config = network.get("encoder_config")
    return isinstance(encoder_config, dict) and bool(encoder_config.get("arch"))


def dump_network_section(network: BaseModel, **kwargs: Any) -> dict[str, Any]:
    """Dump a network section keeping ``arch``, the encoder's discriminator.

    ``arch`` is ``exclude=True`` so it never reaches the network kwargs the
    algorithms are built with. ``encoder_config`` is a union discriminated on
    it, so a dump that has lost it cannot be re-validated.

    :param network: The validated network section.
    :type network: BaseModel
    :returns: The dumped section, with the discriminator put back.
    :rtype: dict[str, Any]
    """
    dumped = network.model_dump(**kwargs)
    if isinstance(network, NetworkSpec):
        encoder_config = dumped.get("encoder_config")
        if isinstance(encoder_config, dict):
            encoder_config["arch"] = network.encoder_config.arch
    return dumped


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

    :param data: The manifest's network section.
    :type data: object
    :returns: The network section with ``arch`` nested under ``encoder_config``.
    :rtype: object
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
        return data

    if encoder_config is None:
        data["encoder_config"] = {"arch": arch}
    else:
        data["encoder_config"] = dict(encoder_config)
        data["encoder_config"].setdefault("arch", arch)

    return data


class NetworkSpec(BaseModel):
    """Base model specification for nested AgileRL evolvable networks."""

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
    encoder_config: EncoderType = Field(
        discriminator="arch",
        description=(
            "Encoder turning observations into a latent. Its `arch` picks the "
            "architecture; unset, the trainer infers one from the observation space."
        ),
    )
    head_config: MlpSpec = Field(
        description="Head mapping the encoder's latent onto the algorithm's outputs."
    )
    random_seed: int | None = Field(
        default=None, description="Seed for weight initialization."
    )
    simba: bool = Field(
        default=False,
        description="Use a SimBa encoder. Set automatically when encoder_config.arch is 'simba'.",
    )

    @model_validator(mode="after")
    def _check_latent_dim(self) -> Self:
        min_max_validator("min_latent_dim", "max_latent_dim")(self)
        min_max_validator("min_latent_dim", "latent_dim")(self)
        return min_max_validator("latent_dim", "max_latent_dim")(self)

    @model_validator(mode="after")
    def _detect_simba(self) -> Self:
        """Auto-detect SimBA from the encoder type so callers don't need to
        set both ``arch: simba`` and ``simba: true`` in the manifest.
        """
        if isinstance(self.encoder_config, SimbaSpec):
            self.simba = True
        return self


class QNetworkSpec(NetworkSpec):
    """Model specification for evolvable Q networks."""


class RainbowQNetworkSpec(NetworkSpec):
    """Model specification for evolvable Rainbow Q networks."""


class ContinuousQNetworkSpec(NetworkSpec):
    """Model specification for evolvable continuous Q networks."""


class DeterministicActorSpec(NetworkSpec):
    """Model specification for evolvable deterministic actor networks."""


class StochasticActorSpec(NetworkSpec):
    """Model specification for evolvable stochastic actor networks."""

    squash_output: bool = Field(
        default=False,
        description="Squash actions through tanh into the action space's bounds.",
    )


class ValueNetworkSpec(NetworkSpec):
    """Model specification for evolvable value networks."""


class LoraConfigDict(BaseModel):
    """Model specification for LLM LoRA configuration."""

    model_config = ConfigDict(extra="forbid")

    lora_r: int = Field(
        default=16,
        ge=1,
        description="LoRA rank. Higher adapts more of the model, at proportional memory cost.",
    )
    lora_alpha: int = Field(
        default=32,
        ge=1,
        description="LoRA scaling. Conventionally set to twice the rank.",
    )
    target_modules: list[str] | str | set[str] = Field(
        default="all-linear",
        description=(
            "Modules to attach adapters to: 'all-linear', or names such as "
            "q_proj, k_proj, v_proj, o_proj."
        ),
    )
    # Packed MoE expert parameter paths (PEFT ``target_parameters``), e.g.
    # ["block_sparse_moe.experts.gate_up_proj", ...]; requires lora_dropout=0.
    target_parameters: list[str] | None = Field(
        default=None,
        description=(
            "Packed MoE expert parameter paths to adapt, e.g. "
            "block_sparse_moe.experts.gate_up_proj. Requires lora_dropout=0 "
            "and a single adapter."
        ),
    )
    task_type: str = Field(
        default="CAUSAL_LM", min_length=1, description="PEFT task type for the adapter."
    )
    lora_dropout: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Dropout on the adapter. Must be 0 when target_parameters is set.",
    )

    @field_serializer("target_modules")
    @classmethod
    def _serialize_target_modules(
        cls, v: list[str] | str | set[str], _info: SerializationInfo
    ) -> list[str] | str:
        return sorted(v) if isinstance(v, set) else v


class CosineLRScheduleConfig(BaseModel):
    """Model specification for cosine LR schedule configuration."""

    model_config = ConfigDict(extra="forbid")

    num_epochs: int = Field(
        default=100, ge=1, description="Epochs the cosine cycle is spread over."
    )
    warmup_proportion: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of the cycle spent warming the learning rate up before it decays.",
    )


class VLLMConfig(BaseModel):
    """Model specification for vLLM configuration."""

    model_config = ConfigDict(extra="forbid")

    tensor_parallel_size: int = Field(
        default=1, ge=1, description="GPUs the model is sharded across for generation."
    )
    gpu_memory_utilization: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Share of GPU memory vLLM reserves for weights and KV cache. When "
            "generation is colocated with training, this has to leave room for "
            "the trainer."
        ),
    )
    max_num_seqs: int = Field(
        default=8,
        ge=1,
        description=(
            "Sequences vLLM decodes concurrently. Set it to at least "
            "group_size so a GRPO group does not queue."
        ),
    )
    max_num_batched_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Tokens vLLM may process in one scheduler pass. Unset lets the "
            "trainer size it; raising it speeds up long prefills but costs memory."
        ),
    )
    enforce_eager: bool | None = Field(
        default=None,
        description="Skip CUDA-graph capture. Starts faster, decodes slower.",
    )
    sleep_mode: bool = Field(
        default=False,
        description=(
            "Release GPU memory between generation calls so the trainer can use "
            "it. Required when generation is colocated with training."
        ),
    )
    sleep_mode_level: Literal[1, 2] = Field(
        default=1,
        description=(
            "1 offloads base weights to host memory and drops the KV cache; 2 "
            "discards the weights, which is only safe when they are pushed back "
            "after every wake."
        ),
    )
    dtype: str | None = Field(
        default=None,
        min_length=1,
        description="Weight dtype for generation, e.g. bfloat16. Unset lets vLLM choose.",
    )
    quantization: str | None = Field(
        default=None,
        min_length=1,
        description="Quantization applied to the generation engine.",
    )
    vllm_model_name_or_path: str | None = Field(
        default=None,
        min_length=1,
        description="Serve generation from a different checkpoint than the trainer fine-tunes.",
    )
    kv_cache_dtype: str | None = Field(
        default=None,
        min_length=1,
        description="KV cache dtype, e.g. fp8. Fits a longer context into the same memory.",
    )
    max_lora_rank: int = Field(
        default=16, ge=1, description="Largest LoRA rank the engine will accept."
    )
    max_loras: int = Field(
        default=1, ge=1, description="LoRA adapters the engine keeps loaded at once."
    )
    strip_multimodal_towers: bool | list[str] = Field(
        default=False,
        description=(
            "Drop the vision or audio towers from the generation copy when the "
            "run is text-only, freeing their memory. May name specific towers."
        ),
    )
    stop_sequences: list[str] | None = Field(
        default=None, description="Strings that end a completion when generated."
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=0.0,
        description="Penalty on tokens that already appeared, at any count.",
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=0.0,
        description="Penalty on tokens in proportion to how often they appeared.",
    )
    kv_cache_memory_bytes: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Pin the KV cache to a fixed size. Needed when several vLLM "
            "processes share a GPU, whose memory profiling would otherwise clash."
        ),
    )
    lora_staging_dir: str | None = Field(
        default=None,
        min_length=1,
        description="Directory the engine reads LoRA adapters from.",
    )


def default_colocated_vllm_config() -> VLLMConfig:
    """The vLLM settings a colocated run gets when the manifest names none.

    Colocated generation shares the trainer's GPU, so it sleeps between calls;
    sleeping is what affords the 0.9 memory utilization.

    :returns: The default colocated engine config.
    :rtype: VLLMConfig
    """
    return VLLMConfig(
        max_num_seqs=16,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        sleep_mode=True,
    )


class FinetuningNetworkSpec(BaseModel):
    """Model specification for LLM finetuning networks."""

    model_config = ConfigDict(extra="forbid")

    pretrained_model_name_or_path: str = Field(
        ...,
        min_length=1,
        description="HuggingFace model id or local path to fine-tune.",
    )
    max_context_length: int = Field(
        ...,
        ge=1,
        description="Context window the run is sized for, prompt plus completion.",
    )
    lora_config: LoraConfigDict | None = Field(
        default=None,
        description="LoRA adapter configuration. Unset fine-tunes without adapters.",
    )
