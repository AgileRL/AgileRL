# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The manifest's ``network`` section: LLM fine-tuning specs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from typing_extensions import Self

from agilerl.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.models.visibility import AUTO, HIDDEN, HIDDEN_PLATFORM_WRITTEN

TARGET_MODULES_IGNORED = (
    "lora_config.target_modules is ignored: Arena trains LoRA on the model "
    "catalog's modules for the model. Remove target_modules from the manifest."
)

T = TypeVar("T", bound=BaseModel)


def target_modules_warnings(document: dict[str, Any]) -> list[str]:
    """Warn when the document names LoRA modules, which Arena ignores.

    :param document: The raw manifest document.
    :type document: dict[str, Any]
    :returns: Zero or one warning.
    :rtype: list[str]
    """
    names_modules = any(
        isinstance(section, dict)
        and isinstance(section.get("lora_config"), dict)
        and "target_modules" in section["lora_config"]
        for section in (document.get("network"), document.get("algorithm"))
    )
    return [TARGET_MODULES_IGNORED] if names_modules else []


VllmModality = Literal["image", "video", "audio"]


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


class LoraConfigDict(BaseModel):
    """Model specification for LLM LoRA configuration.

    It names no modules: Arena trains LoRA on the catalog's modules for the
    model (:func:`~agilerl.models.model_info.catalog_lora_modules`).
    """

    model_config = ConfigDict(extra="forbid")

    lora_r: int = Field(
        default=16,
        ge=1,
        title="Rank",
        description="LoRA rank. Higher adapts more of the model, at proportional memory cost.",
    )
    lora_alpha: int = Field(
        default=32,
        ge=1,
        title="Alpha",
        description="LoRA scaling. Conventionally set to twice the rank.",
    )
    # Packed MoE expert parameter paths (PEFT ``target_parameters``), e.g.
    # ["block_sparse_moe.experts.gate_up_proj", ...]; requires lora_dropout=0.
    target_parameters: list[str] | None = Field(
        default=None,
        json_schema_extra=HIDDEN_PLATFORM_WRITTEN,
        description=(
            "Packed MoE expert parameter paths to adapt, e.g. "
            "block_sparse_moe.experts.gate_up_proj. Requires lora_dropout=0."
        ),
    )
    task_type: str = Field(
        default="CAUSAL_LM",
        json_schema_extra=HIDDEN,
        min_length=1,
        description="PEFT task type for the adapter.",
    )
    lora_dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        title="Dropout",
        description=(
            "Dropout on the adapter. Must be 0 when target_parameters is set. "
            "Rollout algorithms (GRPO, CISPO, GSPO, LLMPPO, LLMREINFORCE) "
            "sample with vLLM, which applies no dropout."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _drop_target_modules(cls, data: object) -> object:
        """Drop ``target_modules``; saved run specs still carry it, so it is not refused.

        Callers warn about it once, from the raw document
        (:func:`target_modules_warnings`).
        """
        if isinstance(data, dict) and "target_modules" in data:
            return {
                key: value for key, value in data.items() if key != "target_modules"
            }
        return data


class CosineLRScheduleConfig(BaseModel):
    """Model specification for cosine LR schedule configuration."""

    model_config = ConfigDict(extra="forbid")

    num_steps: int = Field(
        ge=1,
        description="Training steps (learn calls) from the start of warmup to the floor.",
    )
    warmup_proportion: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Fraction of num_steps spent ramping the learning rate up to its peak.",
    )
    min_lr_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Floor as a fraction of the peak learning rate, held after num_steps.",
    )


class VLLMConfig(BaseModel):
    """Model specification for vLLM configuration."""

    model_config = ConfigDict(extra="forbid")

    tensor_parallel_size: int | None = Field(
        default=None,
        ge=1,
        description=(
            "GPUs the model is sharded across for generation. Unset until the "
            "allocation's memory estimate chooses a width. An explicit integer "
            "is kept."
        ),
    )
    gpu_memory_utilization: float | None = Field(
        default=None,
        json_schema_extra=AUTO,
        ge=0.0,
        le=1.0,
        description=(
            "Share of GPU memory vLLM reserves for weights and KV cache. When "
            "generation is colocated with training, this has to leave room for "
            "the trainer. Unset is 0.9 on a rollout engine, which has its GPUs "
            "to itself, and 0.3 colocated."
        ),
    )
    max_num_seqs: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Sequences vLLM decodes concurrently. Set it to at least "
            "group_size so a GRPO group does not queue. Unset gives colocated "
            "generation 16 and each rollout engine one sequence per episode it "
            "runs at once (rollout_batch_size / rollout_engines_per_agent, "
            "rounded up, x group_size)."
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
    dtype: str | None = Field(
        default=None,
        min_length=1,
        description="Weight dtype for generation, e.g. bfloat16. Unset lets vLLM choose.",
    )
    quantization: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Quantization applied to the generation engine, e.g. fp8. Unset or "
            "none serves unquantized weights."
        ),
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
    stop_sequences: list[str] | None = Field(
        default=None, description="Strings that end a completion when generated."
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Penalty on tokens that already appeared, at any count.",
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Penalty on tokens in proportion to how often they appeared.",
    )
    limit_mm_per_prompt: dict[str, Annotated[int, Field(ge=0)]] | None = Field(
        default=None,
        description=(
            "Most items of each modality one prompt may carry, e.g. "
            "{image: 4, video: 0}. Unset keeps vLLM's limit."
        ),
    )
    lora_staging_dir: str | None = Field(
        default=None,
        min_length=1,
        description="Directory the engine reads LoRA adapters from.",
    )


def default_rollout_vllm_config() -> VLLMConfig:
    """The vLLM settings a rollout run gets when the manifest names none.

    Colocated generation shares the trainer's GPU, so it sleeps between calls;
    sleeping is what affords the 0.9 memory utilization. ``max_num_seqs`` is
    left unset, so each engine resolves it from the episodes it runs.

    :returns: The default engine config.
    :rtype: VLLMConfig
    """
    return VLLMConfig(
        gpu_memory_utilization=0.9,
    )


def resolve_lora_defaults(lora: LoraConfigDict, hub_id: str) -> None:
    """Fill a LoRA config's catalog defaults in place.

    Unset ``target_parameters`` on a packed-MoE catalog model adapts every
    expert path and sets ``lora_dropout`` to 0; ``[]`` adapts none. Untracked
    Hub ids are left as they are.

    :param lora: The config to resolve.
    :type lora: LoraConfigDict
    :param hub_id: The model the config trains.
    :type hub_id: str
    """
    entry = SUPPORTED_MODEL_INFO.get(hub_id)
    if entry is None or not entry.parameters:
        return
    if lora.target_parameters is None:
        lora.target_parameters = sorted(entry.parameters)
        # PEFT parameter-level LoRA cannot apply dropout.
        lora.lora_dropout = 0.0


def check_lora_config_supported(hub_id: str, lora: LoraConfigDict) -> None:
    """Fill a LoRA config's catalog defaults, then reject a rank or expert targets the Hub id cannot train with.

    :param hub_id: The model the config trains.
    :type hub_id: str
    :param lora: The config to check, edited in place.
    :type lora: LoraConfigDict
    """
    entry = SUPPORTED_MODEL_INFO.get(hub_id)
    if entry is None:
        return
    resolve_lora_defaults(lora, hub_id)
    # Ranks below the cap pass; vLLM rounds max_lora_rank up to a value it accepts.
    if entry.lora_ranks is not None and lora.lora_r > max(entry.lora_ranks):
        msg = (
            f"LoRA rank {lora.lora_r} is above the largest rank vLLM serves "
            f"for {hub_id} ({max(entry.lora_ranks)})"
        )
        raise ValueError(msg)
    if unknown := sorted(set(lora.target_parameters or ()) - entry.parameters):
        valid = ", ".join(sorted(entry.parameters)) or "none"
        msg = (
            f"Unknown LoRA target_parameters {unknown} for {hub_id}; "
            f"valid parameters: {valid}"
        )
        raise ValueError(msg)
    if (lora.target_parameters or []) and lora.lora_dropout != 0:
        msg = (
            "lora_config.lora_dropout must be 0 when target_parameters "
            "is set: PEFT parameter-level LoRA cannot factor dropout out."
        )
        raise ValueError(msg)


class FinetuningNetworkSpec(BaseModel):
    """Model specification for LLM finetuning networks."""

    model_config = ConfigDict(extra="forbid")

    pretrained_model_name_or_path: str = Field(
        ...,
        min_length=1,
        description="HuggingFace model id or local path to fine-tune.",
    )
    weights_path: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Local or s3:// directory to load the weights and tokenizer from. "
            "pretrained_model_name_or_path then names the catalog model with "
            "the same architecture, which sizes and configures the run."
        ),
    )
    max_context_length: int = Field(
        ...,
        ge=1,
        description="Context window the run is sized for, prompt plus completion.",
    )
    lora_config: LoraConfigDict = Field(
        default_factory=LoraConfigDict,
        description=(
            "LoRA adapter configuration. Unset or null trains the model's "
            "catalog default adapter; Arena has no full finetune."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _null_lora_is_the_default(cls, data: object) -> object:
        """Treat ``lora_config: null`` as unset, so it gets the default adapter."""
        if (
            isinstance(data, dict)
            and "lora_config" in data
            and data["lora_config"] is None
        ):
            return {key: value for key, value in data.items() if key != "lora_config"}
        return data

    @model_validator(mode="after")
    def _check_lora_config_supported(self) -> Self:
        """Reject a LoRA rank or targets a tracked Hub id cannot train with."""
        check_lora_config_supported(
            self.pretrained_model_name_or_path, self.lora_config
        )
        return self
