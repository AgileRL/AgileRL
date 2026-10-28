# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Runtime configs keyed by Hugging Face ``model_type``."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from agilerl.arena.models.networks import VllmModality


class VllmRuntimeConfig(BaseModel):
    """vLLM engine kwargs that vary by Hugging Face ``model_type``.

    Unset fields stay ``None`` so callers can ``setdefault`` into an existing
    engine-args dict without overwriting user values with empties.
    """

    model_config = ConfigDict(extra="forbid")

    mamba_cache_mode: str | None = Field(default=None, min_length=1)
    max_num_batched_tokens: int | None = Field(default=None, ge=1)
    enable_prefix_caching: bool | None = Field(default=None)
    trust_remote_code: bool | None = Field(default=None)
    hf_overrides: dict[str, object] | Callable[[object], object] | None = Field(
        default=None
    )
    # vLLM >=0.32 keeps Mamba prefix-cache states only at each prompt's replay
    # boundary, so a later turn recomputes the previous reply.
    dense_mamba_prefix_cache: bool | None = Field(default=None)

    def engine_kwargs(self) -> dict[str, Any]:
        """Return the engine kwargs this family sets.

        :return: Set fields as vLLM engine arguments.
        :rtype: dict[str, Any]
        """
        kwargs = self.model_dump(
            exclude_none=True, exclude={"dense_mamba_prefix_cache"}
        )
        if self.dense_mamba_prefix_cache:
            kwargs["prefix_cache_retention_interval"] = None
        return kwargs


TrainerModelClass = type[AutoModelForCausalLM] | type[AutoModelForImageTextToText]


class TrainerRuntimeConfig(BaseModel):
    """Trainer load settings that vary by Hugging Face ``model_type``.

    ``model_class`` is the auto class every trainer load builds. The other
    fields are ``from_pretrained`` kwargs; unset ones stay ``None`` so callers
    can ``setdefault`` into an existing model-config dict without overwriting
    user values with empties.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # VL checkpoints build their full VL class so towers load, frozen.
    model_class: TrainerModelClass = AutoModelForCausalLM
    attn_implementation: str | None = Field(default=None, min_length=1)
    trust_remote_code: bool | None = Field(default=None)

    def load_kwargs(self) -> dict[str, object]:
        """Return the ``from_pretrained`` kwargs this family sets, without the attention backend.

        :return: ``trust_remote_code`` when set.
        :rtype: dict[str, object]
        """
        return self.model_dump(
            exclude_none=True, exclude={"model_class", "attn_implementation"}
        )


class MambaPatchConfig(BaseModel):
    """Mamba2 mixer patches that vary by Hugging Face ``model_type``.

    ``mixer`` and ``block`` are dotted paths resolved when the patch runs; the
    transformers classes may be absent at import time. ``block`` is the decoder
    block that calls ``mixer``; when set, packed rows reset the mixer's scan and
    conv state at every document boundary.
    """

    model_config = ConfigDict(extra="forbid")

    mixer: str = Field(min_length=1)
    block: str | None = Field(default=None, min_length=1)
    fused_path: bool = True
    stream_ordering: bool = True


class PatchRuntimeConfig(BaseModel):
    """Architecture patches that vary by Hugging Face ``model_type``.

    Unset families leave ``install`` as ``None`` so dispatch skips the installer.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    install: Callable[..., object] | None = None
    mamba: MambaPatchConfig | None = Field(default=None)


@dataclass(frozen=True)
class TensorParallelPlan:
    """Tensor-parallel hooks for the module classes one family defines.

    ``shard`` runs before FSDP wraps the model and returns how many modules it
    sharded. ``restore`` runs once weights are loaded into the shards. Both
    leave modules of other families untouched.
    """

    shard: Callable[[nn.Module, DeviceMesh], int]
    restore: Callable[[nn.Module, DeviceMesh], None]


class ModelRuntimeConfig(BaseModel):
    """Per-``model_type`` runtime settings for vLLM, trainer, patches, and TP.

    ``tensor_parallel_plan`` is a ``module:attribute`` path to a
    :class:`TensorParallelPlan`, resolved when sharding runs: plan modules
    import :mod:`agilerl.distributed`, which imports this package.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    vllm: VllmRuntimeConfig = Field(default_factory=VllmRuntimeConfig)
    trainer: TrainerRuntimeConfig = Field(default_factory=TrainerRuntimeConfig)
    patch: PatchRuntimeConfig = Field(default_factory=PatchRuntimeConfig)
    # PEFT target names of the trainer's tower and connector linears that
    # lora_config.tower_target_modules may name. Empty for families without
    # vLLM tower LoRA (it needs get_mm_lora_token_counts).
    tower_connector_lora_targets: frozenset[str] = frozenset()
    # Trainer tower and connector module names mapped to the modalities vLLM
    # builds them for; vLLM skips a module once all its modalities are stripped.
    tower_modalities: Mapping[str, frozenset[VllmModality]] = Field(
        default_factory=dict
    )
    tensor_parallel_plan: str | None = Field(default=None, min_length=1)
    # A name in agilerl.llm_envs.tool_parsers.TOOL_PARSERS; None detects from the vocab.
    tool_call_parser: str | None = Field(default=None, min_length=1)
    # vLLM reasoning parser of a thinking family. Only the Ray rollout engine
    # enables it, for a thinking budget or answer grammar; otherwise its
    # reasoning_config makes every GPU step wait on the last step's tokens.
    reasoning_parser: str | None = Field(default=None, min_length=1)
