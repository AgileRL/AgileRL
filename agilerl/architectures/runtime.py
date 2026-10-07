# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Runtime configs keyed by Hugging Face ``model_type``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.distributed.device_mesh import DeviceMesh


class VllmRuntimeConfig(BaseModel):
    """vLLM engine kwargs that vary by Hugging Face ``model_type``.

    Unset fields stay ``None`` so callers can ``setdefault`` into an existing
    engine-args dict without overwriting user values with empties.
    """

    model_config = ConfigDict(extra="forbid")

    mamba_cache_mode: str | None = Field(default=None, min_length=1)
    max_num_batched_tokens: int | None = Field(default=None, ge=1)
    reasoning_parser: str | None = Field(default=None, min_length=1)
    enable_prefix_caching: bool | None = Field(default=None)
    trust_remote_code: bool | None = Field(default=None)
    hf_overrides: dict[str, object] | None = Field(default=None)


class TrainerRuntimeConfig(BaseModel):
    """Trainer ``from_pretrained`` kwargs that vary by Hugging Face ``model_type``.

    Unset fields stay ``None`` so callers can ``setdefault`` into an existing
    model-config dict without overwriting user values with empties.
    """

    model_config = ConfigDict(extra="forbid")

    attn_implementation: str | None = Field(default=None, min_length=1)
    trust_remote_code: bool | None = Field(default=None)


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
    # vLLM tower LoRA needs get_num_mm_encoder_tokens; stock stubs return None.
    enable_tower_connector_lora: bool = False
    tensor_parallel_plan: str | None = Field(default=None, min_length=1)
