# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Runtime configs keyed by Hugging Face ``model_type``."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field


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

    ``mixer`` is a dotted path resolved when the patch runs; the transformers
    class may be absent at import time.
    """

    model_config = ConfigDict(extra="forbid")

    mixer: str = Field(min_length=1)
    fused_path: bool = True
    stream_ordering: bool = True


class PatchRuntimeConfig(BaseModel):
    """Architecture patches that vary by Hugging Face ``model_type``.

    Unset families leave ``install`` as ``None`` so dispatch skips the installer.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    install: Callable[..., object] | None = None
    mamba: MambaPatchConfig | None = Field(default=None)


class LanguageTowerRuntimeConfig(BaseModel):
    """vLLM language-tower mapping that varies by Hugging Face ``model_type``.

    Generic VL serving peels ``text_config`` / ``llm_config``. A family that
    needs a custom vLLM class sets ``hf_overrides`` and ``model_class_overrides``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    hf_overrides: Callable[[object], object] | None = None
    model_class_overrides: dict[str, str] | None = None


class ModelRuntimeConfig(BaseModel):
    """Per-``model_type`` runtime settings for vLLM, trainer, and patches."""

    model_config = ConfigDict(extra="forbid")

    vllm: VllmRuntimeConfig = Field(default_factory=VllmRuntimeConfig)
    trainer: TrainerRuntimeConfig = Field(default_factory=TrainerRuntimeConfig)
    patch: PatchRuntimeConfig = Field(default_factory=PatchRuntimeConfig)
    language_tower: LanguageTowerRuntimeConfig = Field(
        default_factory=LanguageTowerRuntimeConfig
    )
