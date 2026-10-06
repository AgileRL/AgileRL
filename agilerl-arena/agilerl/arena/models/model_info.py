# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Static info per supported pretrained model.

LoRA targets, sizing, and allowed ranks keyed by Hub id, for manifest
validation and the memory estimator. Arena runs without model weights, so
each id bundles one file under ``supported/``: ``config`` (its raw
``config.json``) and ``inspected`` (the parameter count and per-target LoRA
dims from a meta-device build of that config). Everything else is read from
those.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from functools import cached_property
from importlib.resources import files
from typing import Any, Literal

# Python 3.10: Traversable is importlib.abc, not importlib.resources.abc.
if sys.version_info >= (3, 11):
    from importlib.resources.abc import Traversable
else:
    from importlib.abc import Traversable  # pragma: no cover

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer

ModelArchitecture = Literal["dense", "hybrid", "moe", "hybrid_moe"]
ARCH_DENSE: ModelArchitecture = "dense"
ARCH_HYBRID: ModelArchitecture = "hybrid"
ARCH_MOE: ModelArchitecture = "moe"
ARCH_HYBRID_MOE: ModelArchitecture = "hybrid_moe"

ModelStatus = Literal["live", "deprecated", "preview"]
STATUS_LIVE: ModelStatus = "live"
STATUS_DEPRECATED: ModelStatus = "deprecated"
STATUS_PREVIEW: ModelStatus = "preview"

# vLLM ``LoRAConfig.max_lora_rank`` values.
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)
# vLLM fused MoE LoRA rejects ranks above this.
FUSED_MOE_LORA_MAX_RANK = 128


def supported_path(hub_id: str) -> Traversable:
    """Bundled ``config`` + ``inspected`` JSON for ``hub_id``."""
    return files("agilerl.arena.models.supported").joinpath(f"{hub_id}.json")


def _bundled(hub_id: str) -> dict[str, Any]:
    """Parsed ``supported/`` file for ``hub_id``."""
    return json.loads(supported_path(hub_id).read_text(encoding="utf-8"))


class ModelInfo(BaseModel):
    """Static info for one supported Hub id."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hub_id: str = Field(min_length=1, description="Hugging Face org/name.")
    architecture: ModelArchitecture = Field(
        default=ARCH_DENSE,
        description="Layer layout: dense, hybrid (Mamba), MoE, or hybrid MoE.",
    )
    status: ModelStatus = Field(
        default=STATUS_LIVE,
        description=(
            "Release status: live, deprecated (submits with a warning), or "
            "preview (submission rejected)."
        ),
    )

    @cached_property
    def config(self) -> dict[str, Any]:
        """Raw ``config.json`` this id trains from."""
        return _bundled(self.hub_id)["config"]

    @cached_property
    def inspected(self) -> dict[str, Any]:
        """``num_params`` and per-target ``lora_info`` from a meta-device build."""
        return _bundled(self.hub_id)["inspected"]

    @property
    def text_config(self) -> dict[str, Any]:
        """Language-model section of the config; multimodal ids nest it."""
        return (
            self.config.get("text_config")
            or self.config.get("llm_config")
            or self.config
        )

    @computed_field
    @property
    def modules(self) -> frozenset[str]:
        """Valid ``target_modules`` names."""
        return self._target_keys("module")

    @computed_field
    @property
    def parameters(self) -> frozenset[str]:
        """Valid packed-expert ``target_parameters`` paths."""
        return self._target_keys("parameter")

    @computed_field
    @property
    def num_params(self) -> int | None:
        """Total parameters; None when the config cannot be built offline."""
        return self.inspected["num_params"]

    @computed_field
    @property
    def lora_ranks(self) -> tuple[int, ...] | None:
        """vLLM ``max_lora_rank`` values this model supports; None without dims.

        Capped by the smallest targetable dim, and by the fused MoE limit when
        the model has packed experts.
        """
        entries = [
            entry for group in self.inspected["lora_info"].values() for entry in group
        ]
        dims = [
            entry[field]
            for entry in entries
            for field in ("in_features", "out_features")
            if field in entry
        ]
        if not dims:
            return None
        cap = min(dims)
        if self.parameters:
            cap = min(cap, FUSED_MOE_LORA_MAX_RANK)
        return tuple(rank for rank in VLLM_LORA_RANKS if rank <= cap)

    @computed_field
    @property
    def lora_info(self) -> dict[str, Any]:
        """Per-target LoRA dims plus the ``_gram_estimate`` sidecar."""
        return {
            **self.inspected["lora_info"],
            "_gram_estimate": {
                "hidden_dim": self.hidden_dim,
                "vocab_size": self.vocab_size,
                "num_hidden_layers": self.num_hidden_layers,
            },
        }

    @computed_field
    @property
    def model_type(self) -> str:
        """Hugging Face ``model_type``."""
        return self.config["model_type"]

    @computed_field
    @property
    def max_context_length(self) -> int:
        """Context window the checkpoint supports."""
        return self.text_config["max_position_embeddings"]

    @computed_field
    @property
    def hidden_dim(self) -> int:
        """GRAM estimator hidden size."""
        return self.text_config["hidden_size"]

    @computed_field
    @property
    def vocab_size(self) -> int:
        """GRAM estimator vocab size."""
        return self.text_config["vocab_size"]

    @computed_field
    @property
    def num_hidden_layers(self) -> int:
        """GRAM estimator layer count."""
        text_config = self.text_config
        # Nemotron-H omni configs list blocks instead of counting them.
        if "num_hidden_layers" in text_config:
            return text_config["num_hidden_layers"]
        return len(text_config["layers_block_type"])

    @field_serializer("modules", "parameters")
    def _serialize_sorted(self, value: frozenset[str]) -> list[str]:
        return sorted(value)

    def _target_keys(self, kind: str) -> frozenset[str]:
        return frozenset(
            key
            for key, group in self.inspected["lora_info"].items()
            if group[0]["kind"] == kind
        )


SUPPORTED_MODEL_INFO: Mapping[str, ModelInfo] = {
    info.hub_id: info
    for info in (
        ModelInfo(hub_id="Qwen/Qwen2.5-0.5B-Instruct", status=STATUS_DEPRECATED),
        ModelInfo(hub_id="Qwen/Qwen3-1.7B"),
        ModelInfo(hub_id="Qwen/Qwen3-4B"),
        ModelInfo(hub_id="Qwen/Qwen3.8-27B", architecture=ARCH_HYBRID),
        ModelInfo(hub_id="ibm-granite/granite-4.0-micro", status=STATUS_DEPRECATED),
        ModelInfo(
            hub_id="ibm-granite/granite-4.0-micro-base", status=STATUS_DEPRECATED
        ),
        ModelInfo(
            hub_id="ibm-granite/granite-4.0-h-tiny",
            architecture=ARCH_HYBRID,
            status=STATUS_DEPRECATED,
        ),
        ModelInfo(
            hub_id="ibm-granite/granite-3.1-3b-a800m-instruct",
            architecture=ARCH_MOE,
            status=STATUS_DEPRECATED,
        ),
        ModelInfo(hub_id="google/gemma-4-E4B-it"),
        ModelInfo(
            hub_id="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", architecture=ARCH_HYBRID
        ),
        ModelInfo(
            hub_id="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
            architecture=ARCH_HYBRID_MOE,
        ),
        ModelInfo(
            hub_id="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
            architecture=ARCH_HYBRID_MOE,
            status=STATUS_PREVIEW,
        ),
        ModelInfo(hub_id="openai/gpt-oss-20b", architecture=ARCH_MOE),
    )
}
