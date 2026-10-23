# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""vLLM causal LMs for a Qwen3.5 VL checkpoint's language tower."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor
from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForCausalLM,
)
from vllm.model_executor.models.utils import WeightsMapper

# VL checkpoints store language weights under model.language_model; the
# language-only module expects model.
QWEN3_5_VL_LANGUAGE_MAPPER = WeightsMapper(
    orig_to_new_prefix={
        "model.language_model.": "model.",
        "model.visual.": None,
        "mtp.": None,
    }
)


class Qwen3_5LanguageHybrid(IsHybrid):
    """Linear-attention KV-cache hooks vLLM keeps on the VL class."""

    get_mamba_state_dtype_from_config = (
        Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config
    )
    get_mamba_state_shape_from_config = (
        Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config
    )
    get_mamba_state_copy_func = (
        Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func
    )


class Qwen3_5LanguageMRoPE(SupportsMRoPE):
    """M-RoPE positions for text-only input; the vision towers are stripped."""

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list,
    ) -> tuple[Tensor, int]:
        if mm_features:
            msg = "Qwen3.5 language tower serves text-only input."
            raise ValueError(msg)

        # Text tokens advance every position axis together.
        positions = torch.arange(len(input_tokens)).unsqueeze(0).repeat(3, 1)
        return positions, 0


class Qwen3_5LanguageForCausalLM(
    Qwen3_5ForCausalLM, Qwen3_5LanguageHybrid, Qwen3_5LanguageMRoPE
):
    """Qwen3.5 dense language tower from a VL checkpoint."""

    hf_to_vllm_mapper = QWEN3_5_VL_LANGUAGE_MAPPER

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        return super().load_weights(self.hf_to_vllm_mapper.apply(weights))


class Qwen3_5MoeLanguageForCausalLM(
    Qwen3_5MoeForCausalLM, Qwen3_5LanguageHybrid, Qwen3_5LanguageMRoPE
):
    """Qwen3.5 MoE language tower from a VL checkpoint."""

    # Expert LoRA adapters are stacked on dim 0.
    is_3d_moe_weight = True
    hf_to_vllm_mapper = QWEN3_5_VL_LANGUAGE_MAPPER

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        return super().load_weights(self.hf_to_vllm_mapper.apply(weights))
