# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""vLLM causal LM for a Nemotron omni checkpoint's language tower."""

from __future__ import annotations

from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM
from vllm.model_executor.models.utils import WeightsMapper


class NemotronHOmniLanguageForCausalLM(NemotronHForCausalLM):
    """Nemotron-H language tower from an omni checkpoint."""

    # Expert LoRA adapters are stacked on dim 0.
    is_3d_moe_weight = True
    # Language tensors are prefixed ``language_model.``; vision, the projector,
    # and MTP tensors are not parameters of this module.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={"A_log": "A", "embeddings": "embed_tokens"},
        orig_to_new_prefix={
            "language_model.mtp.": None,
            "language_model.backbone.": "model.",
            "language_model.lm_head.": "lm_head.",
            "vision_model.": None,
            "vision_projector.": None,
            "mlp1.": None,
        },
    )
