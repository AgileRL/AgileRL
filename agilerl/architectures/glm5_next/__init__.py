# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 patches: DSA top-k skip, Liger kernels, and tensor parallel."""

from __future__ import annotations

from agilerl.architectures.glm5_next.patches import (
    CAUSAL_SDPA_WHEN_TOPK_COVERS,
    causal_sdpa_attention_forward,
    install_glm5_next_patches,
    topk_covers_causal_prefix,
    topk_skipping_model_forward,
)

__all__ = [
    "CAUSAL_SDPA_WHEN_TOPK_COVERS",
    "causal_sdpa_attention_forward",
    "install_glm5_next_patches",
    "topk_covers_causal_prefix",
    "topk_skipping_model_forward",
]
