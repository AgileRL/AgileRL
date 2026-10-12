# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import pytest

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.utils.llm_packing import RESETS_AT_DOCUMENT_BOUNDARY

if HAS_LLM_DEPENDENCIES:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5GatedDeltaNet,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer,
        Qwen3_5MoeGatedDeltaNet,
    )

    GDN_CLASSES = (
        (Qwen3_5MoeGatedDeltaNet, Qwen3_5MoeDecoderLayer),
        (Qwen3_5GatedDeltaNet, Qwen3_5DecoderLayer),
    )
else:
    GDN_CLASSES = ()


@pytest.fixture
def pristine_qwen_gdn_classes():
    """Restore Qwen GDN mixer and decoder forwards after a class-level patch."""
    saved = [
        (mixer, mixer.__dict__["forward"], block, block.__dict__["forward"])
        for mixer, block in GDN_CLASSES
    ]
    yield
    for mixer, mixer_forward, block, block_forward in saved:
        mixer.forward = mixer_forward
        block.forward = block_forward
        if RESETS_AT_DOCUMENT_BOUNDARY in vars(mixer):
            delattr(mixer, RESETS_AT_DOCUMENT_BOUNDARY)
