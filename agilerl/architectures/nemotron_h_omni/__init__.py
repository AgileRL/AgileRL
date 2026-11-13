# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-H Omni vision-language, loaded natively; Nemotron-H settings plus the towers."""

from __future__ import annotations

from agilerl.architectures.nemotron_h import FAMILY as NEMOTRON_H
from agilerl.architectures.nemotron_h_omni.vision import (
    install_nemotron_h_omni_vision,
    nemotron_h_omni_vllm_hf_override,
    register_nemotron_h_omni_native_loading,
)
from agilerl.architectures.runtime import (
    ModelTypes,
    TrainerRuntimeConfig,
    VisionSupport,
)

FAMILY = NEMOTRON_H.model_copy(
    update={
        "name": "nemotron_h_omni",
        "model_types": ModelTypes(multimodal=("nemotron_h_omni",)),
        "vllm": NEMOTRON_H.vllm.model_copy(
            update={"hf_overrides": nemotron_h_omni_vllm_hf_override}
        ),
        # Native nemotron_h_omni; remote code stopped loading on 5.19.
        "trainer": TrainerRuntimeConfig(trust_remote_code=False),
        "register_loading": register_nemotron_h_omni_native_loading,
        "vision": VisionSupport(
            install=install_nemotron_h_omni_vision,
            tower_connector_lora_targets=frozenset(
                {
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "fc1",
                    "fc2",
                    "patch_projection",
                    "video_patch_projection",
                    "linear_1",
                    "linear_2",
                }
            ),
        ),
    }
)
