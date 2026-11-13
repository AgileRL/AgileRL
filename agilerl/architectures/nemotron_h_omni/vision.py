# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Native ``nemotron_h_omni`` loading without checkpoint code, vision rows, and the vLLM config override.

Transformers 5.19 resolves the Super-VL checkpoint natively except for four
keys: the vision final norm has no conversion entry and the preprocessor
statistics have no ignore. Registration below closes both gaps, plus the
causal-LM mapping the trainer loads through. The vision install lets the
native image features take one ``(h, w, patch_dim)`` patch-grid row per image,
the layout training batches keep, which the native forward otherwise rejects
without ``image_grid_hw``. Both are idempotent; safe to call twice.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.architectures.runtime import PatchRuntimeConfig
from agilerl.architectures.vision_rows import install_vision_rows, packed_patches
from agilerl.utils.patching import library_shim

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedModel
    from transformers.modeling_outputs import BaseModelOutputWithPooling

OMNI_MODEL_TYPE = "nemotron_h_omni"
OMNI_CLASS_NAME = "NemotronH_Omni_Reasoning_V3"

VISION_FINAL_NORM_SOURCE = r"^vision_projector\.vision_final_layernorm\."
VISION_FINAL_NORM_TARGET = r"vision_final_layernorm\."
INPUT_CONDITIONER_IGNORE = r"input_conditioner\."

if HAS_LLM_DEPENDENCIES or TYPE_CHECKING:
    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping,
        register_checkpoint_conversion_mapping,
    )
    from transformers.core_model_loading import WeightRenaming
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
    from transformers.models.nemotron_h_omni.configuration_nemotron_h_omni import (
        NemotronH_Omni_Reasoning_V3_Config,
    )
    from transformers.models.nemotron_h_omni.modeling_nemotron_h_omni import (
        NemotronH_Omni_Reasoning_V3,
        NemotronH_Omni_Reasoning_V3PreTrainedModel,
    )
else:
    get_checkpoint_conversion_mapping: Any = None
    register_checkpoint_conversion_mapping: Any = None
    WeightRenaming: Any = None
    MODEL_FOR_CAUSAL_LM_MAPPING: Any = None
    NemotronH_Omni_Reasoning_V3_Config: Any = None
    NemotronH_Omni_Reasoning_V3: Any = None
    NemotronH_Omni_Reasoning_V3PreTrainedModel: Any = None


def register_nemotron_h_omni_native_loading() -> None:
    """Register the mappings a native Super-VL load needs."""
    if not HAS_LLM_DEPENDENCIES:
        return

    conversions = list(get_checkpoint_conversion_mapping(OMNI_CLASS_NAME) or [])
    if not any(
        VISION_FINAL_NORM_SOURCE in _as_list(transform.source_patterns)
        for transform in conversions
    ):
        conversions.append(
            WeightRenaming(VISION_FINAL_NORM_SOURCE, VISION_FINAL_NORM_TARGET)
        )
        register_checkpoint_conversion_mapping(
            OMNI_CLASS_NAME, conversions, overwrite=True
        )

    ignored = (
        NemotronH_Omni_Reasoning_V3PreTrainedModel._keys_to_ignore_on_load_unexpected
    )
    if INPUT_CONDITIONER_IGNORE not in ignored:
        ignored.append(INPUT_CONDITIONER_IGNORE)

    # The mapping types values as tuples but stores single classes everywhere.
    MODEL_FOR_CAUSAL_LM_MAPPING.register(
        NemotronH_Omni_Reasoning_V3_Config,
        cast("Any", NemotronH_Omni_Reasoning_V3),
        exist_ok=True,
    )


def install_nemotron_h_omni_vision(
    _patch: PatchRuntimeConfig,
    _model: PreTrainedModel | PeftModel | None = None,
    _use_sequence_packing: bool = False,
) -> None:
    """Let ``NemotronH_Omni_Reasoning_V3`` take one patch-grid row per image and run a filler image.

    :param _patch: Catalog patch config.
    :param _model: Already-built model, or None; the class-level patch covers it.
    :param _use_sequence_packing: Unused; the vision patch does not depend on packing.
    """
    install_vision_rows(
        NemotronH_Omni_Reasoning_V3,
        "get_image_features",
        accept_patch_grid_rows,
        vision_filler_features,
    )


def accept_patch_grid_rows(
    get_image_features: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrap ``get_image_features`` so a call without ``image_grid_hw`` takes ``(N, h, w, patch_dim)`` rows.

    :param get_image_features: The native ``NemotronH_Omni_Reasoning_V3.get_image_features``.
    :type get_image_features: Callable[..., Any]
    :return: The wrapped method.
    :rtype: Callable[..., Any]
    """

    def get_image_features_from_rows(
        self: NemotronH_Omni_Reasoning_V3,
        pixel_values: torch.Tensor,
        image_grid_hw: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPooling:
        if image_grid_hw is None:
            pixel_values, image_grid_hw = packed_patches(pixel_values)
        return get_image_features(self, pixel_values, image_grid_hw, **kwargs)

    return get_image_features_from_rows


def vision_filler_features(
    self: NemotronH_Omni_Reasoning_V3, device: torch.device
) -> BaseModelOutputWithPooling:
    """Image features of one zero image on the smallest grid the pixel shuffle takes."""
    side = round(1 / self.config.downsample_ratio)
    return self.get_image_features(
        torch.zeros(1, side, side, self.vision_model.config.patch_dim, device=device)
    )


@library_shim("vllm", "==0.32.*")
def nemotron_h_omni_vllm_hf_override(config: object) -> object:
    """Point a native ``nemotron_h_omni`` config at NemotronHVLLoRA and restore ``vit_hidden_size``.

    :param config: Hugging Face config loaded from the checkpoint.
    :type config: object
    :return: The config vLLM's Nemotron VL model can build from.
    :rtype: object
    """
    fields = vars(config)
    # Serve through NemotronHVLLoRA, which adds tower and connector LoRA.
    fields["architectures"] = ["NemotronH_Super_Omni_Reasoning_V3"]
    # vLLM's Nemotron VL model reads the checkpoint's legacy name; the native
    # config stores it as vision_hidden_size. vLLM also probes with a dummy config.
    if "vision_hidden_size" in fields:
        fields["vit_hidden_size"] = fields["vision_hidden_size"]
    return config


def _as_list(patterns: str | list[str]) -> list[str]:
    """Source patterns as a list (constructors accept both)."""
    return [patterns] if isinstance(patterns, str) else list(patterns)
