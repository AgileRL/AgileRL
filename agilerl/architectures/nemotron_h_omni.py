# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Native ``nemotron_h_omni`` loading without checkpoint code.

Transformers 5.19 resolves the Super-VL checkpoint natively except for four
keys: the vision final norm has no conversion entry and the preprocessor
statistics have no ignore. Registration below closes both gaps, plus the
causal-LM mapping the trainer loads through. It also lets the native image
features take one ``(h, w, patch_dim)`` patch-grid row per image, the layout
training batches keep, which the native forward otherwise rejects without
``image_grid_hw``. Idempotent; safe to call twice.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch

from agilerl import HAS_LLM_DEPENDENCIES

if TYPE_CHECKING:
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

    native_get_image_features = NemotronH_Omni_Reasoning_V3.get_image_features
    if native_get_image_features.__module__ != __name__:
        NemotronH_Omni_Reasoning_V3.get_image_features = accept_patch_grid_rows(
            native_get_image_features
        )


def packed_patches(pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unfold ``(N, h, w, patch_dim)`` patch-grid rows into the packed patches and grids the native tower reads.

    :param pixel_values: One patch grid per image, all of one shape.
    :type pixel_values: torch.Tensor
    :return: ``(N * h * w, patch_dim)`` patches and the ``(N, 2)`` patch grid of each image.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    images, grid_height, grid_width, patch_dim = pixel_values.shape
    image_grid_hw = torch.tensor(
        [[grid_height, grid_width]] * images, device=pixel_values.device
    )
    return pixel_values.reshape(-1, patch_dim), image_grid_hw


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


def _as_list(patterns: str | list[str]) -> list[str]:
    """Source patterns as a list (constructors accept both)."""
    return [patterns] if isinstance(patterns, str) else list(patterns)
