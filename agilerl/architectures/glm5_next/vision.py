# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 image features from one ``(h, w, patch_dim)`` patch-grid row per image.

The native ``Glm5NextModel.get_image_features`` reads flat patches with an
``image_grid_thw``; training batches keep one row per image instead. Text
positions stay 1-D: GLM-5.3 attention has no rope part (``qk_rope_head_dim``
is 0), so images need no M-RoPE positions.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.architectures.runtime import PatchRuntimeConfig
from agilerl.architectures.vision_rows import install_vision_rows, packed_patches

if HAS_LLM_DEPENDENCIES or TYPE_CHECKING:
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextModel
else:
    Glm5NextModel: Any = None

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedModel

__all__ = [
    "accept_patch_grid_rows",
    "install_glm5_next_vision",
    "vision_filler_features",
]


def accept_patch_grid_rows(
    get_image_features: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrap ``get_image_features`` so a call without ``image_grid_thw`` takes ``(N, h, w, patch_dim)`` rows.

    :param get_image_features: The native ``Glm5NextModel.get_image_features``.
    :return: The wrapped method.
    """

    @functools.wraps(get_image_features)
    def get_image_features_from_rows(
        self: Glm5NextModel,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> object:
        if image_grid_thw is None:
            pixel_values, image_grid_hw = packed_patches(pixel_values)
            image_grid_thw = F.pad(image_grid_hw, (1, 0), value=1)
        return get_image_features(self, pixel_values, image_grid_thw, **kwargs)

    return get_image_features_from_rows


def vision_filler_features(self: Glm5NextModel, device: torch.device) -> object:
    """Image features of one zero image on the smallest grid the vision downsample takes.

    :param device: Device the zero image is built on.
    :return: ``get_image_features`` output for that image.
    """
    vision = self.visual.config
    side = vision.spatial_merge_size
    patch_dim = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    return self.get_image_features(torch.zeros(1, side, side, patch_dim, device=device))


def install_glm5_next_vision(
    _patch: PatchRuntimeConfig,
    _model: PreTrainedModel | PeftModel | None = None,
    _use_sequence_packing: bool = False,
) -> None:
    """Let ``Glm5NextModel`` take one patch-grid row per image and run a filler image.

    :param _patch: Catalog patch config.
    :param _model: Already-built model, or None; the class-level patch covers it.
    :param _use_sequence_packing: Unused; the vision patch does not depend on packing.
    """
    install_vision_rows(
        Glm5NextModel,
        "get_image_features",
        accept_patch_grid_rows,
        vision_filler_features,
    )
