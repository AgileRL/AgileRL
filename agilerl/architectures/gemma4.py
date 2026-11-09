# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 image features from one ``(h, w, patch_dim)`` patch-grid row per image.

The native ``get_image_features`` reads patches padded to the processor's
``max_patches`` with an ``image_position_ids`` grid. Training batches keep
unpadded rows instead; the vision tower pools ``h * w`` patches to
``h * w / k**2`` soft tokens either way, since padding is masked.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch

from agilerl.architectures.runtime import PatchRuntimeConfig
from agilerl.utils.patching import class_is_patched, try_import

if TYPE_CHECKING:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4Model

__all__ = [
    "ACCEPTS_VISION_ROWS",
    "install_gemma4_vision_patches",
    "patch_positions",
]

ACCEPTS_VISION_ROWS = "agilerl_accepts_vision_rows"

GEMMA4_MODELING = "transformers.models.gemma4.modeling_gemma4"


def patch_positions(pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten ``(N, h, w, patch_dim)`` rows into row-major patches and their ``(x, y)`` positions.

    :param pixel_values: One patch grid per image, all of one shape.
    :return: ``(N, h * w, patch_dim)`` patches and ``(N, h * w, 2)`` positions.
    """
    images, grid_height, grid_width, patch_dim = pixel_values.shape
    ys, xs = torch.meshgrid(
        torch.arange(grid_height, device=pixel_values.device),
        torch.arange(grid_width, device=pixel_values.device),
        indexing="ij",
    )
    positions = torch.stack([xs, ys], dim=-1).reshape(1, -1, 2)
    return (
        pixel_values.reshape(images, grid_height * grid_width, patch_dim),
        positions.expand(images, -1, -1),
    )


def _accept_patch_grid_rows(
    get_image_features: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrap ``get_image_features`` so a call without positions takes ``(N, h, w, patch_dim)`` rows."""

    @functools.wraps(get_image_features)
    def get_image_features_from_rows(
        self: Gemma4Model,
        pixel_values: torch.Tensor,
        image_position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> object:
        if image_position_ids is None and pixel_values.dim() == 4:
            pixel_values, image_position_ids = patch_positions(pixel_values)
        return get_image_features(self, pixel_values, image_position_ids, **kwargs)

    return get_image_features_from_rows


def vision_filler_features(self: Gemma4Model, device: torch.device) -> object:
    """Image features of one zero image on the smallest grid the vision pooler takes.

    :param device: Device the zero image is built on.
    :return: ``get_image_features`` output for that image.
    """
    # Gemma4Config builds this sub-config on init; the union types its input.
    vision = cast("Gemma4VisionConfig", self.config.vision_config)
    side = vision.pooling_kernel_size
    return self.get_image_features(
        torch.zeros(1, side, side, 3 * vision.patch_size**2, device=device)
    )


def install_gemma4_vision_patches(
    _patch: PatchRuntimeConfig,
    _model: object | None = None,
) -> None:
    """Let ``Gemma4Model.get_image_features`` take one patch-grid row per image.

    :param _patch: Catalog patch config.
    :param _model: Already-built model, or None; the class-level patch covers it.
    """
    modeling = try_import(GEMMA4_MODELING)
    if modeling is None:
        message = f"[gemma4-vision] {GEMMA4_MODELING} is not importable"
        raise RuntimeError(message)
    model_cls = modeling.Gemma4Model
    if class_is_patched(model_cls, ACCEPTS_VISION_ROWS):
        return
    model_cls.get_image_features = _accept_patch_grid_rows(
        model_cls.__dict__["get_image_features"]
    )
    model_cls.vision_filler_features = vision_filler_features
    setattr(model_cls, ACCEPTS_VISION_ROWS, True)
