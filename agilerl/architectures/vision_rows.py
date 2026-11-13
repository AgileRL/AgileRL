# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers that let a vision-language class take one ``(h, w, patch_dim)`` patch-grid row per image."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from agilerl.utils.patching import class_is_patched

__all__ = ["ACCEPTS_VISION_ROWS", "install_vision_rows", "packed_patches"]

ACCEPTS_VISION_ROWS = "agilerl_accepts_vision_rows"


def packed_patches(pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unfold ``(N, h, w, patch_dim)`` patch-grid rows into the packed patches and grids native towers read.

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


def install_vision_rows(
    model_cls: type,
    method: str,
    wrap: Callable[[Callable[..., Any]], Callable[..., Any]],
    filler: Callable[..., Any],
) -> None:
    """Wrap ``model_cls.<method>`` to take vision rows and set ``vision_filler_features``, once per class.

    :param model_cls: Vision-language class to patch.
    :type model_cls: type
    :param method: Name of the method that receives ``pixel_values``.
    :type method: str
    :param wrap: Returns the row-accepting method given the native one.
    :type wrap: Callable[[Callable[..., Any]], Callable[..., Any]]
    :param filler: ``vision_filler_features(self, device)`` for vision lockstep.
    :type filler: Callable[..., Any]
    """
    if class_is_patched(model_cls, ACCEPTS_VISION_ROWS):
        return
    cls: Any = model_cls
    setattr(cls, method, wrap(cls.__dict__[method]))
    cls.vision_filler_features = filler
    setattr(cls, ACCEPTS_VISION_ROWS, True)
