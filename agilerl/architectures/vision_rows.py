# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers that let a vision-language class take one ``(h, w, patch_dim)`` patch-grid row per image."""

from __future__ import annotations

import functools
import operator
from collections.abc import Callable
from typing import Any

import torch

from agilerl.utils.patching import class_is_patched

__all__ = [
    "ACCEPTS_VISION_ROWS",
    "SHARDS_IMAGE_FEATURES",
    "install_vision_rows",
    "packed_patches",
    "shard_image_features",
]

ACCEPTS_VISION_ROWS = "agilerl_accepts_vision_rows"
SHARDS_IMAGE_FEATURES = "agilerl_shards_image_features"


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


def shard_image_features(
    image_token_id: Callable[[Any], int] = operator.attrgetter("config.image_token_id"),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Run ``get_image_features`` with Ulysses off and keep the feature rows of this CP shard's image tokens.

    Outside a context-parallel forward with images the features pass through.
    Inside one, a tensor output keeps its indexed rows, and a ``pooler_output``
    tuple becomes one element of the indexed rows.

    :param image_token_id: Reads the image token id off the model.
    :return: Decorator of ``get_image_features(self, pixel_values, ...)``.
    """

    def decorate(get_image_features: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(get_image_features)
        def image_features_of_shard(self: object, *args: Any, **kwargs: Any) -> object:
            # Import cycle: agilerl.distributed imports agilerl.architectures, whose catalog imports this module.
            from agilerl.distributed.context_parallel import (
                IMAGE_TOKEN_SHARD,
                PublishedUlyssesParams,
            )

            shard = IMAGE_TOKEN_SHARD.get()
            with PublishedUlyssesParams({}):
                features: Any = get_image_features(self, *args, **kwargs)
            if shard is None:
                return features
            rows = shard.feature_rows(image_token_id(self))
            if isinstance(features, torch.Tensor):
                return features[rows]
            features.pooler_output = (torch.cat(features.pooler_output)[rows],)
            return features

        setattr(image_features_of_shard, SHARDS_IMAGE_FEATURES, True)
        return image_features_of_shard

    return decorate


def install_vision_rows(
    model_cls: type,
    method: str,
    wrap: Callable[[Callable[..., Any]], Callable[..., Any]],
    filler: Callable[..., Any],
    image_token_id: Callable[[Any], int] = operator.attrgetter("config.image_token_id"),
) -> None:
    """Wrap ``model_cls.<method>`` to take vision rows, shard ``get_image_features``, and set ``vision_filler_features``, once per class.

    :param model_cls: Vision-language class to patch.
    :type model_cls: type
    :param method: Name of the method that receives ``pixel_values``.
    :type method: str
    :param wrap: Returns the row-accepting method given the native one.
    :type wrap: Callable[[Callable[..., Any]], Callable[..., Any]]
    :param filler: ``vision_filler_features(self, device)`` for vision lockstep.
    :type filler: Callable[..., Any]
    :param image_token_id: Reads the image token id off the model, for :func:`shard_image_features`.
    :type image_token_id: Callable[[Any], int]
    """
    if class_is_patched(model_cls, ACCEPTS_VISION_ROWS):
        return
    cls: Any = model_cls
    setattr(cls, method, wrap(cls.__dict__[method]))
    cls.get_image_features = shard_image_features(image_token_id)(
        cls.__dict__["get_image_features"]
    )
    cls.vision_filler_features = filler
    setattr(cls, ACCEPTS_VISION_ROWS, True)
