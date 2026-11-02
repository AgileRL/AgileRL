# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 / Qwen3.6 VL forward over one patch-grid row per image.

Training batches keep ``pixel_values`` as ``(N, h, w, patch_dim)`` rows. The
native forward reads flat patches with ``image_grid_thw`` and
``mm_token_type_ids``, and skips M-RoPE whenever ``position_ids`` are given.
The patch unfolds the rows and, for given 2-D ``position_ids``, builds
``[text, t, h, w]`` positions for every packed document at once. The text row
keeps document isolation and the GatedDeltaNet ``seq_idx`` the packed
forward derives from it.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F

from agilerl.architectures.nemotron_h_omni import packed_patches
from agilerl.architectures.qwen3_5.packed import install_qwen_gdn_patches, resolve_class
from agilerl.architectures.runtime import PatchRuntimeConfig
from agilerl.utils.patching import class_is_patched

if TYPE_CHECKING:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeVisionConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModel

logger = logging.getLogger(__name__)

__all__ = [
    "ACCEPTS_VISION_ROWS",
    "install_qwen3_5_vl_patches",
    "mrope_position_ids",
    "patch_qwen_vision_rows",
]

ACCEPTS_VISION_ROWS = "agilerl_accepts_vision_rows"

VISION_ROW_TARGETS = (
    "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5Model",
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeModel",
)


def mrope_position_ids(
    model: Qwen3_5Model | Qwen3_5MoeModel,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> torch.Tensor:
    """``(4, B, L)`` text and M-RoPE positions of packed or padded rows.

    Each document restarts its ``position_ids`` at 0 and takes the image grids
    of its image-token runs in row order. Documents with an image match
    ``get_rope_index`` on that document alone; text-only documents keep
    ``position_ids``. No host sync.

    :param model: The ``Qwen3_5Model`` whose config holds the image token and merge size.
    :param input_ids: ``(B, L)`` token ids.
    :param position_ids: ``(B, L)`` text positions.
    :param image_grid_thw: ``(N, 3)`` grid of every image in the batch, in order.
    :return: Text positions stacked on the temporal, height and width rows.
    """
    # The config builds this sub-config on init; the union types its input.
    vision = cast(
        "Qwen3_5VisionConfig | Qwen3_5MoeVisionConfig", model.config.vision_config
    )
    merge = vision.spatial_merge_size
    flat_positions = position_ids.flatten()
    index = torch.arange(flat_positions.numel(), device=flat_positions.device)
    doc_start = (index % position_ids.shape[1] == 0) | (flat_positions == 0)
    is_image = input_ids.flatten() == model.config.image_token_id
    after_image = F.pad(is_image[:-1], (1, 0))
    run_start = is_image & (doc_start | ~after_image)
    doc_first = torch.where(doc_start, index, 0).cummax(0).values
    run_first = torch.where(run_start, index, 0).cummax(0).values

    # Text lanes read image 0's grid; ``torch.where`` drops them below.
    grid = image_grid_thw[(run_start.cumsum(0) - 1).clamp(min=0)]
    grid_t, grid_h, grid_w = grid[:, 0], grid[:, 1] // merge, grid[:, 2] // merge
    # An image advances the position by its longer side, not by its token count.
    extra = torch.where(
        run_start, torch.maximum(grid_h, grid_w) - grid_t * grid_h * grid_w, 0
    )
    extra_total = extra.cumsum(0)
    shift = extra_total - (extra_total[doc_first] - extra[doc_first])
    shift = shift - torch.where(is_image, extra[run_first], 0)

    doc_id = doc_start.cumsum(0) - 1
    doc_images = torch.zeros_like(index).index_add_(0, doc_id, is_image.long())
    text = torch.where(
        doc_images[doc_id] > 0, index - doc_first + shift, flat_positions
    )
    base = run_first - doc_first + shift
    local = index - run_first
    vision = torch.stack(
        [
            base + local // (grid_h * grid_w),
            base + local // grid_w % grid_h,
            base + local % grid_w,
        ]
    )
    mrope = torch.where(is_image, vision, text).view(3, *position_ids.shape)
    return torch.cat([position_ids[None], mrope])


def _install_vision_rows(model_cls: type) -> None:
    """Let ``model_cls.forward`` take ``(N, h, w, patch_dim)`` ``pixel_values``."""
    cls: Any = model_cls
    forward = cls.__dict__["forward"]

    @functools.wraps(forward)
    def forward_with_vision_rows(
        self: Qwen3_5Model | Qwen3_5MoeModel,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> object:
        if pixel_values is not None and pixel_values.dim() == 4:
            if input_ids is None:
                msg = "Vision-row pixel_values need input_ids to place the images"
                raise ValueError(msg)
            pixel_values, image_grid_hw = packed_patches(pixel_values)
            image_grid_thw = F.pad(image_grid_hw, (1, 0), value=1)
            kwargs["image_grid_thw"] = image_grid_thw
            kwargs["mm_token_type_ids"] = (
                input_ids == self.config.image_token_id
            ).int()
            if position_ids is not None and position_ids.dim() == 2:
                position_ids = mrope_position_ids(
                    self, input_ids, position_ids, image_grid_thw
                )
        return forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            **kwargs,
        )

    cls.forward = forward_with_vision_rows
    cls.vision_filler_features = vision_filler_features


def vision_filler_features(
    self: Qwen3_5Model | Qwen3_5MoeModel, device: torch.device
) -> object:
    """Image features of one zero image on the smallest grid the vision merger takes.

    :param device: Device the zero image is built on.
    :return: ``get_image_features`` output for that image.
    """
    vision = self.visual.config
    side = vision.spatial_merge_size
    patch_dim = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    return self.get_image_features(
        torch.zeros(side * side, patch_dim, device=device),
        image_grid_thw=torch.tensor([[1, side, side]], device=device),
    )


def patch_qwen_vision_rows(model: str) -> None:
    """Let a Qwen VL model class take one patch-grid row per image.

    :param model: Dotted path of the ``Qwen3_5Model``-style class.
    """
    model_cls = resolve_class(model)
    if model_cls is None:
        logger.warning(
            "[qwen-vision] %s unavailable; vision rows are not supported",
            model.rpartition(".")[2],
        )
        return
    if class_is_patched(model_cls, ACCEPTS_VISION_ROWS):
        return
    _install_vision_rows(model_cls)
    setattr(model_cls, ACCEPTS_VISION_ROWS, True)


def install_qwen3_5_vl_patches(
    patch: PatchRuntimeConfig,
    model: object | None = None,
) -> None:
    """Install the Qwen3.5 / Qwen3.6 GDN and vision-row patches.

    :param patch: Catalog patch config.
    :param model: Already-built model the patches also apply to, or None.
    """
    install_qwen_gdn_patches(patch, model)
    for target in VISION_ROW_TARGETS:
        patch_qwen_vision_rows(target)
