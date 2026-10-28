# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 / Qwen3.6 VL forward over one patch-grid row per image.

Training batches keep ``pixel_values`` as ``(N, h, w, patch_dim)`` rows. The
native forward reads flat patches with ``image_grid_thw`` and
``mm_token_type_ids``, and skips M-RoPE whenever ``position_ids`` are given.
The patch unfolds the rows and, for given 2-D ``position_ids``, builds
``[text, t, h, w]`` positions one packed document at a time. The text row
keeps document isolation and the GatedDeltaNet ``seq_idx`` the packed
forward derives from it.
"""

from __future__ import annotations

import functools
import itertools
import logging
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F

from agilerl.architectures.nemotron_h_omni import packed_patches
from agilerl.architectures.qwen3_5.packed import install_qwen_gdn_patches, resolve_class
from agilerl.architectures.runtime import PatchRuntimeConfig
from agilerl.utils.patching import class_is_patched

if TYPE_CHECKING:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
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
    of its image-token runs in row order.

    :param model: The ``Qwen3_5Model`` whose ``get_rope_index`` lays out images.
    :param input_ids: ``(B, L)`` token ids.
    :param position_ids: ``(B, L)`` text positions.
    :param image_grid_thw: ``(N, 3)`` grid of every image in the batch, in order.
    :return: Text positions stacked on the temporal, height and width rows.
    """
    image_token_id = model.config.image_token_id
    grids = iter(image_grid_thw)
    rows = []
    for row_ids, row_positions in zip(input_ids, position_ids, strict=True):
        starts = torch.nonzero(row_positions == 0).flatten().tolist()
        bounds = sorted({0, *starts, len(row_ids)})
        parts = []
        for start, end in itertools.pairwise(bounds):
            doc_ids = row_ids[start:end]
            doc_types = (doc_ids == image_token_id).int()
            images = int(doc_types[0]) + int(
                ((doc_types[1:] == 1) & (doc_types[:-1] == 0)).sum()
            )
            if images == 0:
                parts.append(row_positions[start:end].expand(3, -1))
                continue
            # torch ops type their results as Tensor, never LongTensor.
            doc_grids = cast(
                "torch.LongTensor",
                torch.stack([next(grids) for _ in range(images)]),
            )
            doc_positions, _ = model.get_rope_index(
                doc_ids[None], doc_types[None], image_grid_thw=doc_grids
            )
            parts.append(doc_positions[:, 0])
        rows.append(torch.cat(parts, dim=1))
    return torch.cat([position_ids[None], torch.stack(rows, dim=1)])


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
