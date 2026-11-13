# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""MiMo-V2 image processor, computed as vLLM 0.32's ``MiMoOmniProcessor`` computes images.

The checkpoint's ``preprocessor_config.json`` names Qwen2-VL's processor,
which resizes bicubic on PIL. vLLM resizes bilinear in torch with its own
smart-resize rule and reads the pixel limits from ``config.json``'s
``processor_config``. Training must see the pixels and image-token counts
rollouts saw, so this processor follows vLLM.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BatchEncoding, BatchFeature, PretrainedConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

if TYPE_CHECKING:
    from PIL import Image

__all__ = ["MiMoV2Processor", "smart_resize"]

PIXEL_MEAN = (123.675, 116.28, 103.53)
PIXEL_STD = (58.395, 57.12, 57.375)
VISION_START = "<|vision_start|>"
IMAGE_PAD = "<|image_pad|>"
VISION_END = "<|vision_end|>"
IMAGE_PLACEHOLDER = re.compile(
    re.escape(VISION_START) + f"(?:{re.escape(IMAGE_PAD)})+" + re.escape(VISION_END)
)


def smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """Height and width, multiples of ``factor``, that vLLM resizes an image to.

    :param height: Image height in pixels.
    :param width: Image width in pixels.
    :param factor: Patch size times merge size.
    :param min_pixels: Fewest pixels after the resize.
    :param max_pixels: Most pixels after the resize.
    :return: Resized height and width.
    :raises ValueError: If one side is over 200 times the other.
    """
    if min(height, width) < factor:
        if height < width:
            height, width = factor, int(width * factor / height)
        else:
            width, height = factor, int(height * factor / width)
    elif max(height, width) / min(height, width) > 200:
        msg = f"Aspect ratio > 200 not allowed: {height}x{width}"
        raise ValueError(msg)
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        resized_height = math.floor(height / beta / factor) * factor
        resized_width = math.floor(width / beta / factor) * factor
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor
    return int(resized_height), int(resized_width)


class MiMoV2Processor:
    """Expands each image placeholder into its image tokens and turns images into vision-tower patches.

    :param tokenizer: The checkpoint tokenizer.
    :param patch_size: Vision patch side in pixels.
    :param merge_size: Patches per image-token side.
    :param temporal_patch_size: Frames per patch; an image repeats to fill them.
    :param min_pixels: Fewest pixels an image resizes to.
    :param max_pixels: Most pixels an image resizes to.
    """

    image_token = IMAGE_PAD

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        patch_size: int,
        merge_size: int,
        temporal_patch_size: int,
        min_pixels: int,
        max_pixels: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> MiMoV2Processor:
        """Build from a checkpoint's tokenizer and ``config.json`` ``processor_config``.

        :param model_name_or_path: Hugging Face model id or local path.
        :param kwargs: Tokenizer ``from_pretrained`` kwargs.
        :return: The processor.
        """
        config, _ = PretrainedConfig.get_config_dict(model_name_or_path)
        processor_config = config["processor_config"]
        return cls(
            cast(
                "PreTrainedTokenizerBase",
                AutoTokenizer.from_pretrained(model_name_or_path, **kwargs),
            ),
            patch_size=processor_config["patch_size"],
            merge_size=processor_config["merge_size"],
            temporal_patch_size=processor_config["temporal_patch_size"],
            min_pixels=processor_config["image_min_pixels"],
            max_pixels=processor_config["image_max_pixels"],
        )

    def apply_chat_template(
        self, conversation: list[dict[str, Any]], /, **kwargs: Any
    ) -> str | list[int] | list[str] | list[list[int]] | BatchEncoding:
        """Render ``conversation`` with the tokenizer's chat template.

        :param conversation: Chat messages.
        :param kwargs: ``apply_chat_template`` kwargs.
        :return: The tokenizer's output.
        """
        return self.tokenizer.apply_chat_template(conversation, **kwargs)

    def image_patches(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        """Resize, standardize and patch one image.

        :param image: The image.
        :return: ``(patches, C * temporal * patch**2)`` patches in merge-window
            order and the ``(3,)`` frame, row and column patch grid.
        """
        pixels = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)
        _, height, width = pixels.shape
        factor = self.patch_size * self.merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, self.min_pixels, self.max_pixels
        )
        resized = F.interpolate(
            pixels.float().unsqueeze(0),
            (resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )[0]
        mean = torch.tensor(PIXEL_MEAN).view(-1, 1, 1)
        std = torch.tensor(PIXEL_STD).view(-1, 1, 1)
        frames = ((resized - mean) / std).unsqueeze(0)
        frames = frames.repeat(self.temporal_patch_size, 1, 1, 1)
        channels = frames.shape[1]
        grid_t = frames.shape[0] // self.temporal_patch_size
        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size
        merge = self.merge_size
        patches = (
            frames.view(
                grid_t,
                self.temporal_patch_size,
                channels,
                grid_h // merge,
                merge,
                self.patch_size,
                grid_w // merge,
                merge,
                self.patch_size,
            )
            .permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
            .reshape(
                grid_t * grid_h * grid_w,
                channels * self.temporal_patch_size * self.patch_size**2,
            )
        )
        return patches, torch.tensor([grid_t, grid_h, grid_w])

    def __call__(
        self,
        text: str,
        images: Image.Image | list[Image.Image] | None = None,
        return_tensors: str | None = None,
    ) -> BatchFeature:
        """Tokenize ``text`` with each image placeholder expanded, and patch the images.

        :param text: Prompt with one ``<|vision_start|><|image_pad|><|vision_end|>``
            placeholder per image.
        :param images: The images, in placeholder order.
        :param return_tensors: Tensor type of the outputs.
        :return: ``(1, tokens)`` ``input_ids`` and ``attention_mask``; with
            images, also ``pixel_values`` and ``image_grid_thw``.
        :raises ValueError: If the placeholders and images differ in number.
        """
        # Optional: Pillow, only when an image is turned into patches.
        from PIL import Image

        if images is None:
            images = []
        elif isinstance(images, Image.Image):
            images = [images]
        placeholders = len(IMAGE_PLACEHOLDER.findall(text))
        if placeholders != len(images):
            msg = f"{placeholders} image placeholders for {len(images)} images"
            raise ValueError(msg)
        encoded = [self.image_patches(image) for image in images]
        tokens_per_image = iter(
            int(grid.prod()) // self.merge_size**2 for _, grid in encoded
        )
        expanded = IMAGE_PLACEHOLDER.sub(
            lambda _match: (
                VISION_START + IMAGE_PAD * next(tokens_per_image) + VISION_END
            ),
            text,
        )
        data = dict(self.tokenizer([expanded]))
        if encoded:
            data["pixel_values"] = torch.cat([patches for patches, _ in encoded])
            data["image_grid_thw"] = torch.stack([grid for _, grid in encoded])
        return BatchFeature(data=data, tensor_type=return_tensors)
