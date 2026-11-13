# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""MiMo-V2 image processor: vLLM's resize rule, patch order and placeholder expansion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast

from agilerl.architectures.mimo_v2.processing import (
    IMAGE_PAD,
    PIXEL_MEAN,
    PIXEL_STD,
    VISION_END,
    VISION_START,
    MiMoV2Processor,
    smart_resize,
)
from agilerl.arena.memory.specs import ModelArch
from agilerl.arena.memory.vision import vision_patch_grid
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from tests.test_architectures.mimo_v2.checkpoint_helpers import tiny_hub_config

PLACEHOLDER = VISION_START + IMAGE_PAD + VISION_END
VOCAB = {"[UNK]": 0, "a": 1, "b": 2, VISION_START: 3, IMAGE_PAD: 4, VISION_END: 5}


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    """Word-level tokenizer that splits on the vision tokens and spaces."""
    tokenizer = Tokenizer(WordLevel(VOCAB, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Split(" ", behavior="removed")
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
    fast.add_special_tokens(
        {"additional_special_tokens": [VISION_START, IMAGE_PAD, VISION_END]}
    )
    return fast


def tiny_processor() -> MiMoV2Processor:
    return MiMoV2Processor(
        tiny_tokenizer(),
        patch_size=2,
        merge_size=2,
        temporal_patch_size=2,
        min_pixels=16,
        max_pixels=4096,
    )


def position_image(height: int, width: int) -> Image.Image:
    """RGB image whose red channel is the row, green the column, blue constant."""
    rows, cols = np.mgrid[0:height, 0:width]
    pixels = np.stack([rows, cols, np.full_like(rows, 7)], axis=-1)
    return Image.fromarray(pixels.astype(np.uint8))


class TestSmartResize:
    @pytest.mark.parametrize(
        ("height", "width", "max_pixels", "expected"),
        [
            (224, 224, 8388608, (224, 224)),
            (230, 250, 8388608, (224, 256)),
            (10, 100, 8388608, (32, 320)),
            (4000, 4000, 1_000_000, (992, 992)),
        ],
        ids=["multiple", "rounds", "upscales-short-side", "caps-pixels"],
    )
    def test_resizes_to_factor_multiples_within_the_pixel_limits(
        self, height: int, width: int, max_pixels: int, expected: tuple[int, int]
    ) -> None:
        assert smart_resize(height, width, 32, 8192, max_pixels) == expected

    def test_raises_past_aspect_ratio_200(self) -> None:
        with pytest.raises(ValueError, match="Aspect ratio > 200"):
            smart_resize(32, 7000, 32, 8192, 8388608)

    def test_matches_the_memory_estimator_patch_grid(self) -> None:
        # Arrange
        config = SUPPORTED_MODEL_INFO["XiaomiMiMo/MiMo-V2.6-Flash-RL"].config
        bounds = config["processor_config"]
        patch = bounds["patch_size"]
        arch = ModelArch.from_hf_config(config)
        sides = [8, 31, 33, 100, 457, 1280, 2048, 4097, 6000]
        sizes = [(w, h) for w in sides for h in sides if max(w, h) / min(w, h) <= 200]

        # Act
        processor = {}
        estimator = {}
        for width, height in sizes:
            resized = smart_resize(
                height,
                width,
                patch * bounds["merge_size"],
                bounds["image_min_pixels"],
                bounds["image_max_pixels"],
            )
            processor[width, height] = (resized[0] // patch, resized[1] // patch)
            estimator[width, height] = vision_patch_grid(arch, width, height)

        # Assert
        assert estimator == processor


class TestMiMoV2ProcessorImagePatches:
    def test_patches_are_merge_windows_of_channel_frame_pixel_rows(self) -> None:
        # Arrange
        processor = tiny_processor()

        # Act
        patches, grid = processor.image_patches(position_image(4, 4))

        # Assert
        assert grid.tolist() == [1, 2, 2]
        assert patches.shape == (4, 3 * 2 * 2 * 2)
        mean = torch.tensor(PIXEL_MEAN)[:, None, None, None]
        std = torch.tensor(PIXEL_STD)[:, None, None, None]
        # The second patch of the first merge window covers rows 0-1, columns 2-3.
        rows = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        cols = torch.tensor([[2.0, 3.0], [2.0, 3.0]])
        raw = torch.stack([rows, cols, torch.full((2, 2), 7.0)])[:, None]
        expected = ((raw.expand(3, 2, 2, 2) - mean) / std).flatten()
        torch.testing.assert_close(patches[1], expected)


class TestMiMoV2ProcessorCall:
    def test_expands_each_placeholder_to_its_image_tokens(self) -> None:
        # Arrange
        processor = tiny_processor()
        image = position_image(8, 4)

        # Act
        batch = processor(f"a {PLACEHOLDER} b", images=image, return_tensors="pt")

        # Assert
        assert batch["input_ids"].tolist() == [[1, 3, 4, 4, 5, 2]]
        assert batch["image_grid_thw"].tolist() == [[1, 4, 2]]
        assert batch["pixel_values"].shape == (8, 24)

    def test_text_without_images_has_no_pixels(self) -> None:
        batch = tiny_processor()("a b", return_tensors="pt")

        assert batch["input_ids"].tolist() == [[1, 2]]
        assert "pixel_values" not in batch

    def test_placeholders_that_miss_the_images_raise(self) -> None:
        with pytest.raises(ValueError, match="2 image placeholders for 1 images"):
            tiny_processor()(
                f"{PLACEHOLDER} {PLACEHOLDER}", images=[position_image(4, 4)]
            )


class TestMiMoV2ProcessorFromPretrained:
    def test_reads_the_pixel_limits_from_config_processor_config(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        tiny_tokenizer().save_pretrained(tmp_path)
        processor_config = {
            "image_min_pixels": 8192,
            "image_max_pixels": 8388608,
            "patch_size": 16,
            "merge_size": 2,
            "temporal_patch_size": 2,
        }
        (tmp_path / "config.json").write_text(
            json.dumps({**tiny_hub_config(), "processor_config": processor_config})
        )

        # Act
        processor = MiMoV2Processor.from_pretrained(str(tmp_path))

        # Assert
        assert (
            processor.patch_size,
            processor.merge_size,
            processor.temporal_patch_size,
            processor.min_pixels,
            processor.max_pixels,
        ) == (16, 2, 2, 8192, 8388608)
        assert processor.tokenizer.convert_tokens_to_ids(IMAGE_PAD) == 4
