# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Vision formula tests: golden numbers on known geometries."""

import pytest

from agilerl.arena.memory import vision
from agilerl.arena.memory.specs import GiB, MiB, ModelArch
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from tests.test_memory.test_formulas import (
    GEMMA4_E4B,
    QWEN35_08B,
    QWEN36,
    QWEN_05B,
    SUPER_VL,
)


class TestVisionFeatureCacheBytes:
    def test_merger_tower_keeps_patches_and_merged_tokens(self):
        # Qwen3.5-0.8B 1280x2048: 10240 patches of 768 (15 MiB) and 2560
        # merged tokens of 1024 (5 MiB), in pinned blocks of 16 and 8 MiB.
        arch = ModelArch.from_hf_config(QWEN35_08B.config)

        cached = vision.vision_feature_cache_bytes(arch, 10240, 2.0)

        assert cached == 24 * MiB

    def test_pooling_tower_keeps_pooled_tokens(self):
        # Gemma 4 E4B 1280x2048: 2457 patches pool 3x3 into 273 tokens of 768
        # (409.5 KiB), in one 512 KiB pinned block.
        arch = ModelArch.from_hf_config(GEMMA4_E4B.config)

        cached = vision.vision_feature_cache_bytes(arch, 2457, 2.0)

        assert cached == 512 * 1024


class TestVisionTowerLoraBytes:
    def test_merger_tower_matches_the_qwen_snapshot(self):
        # Qwen3.5-0.8B, two 1280x2048 images: 12 block inputs (360 MiB), bf16
        # pixels (60), tower and merger outputs (60), merger dropout (90).
        arch = ModelArch.from_hf_config(QWEN35_08B.config)

        held = vision.vision_tower_lora_bytes(arch, 20480, 2.0, 2.0, True)

        assert held == 570 * MiB

    def test_dropout_off_drops_the_merger_copies(self):
        arch = ModelArch.from_hf_config(QWEN35_08B.config)

        held = vision.vision_tower_lora_bytes(arch, 20480, 2.0, 2.0, False)

        assert held == 480 * MiB

    def test_pooling_tower_keeps_block_inputs_and_pixels(self):
        arch = ModelArch.from_hf_config(GEMMA4_E4B.config)

        held = vision.vision_tower_lora_bytes(arch, 2520, 2.0, 2.0, True)

        assert held == 2520 * (16 * 768 + 3 * 16 * 16) * 2

    def test_text_only_model_has_no_patch_rule(self):
        assert QWEN_05B.vision_resize == "none"

    def test_qwen36_tower_reads_the_encoder_profile_geometry(self):
        arch = ModelArch.from_hf_config(QWEN36.config)

        assert arch.vision_intermediate_size == 4304
        assert arch.vision_temporal_patch_size == 2
        assert arch.vision_out_hidden_size == 2048
        assert arch.vision_spatial_merge_size == 2
        assert arch.vision_profile_patches == 65536

    def test_unmeasured_family_has_no_profile_image(self):
        assert ModelArch.from_hf_config(SUPER_VL.config).vision_profile_patches == 0


# (width, height) -> patch grid the transformers 5.19 image processors cut:
# Qwen image_grid_thw, Gemma4 soft tokens x 3 x 3, Nemotron imgs_sizes / 16.
PROCESSOR_GRIDS = {
    "Qwen/Qwen3.5-0.8B": {
        (1280, 720): (44, 80),
        (1280, 2048): (128, 80),
        (1024, 768): (48, 64),
        (640, 480): (30, 40),
        (3000, 3000): (188, 188),
        (5000, 5000): (256, 256),
        (100, 100): (16, 16),
    },
    "google/gemma-4-E4B-it": {
        (1280, 720): (36, 66),
        (1280, 2048): (63, 39),
        (1024, 768): (42, 57),
        (640, 480): (42, 57),
        (3000, 3000): (48, 48),
        (6000, 20): (3, 840),
        (20, 6000): (840, 3),
    },
    "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16": {
        (1280, 720): (46, 80),
        (1280, 2048): (128, 80),
        (1024, 768): (48, 64),
        (640, 480): (30, 40),
        (3000, 3000): (114, 116),
        (64, 64): (32, 32),
        (100, 60): (26, 44),
    },
}


class TestVisionPatchGrid:
    @pytest.mark.parametrize(
        ("hub_id", "size", "grid"),
        [
            (hub_id, size, grid)
            for hub_id, grids in PROCESSOR_GRIDS.items()
            for size, grid in grids.items()
        ],
    )
    def test_matches_the_image_processor(self, hub_id, size, grid):
        arch = ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[hub_id].config)

        assert vision.vision_patch_grid(arch, *size) == grid

    def test_qwen36_shares_the_qwen35_grid(self):
        arch = ModelArch.from_hf_config(QWEN36.config)

        assert vision.vision_patch_grid(arch, 1280, 720) == (44, 80)

    def test_no_image_or_no_tower_has_no_grid(self):
        arch = ModelArch.from_hf_config(QWEN35_08B.config)

        assert vision.vision_patch_grid(arch, 0, 0) == (0, 0)
        assert vision.vision_patch_grid(QWEN_05B, 1280, 720) == (0, 0)

    def test_tower_without_a_resize_rule_has_no_grid(self):
        arch = ModelArch.from_hf_config(QWEN35_08B.config).model_copy(
            update={"vision_resize": "none"}
        )

        assert vision.vision_patch_grid(arch, 1280, 720) == (0, 0)


class TestVisionPixelBytes:
    @pytest.mark.parametrize(
        ("hub_id", "patch_dim"),
        [
            ("Qwen/Qwen3.5-0.8B", 3 * 2 * 16 * 16),
            ("google/gemma-4-E4B-it", 3 * 16 * 16),
            ("nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16", 3 * 16 * 16),
        ],
    )
    def test_is_fp32_patch_rows(self, hub_id, patch_dim):
        arch = ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[hub_id].config)

        assert vision.vision_pixel_bytes(arch, 3520) == 3520 * patch_dim * 4


class TestEncoderProfileBytes:
    def test_qwen36_profile_matches_the_measured_vllm_transient(self):
        # Arrange: vLLM 0.32 logged 1.67 GiB at TP=1 and 1.24 GiB at TP=2.
        arch = ModelArch.from_hf_config(QWEN36.config)

        # Act
        tp1, outputs = vision.encoder_profile_bytes(arch, 2.0, 1, 1)
        tp2, _ = vision.encoder_profile_bytes(arch, 2.0, 2, 1)

        # Assert: 65536 patches * (3*2*16*16 + 4*1152 + (3*1152 + 4304) / tp) * 2 B
        assert tp1 == 65536 * 13904 * 2
        assert tp2 == 65536 * 10024 * 2
        assert tp1 / GiB == pytest.approx(1.67, rel=0.03)
        assert tp2 / GiB == pytest.approx(1.24, rel=0.03)
        assert outputs == 16384 * 2048 * 2

    def test_model_without_a_profile_image_adds_nothing(self):
        assert vision.encoder_profile_bytes(QWEN_05B, 2.0, 1, 0) == (0, 0)

    def test_scales_with_the_images_in_the_batch(self):
        arch = ModelArch.from_hf_config(QWEN36.config)
        one = vision.encoder_profile_bytes(arch, 2.0, 1, 1)

        two = vision.encoder_profile_bytes(arch, 2.0, 1, 2)

        assert two == (2 * one[0], 2 * one[1])


class TestEncoderProfileItems:
    # Qwen3.6's largest image is 65536 patches, 16384 merged tokens.

    def test_default_token_cap_encodes_one_image(self):
        arch = ModelArch.from_hf_config(QWEN36.config)

        items = vision.encoder_profile_items(arch, 2048, 16, 32768, None)

        assert items == 1

    def test_token_cap_of_two_images_encodes_two(self):
        arch = ModelArch.from_hf_config(QWEN36.config)

        items = vision.encoder_profile_items(arch, 32768, 16, 32768, None)

        assert items == 2

    def test_image_limit_caps_a_single_sequence(self):
        arch = ModelArch.from_hf_config(QWEN36.config)

        items = vision.encoder_profile_items(arch, 65536, 1, 65536, 1)

        assert items == 1

    def test_text_model_has_no_profile_image(self):
        assert vision.encoder_profile_items(QWEN_05B, 32768, 16, 32768, None) == 0
