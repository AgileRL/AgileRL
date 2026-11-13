# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""MiMo-V2 vision rows, filler image and tower attention (CPU, tiny weights)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from agilerl.algorithms.core.llm_ops.frozen_vision import (
    VisionFeatureCache,
    install_frozen_vision_no_grad,
)
from agilerl.algorithms.core.llm_ops.vision_lockstep import vision_language_module
from agilerl.architectures.catalog import FAMILY_RUNTIME_CONFIGS, model_type_vision
from agilerl.architectures.mimo_v2 import modeling as vision_tower
from agilerl.architectures.mimo_v2.configuration import MiMoV2Config
from agilerl.architectures.mimo_v2.modeling import (
    MiMoV2ForConditionalGeneration,
    window_attention,
)
from agilerl.architectures.mimo_v2.processing import MiMoV2Processor
from agilerl.architectures.vision_rows import packed_patches
from tests.test_architectures.mimo_v2.checkpoint_helpers import (
    IMAGE_TOKEN_ID,
    MERGE,
    PATCH_DIM,
)

OUT_HIDDEN = 64
# fp32 SDPA over a chunk and over the whole image differ only in summation order.
FP32_TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}
INPUT_IDS = torch.tensor([[1, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 2, 3]])
TWO_IMAGE_IDS = torch.tensor([[1, *[IMAGE_TOKEN_ID] * 4, 2, 3]])


@pytest.fixture
def model(native_config: MiMoV2Config) -> MiMoV2ForConditionalGeneration:
    torch.manual_seed(0)
    model = MiMoV2ForConditionalGeneration(native_config).eval()
    # post_init zeroes the sinks; nonzero ones reach the window blocks' output.
    with torch.no_grad():
        for block in model.model.visual.blocks:
            if block.attn.sinks is not None:
                block.attn.sinks.normal_()
    return model


def dense_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
    sinks: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Softmax over all keys within ``window``, with ``sinks`` added to key 0's logit."""
    positions = torch.arange(q.shape[1])
    allowed = (positions[:, None] - positions[None, :]).abs() <= window
    logits = (q @ k.transpose(-1, -2)) * scale
    logits[:, :, 0] = logits[:, :, 0] + sinks[:, None]
    weights = logits.masked_fill(~allowed, float("-inf")).softmax(-1)
    return weights @ v


class TestMiMoV2VisionLanguageModule:
    def test_inner_model_is_the_vision_language_module(self, model) -> None:
        assert vision_language_module(model) is model.model

    def test_decoder_is_the_language_model_and_encoder_the_tower(self, model) -> None:
        assert model.get_decoder() is model.model.language_model
        assert model.get_encoder(modality="image") is model.model.visual

    def test_only_window_blocks_have_sinks(self, model) -> None:
        sinks = [block.attn.sinks is not None for block in model.model.visual.blocks]

        assert sinks == [False, True, True]


class TestMiMoV2GetImageFeatures:
    def test_patch_grid_rows_match_the_tower_on_packed_patches(self, model) -> None:
        # Arrange
        torch.manual_seed(1)
        # One 2x4 patch grid, which merges 2x2 into two image tokens.
        rows = torch.randn(1, 2, 4, PATCH_DIM)
        patches, image_grid_hw = packed_patches(rows)

        # Act
        with torch.no_grad():
            from_rows = model.model.get_image_features(rows)
            from_patches = model.model.visual(
                patches, F.pad(image_grid_hw, (1, 0), value=1)
            )

        # Assert
        assert from_rows.shape == (2, OUT_HIDDEN)
        assert torch.equal(from_rows, from_patches)

    def test_image_changes_the_logits_of_later_tokens(self, model) -> None:
        # Arrange
        torch.manual_seed(1)
        first, second = torch.randn(2, 1, 2, 4, PATCH_DIM)

        # Act
        with torch.no_grad():
            first_logits = model(input_ids=INPUT_IDS, pixel_values=first).logits
            second_logits = model(input_ids=INPUT_IDS, pixel_values=second).logits

        # Assert
        # Matmuls over the whole sequence round the first token differently.
        torch.testing.assert_close(
            first_logits[:, 0], second_logits[:, 0], **FP32_TOLERANCE
        )
        assert not torch.allclose(first_logits[:, -1], second_logits[:, -1])

    def test_text_only_forward_is_the_language_model_and_head(self, model) -> None:
        # Arrange
        input_ids = torch.tensor([[1, 2, 3, 4]])

        # Act
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
            hidden = model.model.language_model(input_ids=input_ids).last_hidden_state

        # Assert
        assert torch.equal(logits, model.lm_head(hidden))

    def test_image_tokens_that_miss_the_image_positions_raise(self, model) -> None:
        rows = torch.randn(1, 2, 4, PATCH_DIM)

        with pytest.raises(ValueError, match="1 image positions for 2 image tokens"):
            model(input_ids=torch.tensor([[1, IMAGE_TOKEN_ID, 2]]), pixel_values=rows)

    def test_pixels_without_input_ids_raise(self, model) -> None:
        rows = torch.randn(1, 2, 4, PATCH_DIM)
        embeds = model.get_input_embeddings()(INPUT_IDS)

        with pytest.raises(ValueError, match="pixel_values need input_ids"):
            model(inputs_embeds=embeds, pixel_values=rows)


class TestMiMoV2VisionFillerFeatures:
    def test_runs_the_tower_once_and_returns_one_merged_token(self, model) -> None:
        # Arrange
        tower_inputs: list[torch.Tensor] = []
        model.model.visual.register_forward_hook(
            lambda _module, args, _output: tower_inputs.append(args[0])
        )

        # Act
        with torch.no_grad():
            features = model.model.vision_filler_features(torch.device("cpu"))

        # Assert
        assert [tuple(patches.shape) for patches in tower_inputs] == [
            (MERGE * MERGE, PATCH_DIM)
        ]
        assert features.shape == (1, OUT_HIDDEN)


def frozen_tower_cache(
    model: MiMoV2ForConditionalGeneration,
) -> tuple[VisionFeatureCache, list[int]]:
    """Install a feature cache on the frozen tower and count its first block's calls."""
    model.requires_grad_(False)
    cache = VisionFeatureCache()
    assert install_frozen_vision_no_grad(model, cache) == "model.visual"
    calls = [0]
    model.model.visual.blocks[0].register_forward_pre_hook(
        lambda *_: calls.__setitem__(0, calls[0] + 1)
    )
    return cache, calls


class TestMiMoV2VisionFeatureCache:
    def test_packed_patches_reuse_the_tower_output(self, model) -> None:
        # Arrange
        torch.manual_seed(1)
        rows = torch.randn(2, 2, 4, PATCH_DIM)
        with torch.no_grad():
            expected = model(input_ids=TWO_IMAGE_IDS, pixel_values=rows).logits
        cache, calls = frozen_tower_cache(model)

        # Act
        with torch.no_grad(), cache.step(), cache.images(torch.arange(2)):
            outputs = [
                model(input_ids=TWO_IMAGE_IDS, pixel_values=rows).logits
                for _ in range(3)
            ]

        # Assert
        assert calls == [1]
        assert all(torch.equal(output, expected) for output in outputs)

    def test_reordered_keys_return_each_image_in_call_order(self, model) -> None:
        # Arrange
        torch.manual_seed(1)
        rows = torch.randn(2, 2, 4, PATCH_DIM)
        with torch.no_grad():
            expected = model(input_ids=TWO_IMAGE_IDS, pixel_values=rows.flip(0)).logits
        cache, calls = frozen_tower_cache(model)

        # Act
        with torch.no_grad(), cache.step():
            with cache.images(torch.tensor([0, 1])):
                model(input_ids=TWO_IMAGE_IDS, pixel_values=rows)
            with cache.images(torch.tensor([1, 0])):
                reordered = model(
                    input_ids=TWO_IMAGE_IDS, pixel_values=rows.flip(0)
                ).logits

        # Assert
        assert calls == [1]
        torch.testing.assert_close(reordered, expected, **FP32_TOLERANCE)


class TestWindowAttention:
    def test_matches_dense_attention_within_the_window_with_a_key_zero_sink(
        self,
    ) -> None:
        # Arrange
        torch.manual_seed(0)
        q, k, v = torch.randn(3, 2, 12, 8)
        sinks = torch.tensor([0.5, -1.0])

        # Act
        output = window_attention(q, k, v, 3, sinks, 8**-0.5)

        # Assert
        expected = dense_window_attention(q, k, v, 3, sinks, 8**-0.5)
        torch.testing.assert_close(output, expected, **FP32_TOLERANCE)

    def test_query_chunks_match_one_chunk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        torch.manual_seed(0)
        q, k, v = torch.randn(3, 2, 12, 8)
        sinks = torch.tensor([0.5, -1.0])
        whole = window_attention(q, k, v, 3, sinks, 8**-0.5)
        monkeypatch.setattr(vision_tower, "WINDOW_QUERY_CHUNK", 5)

        # Act
        chunked = window_attention(q, k, v, 3, sinks, 8**-0.5)

        # Assert
        torch.testing.assert_close(chunked, whole, **FP32_TOLERANCE)


class TestMiMoV2VisionModelColumnOrder:
    def test_orders_merge_windows_column_major_per_image(self, model) -> None:
        # Two images: 2x2 and 1x2 merge windows.
        grid_thw = torch.tensor([[1, 4, 4], [1, 2, 4]])

        order = model.model.visual.column_order(grid_thw, torch.device("cpu"))

        assert order.tolist() == [0, 2, 1, 3, 4, 5]

    def test_reorder_moves_whole_merge_windows(self, model) -> None:
        visual = model.model.visual
        states = torch.arange(8)

        reordered = visual.reorder(states, torch.tensor([1, 0]))

        assert reordered.tolist() == [4, 5, 6, 7, 0, 1, 2, 3]


class TestMiMoV2CatalogVision:
    def test_catalog_serves_and_trains_the_vision_language_model(self) -> None:
        runtime = FAMILY_RUNTIME_CONFIGS["mimo_v2"]
        vision = model_type_vision("mimo_v2")

        assert vision is not None
        assert vision.image_token == "<|image_pad|>"
        assert vision.processor_class is MiMoV2Processor
        assert runtime.vllm.hf_overrides == {"architectures": ["MiMoV2OmniForCausalLM"]}
        assert vision.tower_modalities == {"visual": frozenset({"image", "video"})}
