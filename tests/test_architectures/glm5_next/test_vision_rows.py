# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 vision rows and filler image (CPU, tiny weights)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers.models.glm5_next.configuration_glm5_next import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)

from agilerl.algorithms.core.llm_ops.vision_lockstep import vision_language_module
from agilerl.architectures.catalog import model_type_vision
from agilerl.architectures.glm5_next import vision
from agilerl.architectures.glm5_next.vision import install_glm5_next_vision
from agilerl.architectures.runtime import PatchRuntimeConfig
from agilerl.architectures.vision_rows import ACCEPTS_VISION_ROWS, packed_patches
from agilerl.arena.memory.specs import ModelArch
from agilerl.arena.memory.vision import vision_patch_grid
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from tests.helper_functions import load_without_llm_dependencies

HIDDEN = 32
IMAGE_TOKEN_ID = 5
PATCH_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
MERGE = 2
# Three channels, two frames, 2x2 pixels.
PATCH_DIM = 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE**2


def _tiny_config() -> Glm5NextConfig:
    text = Glm5NextTextConfig(
        vocab_size=32,
        hidden_size=HIDDEN,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        layer_types=["indexed_attention", "indexed_attention"],
        mlp_layer_types=["dense", "sparse"],
        indexer_types=["full", "full"],
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_rope_head_dim=0,
        qk_nope_head_dim=16,
        v_head_dim=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=16,
        index_kpool=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        hc_mult=2,
        pad_token_id=0,
    )
    vision_config = Glm5NextVisionConfig(
        depth=1,
        hidden_size=HIDDEN,
        num_heads=2,
        intermediate_size=64,
        out_hidden_size=HIDDEN,
        projection_intermediate_size=64,
        patch_size=PATCH_SIZE,
        spatial_merge_size=MERGE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        image_size=8,
    )
    config = Glm5NextConfig(
        text_config=text,
        vision_config=vision_config,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=IMAGE_TOKEN_ID,
        image_start_token_id=6,
        image_end_token_id=7,
        video_start_token_id=8,
        video_end_token_id=9,
    )
    config._attn_implementation = "sdpa"
    return config


def _tiny_model() -> Glm5NextForConditionalGeneration:
    install_glm5_next_vision(PatchRuntimeConfig())
    torch.manual_seed(0)
    return Glm5NextForConditionalGeneration(_tiny_config()).eval()


class TestInstallGlm5NextVision:
    def test_patch_is_idempotent(self) -> None:
        install_glm5_next_vision(PatchRuntimeConfig())
        wrapped = Glm5NextModel.get_image_features
        install_glm5_next_vision(PatchRuntimeConfig())

        assert Glm5NextModel.get_image_features is wrapped
        assert vars(Glm5NextModel)[ACCEPTS_VISION_ROWS] is True

    def test_makes_the_inner_model_the_vision_language_module(self) -> None:
        model = _tiny_model()

        assert vision_language_module(model) is model.model

    def test_without_llm_dependencies_leaves_the_model_class_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = load_without_llm_dependencies(vision, monkeypatch)

        assert module.Glm5NextModel is None

    def test_catalog_installs_the_vision_patch(self) -> None:
        vision_support = model_type_vision("glm5_next")

        assert vision_support is not None
        assert vision_support.install is install_glm5_next_vision

    def test_catalog_names_the_processor_image_token(self) -> None:
        # The GLM-5.3 tokenizer names no image_token; its processor expands <|image|>.
        vision_support = model_type_vision("glm5_next")

        assert vision_support is not None
        assert vision_support.image_token == "<|image|>"
        assert model_type_vision("glm5_next_text") is None


@pytest.mark.vllm
class TestGlm5NextImageProcessorResize:
    def test_matches_the_memory_estimator_patch_grid(self) -> None:
        # Arrange
        pytest.importorskip("torchvision")
        # Deferred: the GLM-5.3 image processor needs torchvision, which agilerl does not install.
        from transformers.models.glm5_next.image_processing_glm5_next import (
            Glm5NextImageProcessor,
            smart_resize,
        )

        processor = Glm5NextImageProcessor
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["zai-org/GLM-5.3-Flash"].config
        )
        # 112 squared is the 16-token floor; 2800 x 2240 fills the 8000-token cap.
        sides = [8, 27, 29, 100, 112, 457, 1280, 2048, 2240, 2800, 4097, 6000]
        sizes = [(w, h) for w in sides for h in sides]

        # Act
        from_processor = {}
        from_estimator = {}
        for width, height in sizes:
            resized = smart_resize(
                num_frames=processor.temporal_patch_size,
                height=height,
                width=width,
                temporal_factor=processor.temporal_patch_size,
                factor=processor.patch_size * processor.merge_size,
                min_pixels=processor.min_image_tokens,
                max_pixels=processor.max_image_tokens,
            )
            from_processor[width, height] = (
                resized[0] // processor.patch_size,
                resized[1] // processor.patch_size,
            )
            from_estimator[width, height] = vision_patch_grid(arch, width, height)

        # Assert
        assert from_estimator == from_processor


class TestGlm5NextGetImageFeatures:
    def test_patch_grid_rows_forward_matches_packed_patches(self) -> None:
        # Arrange
        model = _tiny_model()
        # One 2x4 patch grid, which merges 2x2 into two image tokens.
        rows = torch.randn(1, 2, 4, PATCH_DIM)
        patches, image_grid_hw = packed_patches(rows)
        image_grid_thw = F.pad(image_grid_hw, (1, 0), value=1)
        input_ids = torch.tensor([[1, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 2, 3]])

        # Act
        with torch.no_grad():
            from_rows = model(input_ids=input_ids, pixel_values=rows).logits
            from_patches = model(
                input_ids=input_ids,
                pixel_values=patches,
                image_grid_thw=image_grid_thw,
            ).logits

        # Assert
        assert torch.equal(from_rows, from_patches)

    def test_image_changes_the_logits_of_later_tokens(self) -> None:
        # Arrange
        model = _tiny_model()
        input_ids = torch.tensor([[1, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 2, 3]])
        torch.manual_seed(1)
        first, second = torch.randn(2, 1, 2, 4, PATCH_DIM)

        # Act
        with torch.no_grad():
            first_logits = model(input_ids=input_ids, pixel_values=first).logits
            second_logits = model(input_ids=input_ids, pixel_values=second).logits

        # Assert
        assert torch.equal(first_logits[:, 0], second_logits[:, 0])
        assert not torch.allclose(first_logits[:, -1], second_logits[:, -1])

    def test_text_only_forward_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        model = _tiny_model()
        input_ids = torch.tensor([[1, 2, 3, 4]])

        # Act
        with torch.no_grad():
            patched = model(input_ids=input_ids).logits
            monkeypatch.setattr(
                Glm5NextModel,
                "get_image_features",
                Glm5NextModel.get_image_features.__wrapped__,
            )
            stock = model(input_ids=input_ids).logits

        # Assert
        assert torch.equal(patched, stock)


class TestGlm5NextVisionFillerFeatures:
    def test_runs_the_tower_once_and_returns_one_merged_token(self) -> None:
        # Arrange
        model = _tiny_model()
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
        assert [tuple(image.shape) for image in features.pooler_output] == [(1, HIDDEN)]
