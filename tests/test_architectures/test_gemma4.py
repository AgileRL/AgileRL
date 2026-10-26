# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 image features from patch-grid rows match the processor's padded layout."""

import pytest
import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForConditionalGeneration,
    Gemma4Model,
)

from agilerl.algorithms.core.llm_ops.vision_lockstep import FILLER_METHOD
from agilerl.architectures.gemma4 import (
    ACCEPTS_VISION_ROWS,
    install_gemma4_vision_patches,
    patch_positions,
)
from agilerl.architectures.runtime import PatchRuntimeConfig

IMAGE, BOI, EOI = 100, 101, 102
GRID = 4
PATCH_DIM = 3 * 2 * 2
TOKENS_PER_IMAGE = GRID * GRID // 4


@pytest.fixture
def pristine_gemma4_model():
    """Restore ``Gemma4Model.get_image_features`` after a class-level patch."""
    get_image_features = Gemma4Model.__dict__["get_image_features"]
    yield
    Gemma4Model.get_image_features = get_image_features
    for name in (ACCEPTS_VISION_ROWS, FILLER_METHOD):
        if name in vars(Gemma4Model):
            delattr(Gemma4Model, name)


def tiny_gemma4() -> Gemma4ForConditionalGeneration:
    torch.manual_seed(0)
    config = Gemma4Config(
        text_config={
            "vocab_size": 128,
            "vocab_size_per_layer_input": 128,
            "hidden_size": 32,
            "hidden_size_per_layer_input": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "global_head_dim": 16,
            "intermediate_size": 64,
            "num_kv_shared_layers": 0,
            "sliding_window": 8,
            "pad_token_id": 0,
        },
        vision_config={
            "num_hidden_layers": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "global_head_dim": 8,
            "patch_size": 2,
            "pooling_kernel_size": 2,
            "position_embedding_size": 64,
        },
        audio_config=None,
        image_token_id=IMAGE,
        boi_token_id=BOI,
        eoi_token_id=EOI,
        audio_token_id=103,
        boa_token_id=104,
        eoa_token_id=105,
        video_token_id=106,
    )
    return Gemma4ForConditionalGeneration(config).eval()


def padded_inputs(
    rows: torch.Tensor, max_patches: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The processor's layout: shuffled patches padded to ``max_patches`` with ``-1`` positions."""
    patches, positions = patch_positions(rows)
    order = torch.randperm(patches.shape[1])
    pad = max_patches - patches.shape[1]
    return (
        torch.cat(
            [patches[:, order], torch.zeros(rows.shape[0], pad, PATCH_DIM)], dim=1
        ),
        torch.cat(
            [
                positions[:, order],
                torch.full((rows.shape[0], pad, 2), -1, dtype=torch.long),
            ],
            dim=1,
        ),
    )


class TestPatchPositions:
    def test_positions_are_row_major_x_then_y(self) -> None:
        rows = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(1, 2, 3, 2)

        patches, positions = patch_positions(rows)

        assert positions[0].tolist() == [
            [0, 0],
            [1, 0],
            [2, 0],
            [0, 1],
            [1, 1],
            [2, 1],
        ]
        assert torch.equal(patches[0, 4], rows[0, 1, 1])


class TestGemma4VisionRowsForward:
    def test_rows_match_padded_processor_layout(self, pristine_gemma4_model) -> None:
        # Arrange
        model = tiny_gemma4()
        image = [BOI, *([IMAGE] * TOKENS_PER_IMAGE), EOI]
        input_ids = torch.tensor([[3, *image, 5, *image, 6]])
        torch.manual_seed(1)
        rows = torch.rand(2, GRID, GRID, PATCH_DIM)
        pixel_values, image_position_ids = padded_inputs(rows, max_patches=24)
        with torch.no_grad():
            expected = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_position_ids=image_position_ids,
            ).logits
        install_gemma4_vision_patches(PatchRuntimeConfig())

        # Act
        with torch.no_grad():
            actual = model(input_ids=input_ids, pixel_values=rows).logits

        # Assert: the processor layout runs the tower over shuffled, padded
        # patches, so its fp32 sums add up in a different order.
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6)


class TestInstallGemma4VisionPatches:
    def test_install_is_idempotent(self, pristine_gemma4_model) -> None:
        install_gemma4_vision_patches(PatchRuntimeConfig())
        get_image_features = Gemma4Model.get_image_features

        install_gemma4_vision_patches(PatchRuntimeConfig())

        assert Gemma4Model.get_image_features is get_image_features

    def test_unimportable_modeling_module_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "agilerl.architectures.gemma4.GEMMA4_MODELING", "agilerl_missing_module"
        )

        with pytest.raises(RuntimeError, match="is not importable"):
            install_gemma4_vision_patches(PatchRuntimeConfig())


class TestGemma4VisionFillerFeatures:
    def test_runs_the_tower_on_one_image(self, pristine_gemma4_model) -> None:
        # Arrange
        model = tiny_gemma4()
        install_gemma4_vision_patches(PatchRuntimeConfig())
        tower_inputs: list[torch.Tensor] = []
        model.model.vision_tower.register_forward_hook(
            lambda _module, _args, kwargs, _output: tower_inputs.append(
                kwargs["pixel_values"]
            ),
            with_kwargs=True,
        )

        # Act
        with torch.no_grad():
            features = model.model.vision_filler_features(torch.device("cpu"))

        # Assert
        assert len(tower_inputs) == 1
        assert [tuple(image.shape) for image in features.pooler_output] == [(1, 32)]
