# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 VL forward over one patch-grid row per image matches the native flat-patch forward."""

import logging

import pytest
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

from agilerl.architectures.qwen3_5.vision import (
    install_qwen3_5_vl_patches,
    mrope_position_ids,
    patch_qwen_vision_rows,
)
from agilerl.architectures.runtime import PatchRuntimeConfig
from tests.test_architectures.qwen3_5.vision_helpers import (
    GRID,
    IMAGE,
    PATCH_DIM,
    flat_inputs,
    image_prompt,
    restored_qwen_vl_model,
    tiny_qwen_vl,
)


@pytest.fixture
def pristine_qwen_vl_model():
    """Restore ``Qwen3_5Model`` and the Qwen GDN classes after a class-level patch."""
    with restored_qwen_vl_model():
        yield


@pytest.mark.usefixtures("torch_kernels")
class TestQwenVisionRowsForward:
    def test_rows_match_native_flat_patches(self, pristine_qwen_vl_model) -> None:
        # Arrange
        model = tiny_qwen_vl()
        input_ids = torch.tensor([image_prompt([3, 4, 5], images=2)])
        torch.manual_seed(1)
        rows = torch.randn(2, GRID, GRID, PATCH_DIM)
        flat, grids = flat_inputs(rows)
        with torch.no_grad():
            expected = model(
                input_ids=input_ids,
                pixel_values=flat,
                image_grid_thw=grids,
                mm_token_type_ids=(input_ids == IMAGE).int(),
            ).logits
        install_qwen3_5_vl_patches(PatchRuntimeConfig())

        # Act
        with torch.no_grad():
            unpositioned = model(input_ids=input_ids, pixel_values=rows).logits
            positioned = model(
                input_ids=input_ids,
                pixel_values=rows,
                position_ids=torch.arange(input_ids.shape[1])[None],
            ).logits

        # Assert
        assert torch.equal(unpositioned, expected)
        assert torch.equal(positioned, expected)

    def test_rows_without_input_ids_raise(self, pristine_qwen_vl_model) -> None:
        model = tiny_qwen_vl()
        install_qwen3_5_vl_patches(PatchRuntimeConfig())
        rows = torch.randn(1, GRID, GRID, PATCH_DIM)

        with pytest.raises(ValueError, match="need input_ids"):
            model.model(inputs_embeds=torch.randn(1, 8, 32), pixel_values=rows)

    def test_text_forward_is_unchanged(self, pristine_qwen_vl_model) -> None:
        model = tiny_qwen_vl()
        input_ids = torch.tensor([[3, 4, 5, 6]])
        with torch.no_grad():
            expected = model(input_ids=input_ids).logits
        install_qwen3_5_vl_patches(PatchRuntimeConfig())

        with torch.no_grad():
            actual = model(input_ids=input_ids).logits

        assert torch.equal(actual, expected)

    def test_multi_row_packed_images_match_each_document_alone(
        self, pristine_qwen_vl_model
    ) -> None:
        # Arrange
        model = tiny_qwen_vl()
        install_qwen3_5_vl_patches(PatchRuntimeConfig())
        rows = [
            [image_prompt([3, 4], images=1), [5, 6, 7, 8, 9]],
            [[10, 11, 12, 13, 14], image_prompt([15, 16], images=1)],
        ]
        torch.manual_seed(1)
        images = {(0, 0): torch.randn(1, GRID, GRID, PATCH_DIM)}
        images[1, 1] = torch.randn(1, GRID, GRID, PATCH_DIM)
        packed_ids = torch.tensor(
            [[token for doc in row for token in doc] for row in rows]
        )
        packed_positions = torch.stack(
            [torch.cat([torch.arange(len(doc)) for doc in row]) for row in rows]
        )

        # Act
        with torch.no_grad():
            packed = model(
                input_ids=packed_ids,
                position_ids=packed_positions,
                pixel_values=torch.cat([images[0, 0], images[1, 1]]),
                use_cache=False,
            ).logits
            alone = [
                torch.cat(
                    [
                        model(
                            input_ids=torch.tensor([doc]),
                            position_ids=torch.arange(len(doc))[None],
                            pixel_values=images.get((row_index, doc_index)),
                            use_cache=False,
                        ).logits[0]
                        for doc_index, doc in enumerate(row)
                    ]
                )
                for row_index, row in enumerate(rows)
            ]

        # Assert: packed rows and lone documents run fp32 GEMMs of different
        # shapes, which round differently in the last bits.
        torch.testing.assert_close(packed, torch.stack(alone), rtol=0, atol=1e-6)


class TestQwenVisionFillerFeatures:
    def test_runs_the_tower_on_one_image(self, pristine_qwen_vl_model) -> None:
        # Arrange
        model = tiny_qwen_vl()
        install_qwen3_5_vl_patches(PatchRuntimeConfig())
        tower_inputs: list[torch.Tensor] = []
        model.model.visual.register_forward_hook(
            lambda _module, args, _output: tower_inputs.append(args[0])
        )

        # Act
        with torch.no_grad():
            features = model.model.vision_filler_features(torch.device("cpu"))

        # Assert
        assert len(tower_inputs) == 1
        assert [tuple(image.shape) for image in features.pooler_output] == [(1, 32)]


class TestMropePositionIds:
    def test_packed_documents_get_their_own_positions(self) -> None:
        # Arrange
        model = tiny_qwen_vl().model
        first = image_prompt([3, 4], images=2)
        second = image_prompt([5], images=1)
        text_only = [8, 9, 10]
        docs = [first, text_only, second]
        packed = torch.tensor([[token for doc in docs for token in doc]])
        positions = torch.cat([torch.arange(len(doc)) for doc in docs])[None]
        grids = torch.tensor([[1, GRID, GRID]] * 3)

        # Act
        actual = mrope_position_ids(model, packed, positions, grids)

        # Assert
        expected_vision = []
        for doc, doc_grids in (
            (first, grids[:2]),
            (text_only, None),
            (second, grids[2:]),
        ):
            ids = torch.tensor([doc])
            if doc_grids is None:
                expected_vision.append(torch.arange(len(doc)).expand(3, -1))
            else:
                doc_positions, _ = model.get_rope_index(
                    ids, (ids == IMAGE).int(), image_grid_thw=doc_grids
                )
                expected_vision.append(doc_positions[:, 0])
        assert actual.shape == (4, 1, packed.shape[1])
        assert torch.equal(actual[0], positions)
        assert torch.equal(actual[1:, 0], torch.cat(expected_vision, dim=1))

    def test_image_positions_differ_from_text_positions(self) -> None:
        model = tiny_qwen_vl().model
        ids = torch.tensor([image_prompt([3], images=1)])
        positions = torch.arange(ids.shape[1])[None]

        actual = mrope_position_ids(
            model, ids, positions, torch.tensor([[1, GRID, GRID]])
        )

        image = ids[0] == IMAGE
        assert not torch.equal(actual[2, 0, image], positions[0, image])


class TestPatchQwenVisionRows:
    def test_absent_model_module_warns_and_leaves_the_model_unpatched(
        self, caplog
    ) -> None:
        forward = Qwen3_5Model.__dict__["forward"]

        with caplog.at_level(logging.WARNING):
            patch_qwen_vision_rows("agilerl_missing_module.Qwen3_5Model")

        assert "vision rows are not supported" in caplog.text
        assert Qwen3_5Model.__dict__["forward"] is forward
