# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for image observation encoding."""

from __future__ import annotations

import pytest
import torch

from agilerl.llm_envs.observation import encode_image_training_inputs, patch_grid_rows


class TestEncodeImageTrainingInputs:
    def test_keeps_per_image_pixels_without_a_patch_grid(self) -> None:
        pixel_values = torch.randn(2, 3, 4, 4)

        def processor(**kwargs: object) -> dict[str, torch.Tensor]:
            return {"input_ids": torch.tensor([[1, 2]]), "pixel_values": pixel_values}

        input_ids, encoded = encode_image_training_inputs(
            text="<image>", image=[object(), object()], processor=processor
        )

        assert input_ids.tolist() == [[1, 2]]
        assert encoded is pixel_values

    def test_folds_packed_patches_into_one_row_per_image(self) -> None:
        # Arrange
        packed = torch.arange(2 * 2 * 3 * 5, dtype=torch.float32).reshape(12, 5)

        def processor(**kwargs: object) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "pixel_values": packed,
                "image_grid_hw": torch.tensor([[2, 3], [2, 3]]),
            }

        # Act
        _, rows = encode_image_training_inputs(
            text="<image><image>", image=[object(), object()], processor=processor
        )

        # Assert
        assert rows.shape == (2, 2, 3, 5)
        assert torch.equal(rows[1, 0, 0], packed[6])
        assert torch.equal(rows.reshape(-1, 5), packed)


class TestPatchGridRows:
    def test_images_with_different_grids_raise(self) -> None:
        packed = torch.zeros(4 + 6, 5)

        with pytest.raises(ValueError, match=r"patch grids \[\(2, 2\), \(2, 3\)\]"):
            patch_grid_rows(packed, torch.tensor([[2, 2], [2, 3]]))
