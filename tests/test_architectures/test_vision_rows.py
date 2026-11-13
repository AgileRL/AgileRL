# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from agilerl.architectures.vision_rows import (
    ACCEPTS_VISION_ROWS,
    install_vision_rows,
    packed_patches,
)


def doubling(method: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(self: object, value: int) -> int:
        return 2 * method(self, value)

    return wrapped


def filler(_self: object, _device: torch.device) -> str:
    return "filler"


class TestPackedPatches:
    def test_unfolds_rows_into_patches_and_grids(self) -> None:
        # Arrange
        rows = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).reshape(2, 2, 3, 4)

        # Act
        patches, image_grid_hw = packed_patches(rows)

        # Assert
        assert torch.equal(patches, rows.reshape(12, 4))
        assert image_grid_hw.tolist() == [[2, 3], [2, 3]]


class TestInstallVisionRows:
    def test_wraps_the_method_and_sets_the_filler(self) -> None:
        # Arrange
        class Model:
            def forward(self, value: int) -> int:
                return value + 1

        # Act
        install_vision_rows(Model, "forward", doubling, filler)

        # Assert
        model: Any = Model()
        assert model.forward(1) == 4
        assert model.vision_filler_features(torch.device("cpu")) == "filler"
        assert vars(Model)[ACCEPTS_VISION_ROWS] is True

    def test_second_install_keeps_the_first_wrap(self) -> None:
        # Arrange
        class Model:
            def forward(self, value: int) -> int:
                return value

        install_vision_rows(Model, "forward", doubling, filler)

        # Act
        install_vision_rows(Model, "forward", doubling, filler)

        # Assert
        assert Model().forward(3) == 6

    def test_subclass_of_a_patched_class_is_patched_again(self) -> None:
        # Arrange
        class Base:
            def forward(self, value: int) -> int:
                return value

        class Child(Base):
            def forward(self, value: int) -> int:
                return value + 10

        install_vision_rows(Base, "forward", doubling, filler)

        # Act
        install_vision_rows(Child, "forward", doubling, filler)

        # Assert
        assert Child().forward(1) == 22
