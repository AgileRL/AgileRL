# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from agilerl.architectures.vision_rows import (
    ACCEPTS_VISION_ROWS,
    SHARDS_IMAGE_FEATURES,
    install_vision_rows,
    packed_patches,
    shard_image_features,
)
from agilerl.distributed.context_parallel import (
    IMAGE_TOKEN_SHARD,
    ULYSSES_PARAMS,
    ImageTokenShard,
    clear_ulysses_params,
)
from tests.parallel_helpers import publish_varlen_offsets

IMAGE = 9


def doubling(method: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(self: object, value: int) -> int:
        return 2 * method(self, value)

    return wrapped


def filler(_self: object, _device: torch.device) -> str:
    return "filler"


def image_token_id(model: Any) -> int:
    return model.image_token_id


class Features:
    """Records the Ulysses params each call sees and returns one feature row per image token."""

    image_token_id = IMAGE

    def __init__(self) -> None:
        self.seen: list[dict[str, object]] = []

    def tensor(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.seen.append(dict(ULYSSES_PARAMS))
        return pixel_values

    def pooled(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        self.seen.append(dict(ULYSSES_PARAMS))
        return SimpleNamespace(pooler_output=pixel_values.split([2, 2]))


def image_shard(rank: int) -> ImageTokenShard:
    input_ids = torch.tensor([[1, 1, 1, IMAGE, IMAGE, 2, IMAGE, IMAGE]])
    return ImageTokenShard(input_ids, torch.zeros_like(input_ids), rank, 2)


class TestPackedPatches:
    def test_unfolds_rows_into_patches_and_grids(self) -> None:
        # Arrange
        rows = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).reshape(2, 2, 3, 4)

        # Act
        patches, image_grid_hw = packed_patches(rows)

        # Assert
        assert torch.equal(patches, rows.reshape(12, 4))
        assert image_grid_hw.tolist() == [[2, 3], [2, 3]]


class TestShardImageFeatures:
    def test_passes_features_through_outside_a_cp_forward(self) -> None:
        features = Features()
        pixel_values = torch.arange(4.0)[:, None]

        output = shard_image_features(image_token_id)(Features.tensor)(
            features, pixel_values
        )

        assert output is pixel_values

    @pytest.mark.parametrize(("rank", "rows"), [(0, [0]), (1, [1, 2, 3])])
    def test_tensor_output_keeps_this_shards_rows(
        self, rank: int, rows: list[int]
    ) -> None:
        # Arrange
        features = Features()
        pixel_values = torch.arange(4.0)[:, None]

        # Act
        token = IMAGE_TOKEN_SHARD.set(image_shard(rank))
        try:
            output = shard_image_features(image_token_id)(Features.tensor)(
                features, pixel_values
            )
        finally:
            IMAGE_TOKEN_SHARD.reset(token)

        # Assert
        assert output.flatten().tolist() == rows

    def test_pooled_output_becomes_one_element_of_this_shards_rows(self) -> None:
        # Arrange
        features = Features()
        pixel_values = torch.arange(4.0)[:, None]

        # Act
        token = IMAGE_TOKEN_SHARD.set(image_shard(1))
        try:
            output = shard_image_features(image_token_id)(Features.pooled)(
                features, pixel_values
            )
        finally:
            IMAGE_TOKEN_SHARD.reset(token)

        # Assert
        assert len(output.pooler_output) == 1
        assert output.pooler_output[0].flatten().tolist() == [1, 2, 3]

    def test_tower_runs_with_ulysses_off_and_restores_the_params(self) -> None:
        # Arrange
        features = Features()
        publish_varlen_offsets(torch.tensor([0, 8], dtype=torch.int32))
        published = dict(ULYSSES_PARAMS)

        # Act
        token = IMAGE_TOKEN_SHARD.set(image_shard(0))
        try:
            shard_image_features(image_token_id)(Features.tensor)(
                features, torch.zeros(4, 1)
            )
            after = dict(ULYSSES_PARAMS)
        finally:
            IMAGE_TOKEN_SHARD.reset(token)
            clear_ulysses_params()

        # Assert
        assert features.seen == [{}]
        assert after == published

    def test_marks_the_wrapped_method(self) -> None:
        wrapped = shard_image_features(image_token_id)(Features.tensor)

        assert getattr(wrapped, SHARDS_IMAGE_FEATURES) is True


class TestInstallVisionRows:
    @staticmethod
    def _model_class() -> type:
        class Model:
            image_token_id = IMAGE

            def forward(self, value: int) -> int:
                return value + 1

            def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
                return pixel_values

        return Model

    def test_wraps_the_method_and_sets_the_filler(self) -> None:
        # Arrange
        model_cls = self._model_class()

        # Act
        install_vision_rows(model_cls, "forward", doubling, filler, image_token_id)

        # Assert
        model: Any = model_cls()
        assert model.forward(1) == 4
        assert model.vision_filler_features(torch.device("cpu")) == "filler"
        assert vars(model_cls)[ACCEPTS_VISION_ROWS] is True

    def test_shards_get_image_features(self) -> None:
        # Arrange
        model_cls = self._model_class()
        install_vision_rows(model_cls, "forward", doubling, filler, image_token_id)
        model: Any = model_cls()

        # Act
        token = IMAGE_TOKEN_SHARD.set(image_shard(0))
        try:
            output = model.get_image_features(torch.arange(4.0)[:, None])
        finally:
            IMAGE_TOKEN_SHARD.reset(token)

        # Assert
        assert output.flatten().tolist() == [0]

    def test_wraps_get_image_features_when_it_takes_the_rows(self) -> None:
        # Arrange
        model_cls = self._model_class()

        # Act
        install_vision_rows(
            model_cls, "get_image_features", doubling, filler, image_token_id
        )

        # Assert
        model: Any = model_cls()
        assert model.get_image_features(3) == 6
        assert getattr(model_cls.get_image_features, SHARDS_IMAGE_FEATURES) is True

    def test_second_install_keeps_the_first_wrap(self) -> None:
        # Arrange
        model_cls = self._model_class()
        install_vision_rows(model_cls, "forward", doubling, filler, image_token_id)
        get_image_features = model_cls.get_image_features

        # Act
        install_vision_rows(model_cls, "forward", doubling, filler, image_token_id)

        # Assert
        assert model_cls().forward(3) == 8
        assert model_cls.get_image_features is get_image_features

    def test_subclass_of_a_patched_class_is_patched_again(self) -> None:
        # Arrange
        base = self._model_class()

        class Child(base):
            def forward(self, value: int) -> int:
                return value + 10

            def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
                return pixel_values

        install_vision_rows(base, "forward", doubling, filler, image_token_id)

        # Act
        install_vision_rows(Child, "forward", doubling, filler, image_token_id)

        # Assert
        assert Child().forward(1) == 22
