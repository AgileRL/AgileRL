# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``agilerl.arena.models.fsdp``."""

from __future__ import annotations

from dataclasses import fields

import pytest

from agilerl.arena.models.algorithms.base import FSDP_JSON_SCHEMA
from agilerl.arena.models.fsdp import FSDPConfig


class TestFSDPConfig:
    def test_defaults_use_bf16_params_and_fp32_reduce(self) -> None:
        config = FSDPConfig()

        assert config.param_dtype == "bfloat16"
        assert config.reduce_dtype == "float32"

    @pytest.mark.parametrize("value", ["float16", "torch.float16"])
    def test_normalizes_dtype_names(self, value: str) -> None:
        assert FSDPConfig(param_dtype=value).param_dtype == "float16"

    def test_stores_torch_dtype_like_value_by_name(self) -> None:
        class FakeDtype:
            def __str__(self) -> str:
                return "torch.bfloat16"

        assert FSDPConfig(reduce_dtype=FakeDtype()).reduce_dtype == "bfloat16"

    def test_rejects_unknown_dtype(self) -> None:
        with pytest.raises(ValueError, match="Unknown FSDP dtype 'int8'"):
            FSDPConfig(param_dtype="int8")

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("prefetch_units", 0, "prefetch_units must be >= 1"),
            ("wrap_every_n_blocks", 0, "wrap_every_n_blocks must be >= 1"),
            (
                "param_persistence_threshold",
                -1,
                "param_persistence_threshold must be >= 0",
            ),
        ],
    )
    def test_rejects_out_of_range_values(
        self, field: str, value: int, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            FSDPConfig(**{field: value})


class TestFSDPJsonSchema:
    def test_every_field_has_a_description(self) -> None:
        # Arrange
        object_schema = FSDP_JSON_SCHEMA["anyOf"][2]

        # Act
        properties = object_schema["properties"]

        # Assert
        assert set(properties) == {field.name for field in fields(FSDPConfig)}
        assert all(prop.get("description") for prop in properties.values())
        assert object_schema["additionalProperties"] is False

    def test_carries_integer_bounds(self) -> None:
        properties = FSDP_JSON_SCHEMA["anyOf"][2]["properties"]

        assert properties["prefetch_units"]["minimum"] == 1
        assert properties["wrap_every_n_blocks"]["minimum"] == 1
        assert properties["param_persistence_threshold"]["minimum"] == 0
