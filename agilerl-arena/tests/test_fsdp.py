# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``agilerl.arena.models.fsdp``."""

from __future__ import annotations

import copy
from dataclasses import fields, replace

import pytest
from pydantic import TypeAdapter, ValidationError

from agilerl.arena.models.algorithms.base import FSDP_JSON_SCHEMA
from agilerl.arena.models.fsdp import (
    PARALLEL_LAYOUT_FIELDS,
    FSDPConfig,
    parallel_layout_pins,
    record_parallel_pins,
)


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
            (
                "checkpoint_every_n_blocks",
                0,
                "FSDPConfig.checkpoint_every_n_blocks must be >= 1",
            ),
            ("ep", 0, "FSDPConfig.ep must be >= 1"),
            ("ep_token_blocks", 0, "FSDPConfig.ep_token_blocks must be >= 1"),
            (
                "routed_expert_chunk_mib",
                0,
                "FSDPConfig.routed_expert_chunk_mib must be >= 1",
            ),
            ("tp", 0, "FSDPConfig.tp must be >= 1"),
            ("cp", 0, "FSDPConfig.cp must be >= 1"),
            ("shard_group_size", 0, "FSDPConfig.shard_group_size must be >= 1"),
        ],
    )
    def test_rejects_out_of_range_values(
        self, field: str, value: int, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            FSDPConfig(**{field: value})

    def test_ep_defaults_to_one(self) -> None:
        assert FSDPConfig().ep == 1
        assert FSDPConfig(ep=4).ep == 4

    def test_ep_token_blocks_defaults_to_one(self) -> None:
        assert FSDPConfig().ep_token_blocks == 1
        assert FSDPConfig(ep=8, ep_token_blocks=4).ep_token_blocks == 4

    def test_memory_fields_default_to_unset(self) -> None:
        config = FSDPConfig()

        assert config.routed_expert_chunk_mib is None
        assert config.optim_cpu_offload is None
        assert config.cpu_offload is None
        assert config.checkpoint_every_n_blocks is None

    def test_memory_fields_keep_set_values(self) -> None:
        config = FSDPConfig(
            routed_expert_chunk_mib=1024, optim_cpu_offload=False, cpu_offload=False
        )

        assert config.routed_expert_chunk_mib == 1024
        assert config.optim_cpu_offload is False
        assert config.cpu_offload is False

    def test_tp_and_shard_group_default_to_unset(self) -> None:
        config = FSDPConfig()

        assert config.tp == 1
        assert config.shard_group_size is None

    def test_cp_and_style_defaults(self) -> None:
        config = FSDPConfig()

        assert config.cp == 1
        assert config.cp_style == "ulysses"

    def test_rejects_non_ulysses_style(self) -> None:
        with pytest.raises(ValueError, match="only 'ulysses' is supported"):
            FSDPConfig(cp_style="ring")

    def test_validation_rejects_non_ulysses_style(self) -> None:
        with pytest.raises(ValidationError, match="Input should be 'ulysses'"):
            TypeAdapter(FSDPConfig).validate_python({"cp_style": "ring"})

    def test_validation_rejects_pipeline_parallel_field(self) -> None:
        with pytest.raises(ValidationError, match="Unexpected keyword argument"):
            TypeAdapter(FSDPConfig).validate_python({"pp": 2})

    def test_accepts_shard_group_divisible_by_ep_and_tp(self) -> None:
        config = FSDPConfig(ep=8, tp=2, shard_group_size=8)

        assert config.shard_group_size == 8

    def test_accepts_shard_group_divisible_by_tp_times_cp(self) -> None:
        config = FSDPConfig(tp=2, cp=2, shard_group_size=8)

        assert config.shard_group_size == 8

    @pytest.mark.parametrize(
        ("size", "kwargs", "match"),
        [
            (8, {"ep": 3, "tp": 1}, r"shard_group_size=8 must be divisible by ep=3"),
            (
                8,
                {"ep": 1, "tp": 3},
                r"shard_group_size=8 must be divisible by tp \* cp=3",
            ),
            (8, {"cp": 3}, r"shard_group_size=8 must be divisible by tp \* cp=3"),
            (
                6,
                {"tp": 2, "cp": 2},
                r"shard_group_size=6 must be divisible by tp \* cp=4",
            ),
        ],
    )
    def test_rejects_shard_group_not_divisible(
        self, size: int, kwargs: dict[str, int], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            FSDPConfig(shard_group_size=size, **kwargs)


class TestFSDPConfigFrozenTpResidentProperty:
    @pytest.mark.parametrize(
        ("tp", "shard_frozen_tp", "resident"),
        [
            (2, False, True),
            (2, True, False),
            (2, None, False),
            (1, False, False),
        ],
    )
    def test_resident_only_when_tp_and_picked_unsharded(
        self, tp: int, shard_frozen_tp: bool | None, resident: bool
    ) -> None:
        config = FSDPConfig(tp=tp, shard_frozen_tp=shard_frozen_tp)

        assert config.frozen_tp_resident is resident


class TestParallelLayoutPins:
    def test_returns_the_recorded_fields(self) -> None:
        config = record_parallel_pins(FSDPConfig(tp=2), {"tp", "cpu_offload"})

        assert parallel_layout_pins(config) == frozenset({"tp"})

    def test_a_config_with_no_recorded_pins_pins_every_field(self) -> None:
        assert parallel_layout_pins(FSDPConfig()) == PARALLEL_LAYOUT_FIELDS

    def test_pins_survive_dataclass_replace(self) -> None:
        # Arrange
        config = record_parallel_pins(FSDPConfig(ep=2), {"ep"})

        # Act
        replaced = replace(config, reshard_after_forward=True)

        # Assert
        assert replaced.reshard_after_forward is True
        assert parallel_layout_pins(replaced) == frozenset({"ep"})

    def test_pins_survive_deepcopy(self) -> None:
        config = record_parallel_pins(FSDPConfig(), ())

        assert parallel_layout_pins(copy.deepcopy(config)) == frozenset()

    def test_pins_leave_equality_and_repr_alone(self) -> None:
        # Arrange
        config = FSDPConfig(tp=2)

        # Act
        recorded = record_parallel_pins(FSDPConfig(tp=2), ())

        # Assert
        assert recorded == config
        assert repr(recorded) == repr(config)
        assert "parallel_pins" not in repr(recorded)

    def test_pins_are_not_serialized(self) -> None:
        config = record_parallel_pins(FSDPConfig(tp=2), {"tp"})

        dumped = TypeAdapter(FSDPConfig).dump_python(config)

        assert "parallel_pins" not in dumped
        assert dumped["tp"] == 2


class TestFSDPJsonSchema:
    def test_every_field_has_a_description(self) -> None:
        # Arrange
        object_schema = FSDP_JSON_SCHEMA["anyOf"][2]

        # Act
        properties = object_schema["properties"]

        # Assert
        assert set(properties) == {field.name for field in fields(FSDPConfig)} - {
            "parallel_pins"
        }
        assert all(prop.get("description") for prop in properties.values())
        assert object_schema["additionalProperties"] is False

    def test_carries_integer_bounds(self) -> None:
        properties = FSDP_JSON_SCHEMA["anyOf"][2]["properties"]

        assert properties["ep"]["minimum"] == 1
        assert properties["ep_token_blocks"]["minimum"] == 1
        assert properties["tp"]["minimum"] == 1
        assert properties["checkpoint_every_n_blocks"]["anyOf"][0]["minimum"] == 1
