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
            ("backward_prefetch_units", 0, "backward_prefetch_units must be >= 1"),
            ("checkpoint_every_n_blocks", 0, "checkpoint_every_n_blocks must be >= 1"),
            ("wrap_every_n_blocks", 0, "wrap_every_n_blocks must be >= 1"),
            (
                "param_persistence_threshold",
                -1,
                "param_persistence_threshold must be >= 0",
            ),
            ("ep", 0, "FSDPConfig.ep must be >= 1"),
            ("ep_token_blocks", 0, "FSDPConfig.ep_token_blocks must be >= 1"),
            (
                "routed_expert_chunk_mib",
                0,
                "FSDPConfig.routed_expert_chunk_mib must be >= 1",
            ),
            ("tp", 0, "FSDPConfig.tp must be >= 1"),
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

    def test_memory_fields_keep_set_values(self) -> None:
        config = FSDPConfig(routed_expert_chunk_mib=1024, optim_cpu_offload=False)

        assert config.routed_expert_chunk_mib == 1024
        assert config.optim_cpu_offload is False

    def test_tp_and_shard_group_default_to_unset(self) -> None:
        config = FSDPConfig()

        assert config.tp == 1
        assert config.shard_group_size is None

    def test_accepts_shard_group_divisible_by_ep_and_tp(self) -> None:
        config = FSDPConfig(ep=8, tp=2, shard_group_size=8)

        assert config.shard_group_size == 8

    @pytest.mark.parametrize(
        ("ep", "tp", "match"),
        [
            (3, 1, "shard_group_size=8 must be divisible by ep=3"),
            (1, 3, "shard_group_size=8 must be divisible by tp=3"),
        ],
    )
    def test_rejects_shard_group_not_divisible(
        self, ep: int, tp: int, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            FSDPConfig(ep=ep, tp=tp, shard_group_size=8)

    def test_prefetch_and_checkpoint_policy_defaults(self) -> None:
        config = FSDPConfig()

        assert config.prefetch_units == 1
        assert config.backward_prefetch_units == 1
        assert config.checkpoint_skip_layer_types == ()
        assert config.checkpoint_every_n_blocks == 1

    def test_stores_checkpoint_skip_list_as_tuple(self) -> None:
        config = FSDPConfig(checkpoint_skip_layer_types=["attention", "mlp"])

        assert config.checkpoint_skip_layer_types == ("attention", "mlp")


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
        assert properties["backward_prefetch_units"]["minimum"] == 1
        assert properties["checkpoint_every_n_blocks"]["minimum"] == 1
        assert properties["param_persistence_threshold"]["minimum"] == 0
