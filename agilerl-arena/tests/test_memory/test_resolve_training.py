# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Trainer-side estimator settings derive from plain run values."""

from typing import Any

import pytest

from agilerl.arena.memory.resolve.training import (
    liger_kernel_arch,
    lora_target_scope,
    packed_moe_dispatch,
    sampling_support_width,
)
from agilerl.arena.memory.specs import ModelArch


class TestLoraTargetScope:
    @pytest.mark.parametrize(
        ("targets", "scope"),
        [
            (["q_proj", "v_proj"], "attention-only"),
            ({"query", "key", "value"}, "attention-only"),
            (["q_proj", "up_proj"], "all-linear"),
            ("all-linear", "all-linear"),
            ([], "all-linear"),
            (None, "all-linear"),
        ],
    )
    def test_target_modules_set_the_lora_scope(self, targets: Any, scope: str) -> None:
        assert lora_target_scope(targets) == scope


class TestPackedMoeDispatch:
    def test_split_expert_lora_is_contracted(self) -> None:
        assert packed_moe_dispatch(["experts.up_proj"], False) == "contracted"

    @pytest.mark.parametrize(
        ("target_parameters", "materialized"),
        [(["experts.up_proj"], True), (None, False), ([], False)],
    )
    def test_dispatch_is_materialized_without_split_expert_lora(
        self, target_parameters: list[str] | None, materialized: bool
    ) -> None:
        assert packed_moe_dispatch(target_parameters, materialized) == "materialized"


class TestSamplingSupportWidth:
    @pytest.mark.parametrize(
        ("records", "top_k", "width"), [(True, 20, 20), (True, 1, 2), (False, 20, 0)]
    )
    def test_recorded_supports_set_the_support_width(
        self, records: bool, top_k: int, width: int
    ) -> None:
        assert sampling_support_width(records, top_k) == width


class TestLigerKernelArch:
    def test_liger_layers_keep_the_checkpoint_geometry(self) -> None:
        arch = ModelArch(
            n_layers=2,
            hidden_size=64,
            intermediate_size=128,
            n_heads=4,
            n_kv_heads=4,
            head_dim=16,
            vocab_size=1000,
        )

        assert liger_kernel_arch(arch, True) is arch

    def test_stock_layers_keep_fp32_norm_inputs_and_the_activated_gate(self) -> None:
        arch = ModelArch(
            n_layers=2,
            hidden_size=64,
            intermediate_size=128,
            n_heads=4,
            n_kv_heads=4,
            head_dim=16,
            vocab_size=1000,
        )

        stock = liger_kernel_arch(arch, False)

        assert stock.fp32_norm_inputs is True
        assert stock.stock_gated_mlp is True
