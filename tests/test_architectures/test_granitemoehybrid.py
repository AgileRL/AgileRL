# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Granite 4 hybrid vLLM config override."""

from __future__ import annotations

from transformers import GraniteMoeHybridConfig

from agilerl.architectures.catalog import (
    FAMILY_RUNTIME_CONFIGS,
    granitemoehybrid_vllm_hf_override,
)


class TestGranitemoehybridVllmHfOverride:
    def test_renames_layer_types_to_vllm_names(self) -> None:
        # Arrange
        config = GraniteMoeHybridConfig(
            num_hidden_layers=3,
            layer_types=["linear_attention", "full_attention", "linear_attention"],
        )

        # Act
        overridden = granitemoehybrid_vllm_hf_override(config)

        # Assert
        assert overridden is config
        assert config.layer_types == ["mamba", "attention", "mamba"]

    def test_keeps_names_vllm_already_reads(self) -> None:
        config = GraniteMoeHybridConfig(
            num_hidden_layers=2, layer_types=["full_attention", "full_attention"]
        )
        granitemoehybrid_vllm_hf_override(config)

        overridden = granitemoehybrid_vllm_hf_override(config)

        assert overridden.layer_types == ["attention", "attention"]

    def test_leaves_a_config_without_layer_types_untouched(self) -> None:
        # Arrange
        config = GraniteMoeHybridConfig(num_hidden_layers=2)
        del vars(config)["layer_types"]

        # Act
        overridden = granitemoehybrid_vllm_hf_override(config)

        # Assert
        assert overridden is config
        assert "layer_types" not in vars(config)


class TestGranitemoehybridRuntimeConfig:
    def test_engine_kwargs_carry_the_override(self) -> None:
        vllm_kwargs = FAMILY_RUNTIME_CONFIGS["granitemoehybrid"].vllm.model_dump(
            exclude_none=True
        )

        assert vllm_kwargs == {"hf_overrides": granitemoehybrid_vllm_hf_override}
