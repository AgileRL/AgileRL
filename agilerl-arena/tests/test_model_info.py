# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-model info, LoRA validation, and bundled configs."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agilerl.arena.models.model_info import (
    ARCH_DENSE,
    ARCH_HYBRID,
    ARCH_HYBRID_MOE,
    ARCH_MOE,
    SUPPORTED_MODEL_INFO,
    ModelInfo,
    inspected_path,
)
from agilerl.arena.models.networks import FinetuningNetworkSpec


def finetuning_network(
    hub_id: str,
    target_modules: object = ("q_proj",),
    target_parameters: object = None,
    lora_r: int = 16,
) -> FinetuningNetworkSpec:
    """Finetuning section for a Hub id with explicit LoRA targets."""
    lora: dict[str, object] = {"target_modules": target_modules, "lora_r": lora_r}
    if target_parameters is not None:
        lora["target_parameters"] = target_parameters
    return FinetuningNetworkSpec.model_validate(
        {
            "pretrained_model_name_or_path": hub_id,
            "max_context_length": 512,
            "lora_config": lora,
        }
    )


class TestSupportedModelInfo:
    def test_nemotron_nano_sizing_matches_catalog(self) -> None:
        entry = SUPPORTED_MODEL_INFO["nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"]

        assert entry.model_type == "nemotron_h"
        assert entry.num_params == 3973556832
        assert entry.max_context_length == 262144
        assert entry.lora_ranks == (1, 8, 16, 32, 64, 128, 256, 320, 512)
        assert (entry.hidden_dim, entry.vocab_size, entry.num_hidden_layers) == (
            3136,
            131072,
            42,
        )
        assert "conv1d" not in entry.modules

    def test_nemotron_excludes_gate_proj(self) -> None:
        entry = SUPPORTED_MODEL_INFO["nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"]

        assert "gate_proj" not in entry.modules
        assert {"up_proj", "in_proj"} <= entry.modules

    def test_super_vl_lists_vision_and_experts(self) -> None:
        entry = SUPPORTED_MODEL_INFO[
            "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
        ]

        assert {"query", "fc1", "linear1"} <= entry.modules
        assert "mixer.experts.up_proj" in entry.parameters
        assert entry.num_params is None
        assert entry.lora_ranks is None

    def test_granite_micro_has_no_mamba_or_experts(self) -> None:
        entry = SUPPORTED_MODEL_INFO["ibm-granite/granite-4.0-micro"]

        assert {"input_linear", "output_linear"} <= entry.modules
        assert "in_proj" not in entry.modules
        assert entry.parameters == frozenset()

    def test_gemma4_lists_wrapped_and_bare_projections(self) -> None:
        entry = SUPPORTED_MODEL_INFO["google/gemma-4-E4B-it"]

        assert {"linear", "output_proj"} <= entry.modules
        assert {"q_proj", "gate_proj"} <= entry.modules


class TestModelInfoConfig:
    @pytest.mark.parametrize("hub_id", sorted(SUPPORTED_MODEL_INFO))
    def test_every_id_bundles_its_config(self, hub_id: str) -> None:
        entry = SUPPORTED_MODEL_INFO[hub_id]

        assert entry.hub_id == hub_id
        assert entry.config["model_type"] == entry.model_type
        assert entry.max_context_length > 0
        assert entry.num_hidden_layers > 0

    def test_multimodal_ids_read_text_config(self) -> None:
        gemma = SUPPORTED_MODEL_INFO["google/gemma-4-E4B-it"]
        super_vl = SUPPORTED_MODEL_INFO[
            "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
        ]

        assert (gemma.hidden_dim, gemma.vocab_size, gemma.num_hidden_layers) == (
            2560,
            262144,
            42,
        )
        assert gemma.max_context_length == 131072
        assert super_vl.num_hidden_layers == 88
        assert super_vl.max_context_length == 1048576

    def test_dump_includes_derived_fields(self) -> None:
        dumped = SUPPORTED_MODEL_INFO["Qwen/Qwen3-4B"].model_dump()

        assert dumped["model_type"] == "qwen3"
        assert dumped["max_context_length"] == 40960
        assert "config" not in dumped


class TestModelInfoInspected:
    def test_inspected_path_reads_bundled_json(self) -> None:
        entry = SUPPORTED_MODEL_INFO["Qwen/Qwen3-4B"]

        payload = json.loads(inspected_path(entry.hub_id).read_text(encoding="utf-8"))

        assert payload == entry.inspected

    def test_targets_and_ranks_come_from_lora_info(self) -> None:
        entry = SUPPORTED_MODEL_INFO["openai/gpt-oss-20b"]

        assert entry.modules == frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
        assert entry.parameters == frozenset(
            {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
        )
        assert entry.num_params == 20914757184
        assert entry.lora_ranks == (1, 8, 16, 32, 64, 128)

    def test_lora_info_carries_dims_and_gram_sidecar(self) -> None:
        lora_info = SUPPORTED_MODEL_INFO["Qwen/Qwen3-4B"].lora_info

        assert lora_info["_gram_estimate"] == {
            "hidden_dim": 2560,
            "vocab_size": 151936,
            "num_hidden_layers": 36,
        }
        assert lora_info["k_proj"] == [
            {
                "kind": "module",
                "scope": "model.layers.self_attn",
                "type": "Linear",
                "in_features": 2560,
                "out_features": 1024,
                "count": 36,
            }
        ]

    def test_entry_without_dims_has_no_ranks(self) -> None:
        entry = SUPPORTED_MODEL_INFO[
            "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
        ]

        assert entry.lora_ranks is None
        assert entry.num_params is None
        assert "mixer.experts.up_proj" in entry.parameters

    def test_dump_sorts_target_names(self) -> None:
        dumped = SUPPORTED_MODEL_INFO["openai/gpt-oss-20b"].model_dump(mode="json")

        assert dumped["modules"] == ["k_proj", "o_proj", "q_proj", "v_proj"]
        assert dumped["parameters"] == [
            "mlp.experts.down_proj",
            "mlp.experts.gate_up_proj",
        ]


class TestModelInfoArchitecture:
    def test_listed_architectures(self) -> None:
        architectures = {
            hub_id: entry.architecture for hub_id, entry in SUPPORTED_MODEL_INFO.items()
        }

        assert architectures["Qwen/Qwen3-4B"] == ARCH_DENSE
        assert architectures["google/gemma-4-E4B-it"] == ARCH_DENSE
        assert architectures["ibm-granite/granite-4.0-h-tiny"] == ARCH_HYBRID
        assert architectures["ibm-granite/granite-3.1-3b-a800m-instruct"] == ARCH_MOE
        assert architectures["openai/gpt-oss-20b"] == ARCH_MOE
        assert (
            architectures["nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"]
            == ARCH_HYBRID_MOE
        )

    def test_rejects_unknown_architecture(self) -> None:
        with pytest.raises(ValidationError, match="architecture"):
            ModelInfo.model_validate({"hub_id": "org/x", "architecture": "sparse"})


class TestFinetuningNetworkSpecLoraTargets:
    def test_valid_targets_pass(self) -> None:
        spec = finetuning_network(
            "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
            ("q_proj", "in_proj"),
        )

        assert spec.lora_config is not None

    def test_unknown_module_raises_with_valid_options(self) -> None:
        with pytest.raises(ValidationError, match="q_projj"):
            finetuning_network("Qwen/Qwen3-4B", ("q_proj", "q_projj"))

    def test_expert_parameters_on_dense_family_raise(self) -> None:
        with pytest.raises(ValidationError, match="target_parameters"):
            finetuning_network(
                "Qwen/Qwen3-4B",
                ("q_proj",),
                ("mixer.experts.up_proj",),
            )

    def test_valid_expert_parameters_pass(self) -> None:
        spec = finetuning_network(
            "openai/gpt-oss-20b",
            ("q_proj",),
            ("mlp.experts.gate_up_proj",),
        )

        assert spec.lora_config is not None

    def test_unlisted_id_skips_validation(self) -> None:
        spec = finetuning_network("org/custom-model", ("anything_at_all",))

        assert spec.lora_config is not None

    def test_listed_rank_passes(self) -> None:
        spec = finetuning_network("Qwen/Qwen2.5-0.5B-Instruct", lora_r=128)

        assert spec.lora_config is not None
        assert spec.lora_config.lora_r == 128

    def test_rank_below_cap_passes(self) -> None:
        spec = finetuning_network("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", lora_r=2)

        assert spec.lora_config is not None

    def test_rank_above_model_cap_raises(self) -> None:
        with pytest.raises(
            ValidationError, match=r"LoRA rank 256 .*Qwen2.5-0.5B-Instruct \(128\)"
        ):
            finetuning_network("Qwen/Qwen2.5-0.5B-Instruct", lora_r=256)

    def test_unverified_ranks_skip_rank_check(self) -> None:
        spec = finetuning_network(
            "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16", lora_r=1024
        )

        assert spec.lora_config is not None

    def test_all_linear_in_a_list_is_a_module_name(self) -> None:
        with pytest.raises(ValidationError, match="all-linear"):
            finetuning_network("Qwen/Qwen3-4B", ["all-linear"])

    def test_all_linear_string_passes(self) -> None:
        spec = finetuning_network("Qwen/Qwen3-4B", "all-linear")

        assert spec.lora_config is not None

    def test_regex_string_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be 'all-linear' or a list"):
            finetuning_network("Qwen/Qwen3-4B", r".*\.language_model.*\.(q_proj)$")

    def test_dotted_target_raises(self) -> None:
        with pytest.raises(ValidationError, match=r"\['q_proj.linear'\]"):
            finetuning_network("google/gemma-4-E4B-it", ("q_proj.linear",))

    def test_bare_projection_on_wrapped_family_passes(self) -> None:
        spec = finetuning_network("google/gemma-4-E4B-it", ("q_proj",))

        assert spec.lora_config is not None


class TestFinetuningNetworkSpecExpertDefault:
    def test_unset_parameters_adapt_every_expert_path(self) -> None:
        spec = finetuning_network("openai/gpt-oss-20b")

        assert spec.lora_config is not None
        assert spec.lora_config.target_parameters == [
            "mlp.experts.down_proj",
            "mlp.experts.gate_up_proj",
        ]
        assert spec.lora_config.lora_dropout == 0.0

    def test_empty_parameters_adapt_no_experts(self) -> None:
        spec = finetuning_network("openai/gpt-oss-20b", target_parameters=[])

        assert spec.lora_config is not None
        assert spec.lora_config.target_parameters == []
        assert spec.lora_config.lora_dropout == 0.05

    def test_explicit_dropout_is_kept(self) -> None:
        spec = FinetuningNetworkSpec.model_validate(
            {
                "pretrained_model_name_or_path": "openai/gpt-oss-20b",
                "max_context_length": 512,
                "lora_config": {"lora_dropout": 0.1},
            }
        )

        assert spec.lora_config is not None
        assert spec.lora_config.target_parameters is not None
        assert spec.lora_config.lora_dropout == 0.1

    def test_dense_model_stays_without_parameters(self) -> None:
        spec = finetuning_network("Qwen/Qwen3-4B")

        assert spec.lora_config is not None
        assert spec.lora_config.target_parameters is None
