# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""End-to-end loading of a tiny FP8 / MXFP4 ``mimo_v2`` checkpoint as the native model."""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.mimo_v2_flash.modeling_mimo_v2_flash import (
    MiMoV2FlashForCausalLM,
    MiMoV2FlashPreTrainedModel,
)

from agilerl.architectures import mimo_v2
from agilerl.architectures.mimo_v2 import install_mimo_v2_native_training
from agilerl.architectures.moe_router import patch_moe_routers
from agilerl.architectures.runtime import PatchRuntimeConfig
from agilerl.distributed.fsdp import (
    _checkpoint_transform_groups,
    _load_converter_route,
    _load_sharded_weights_from_safetensors,
)
from agilerl.distributed.fsdp_meta import converter_checkpoint_routes, init_rope_buffers
from agilerl.lora.moe.layouts import is_routed_experts_module
from tests.helper_functions import load_without_llm_dependencies

INPUT_IDS = torch.tensor([[3, 17, 42, 5, 99, 64, 8, 21, 77, 1, 120, 33]])


def safetensors_keys(path) -> list[str]:
    with safe_open(str(path), framework="pt") as handle:
        return list(handle.keys())


def load_native(directory, dtype: torch.dtype) -> MiMoV2FlashForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        directory, dtype=dtype, attn_implementation="eager"
    )


def reference_model(
    config, reference: dict[str, torch.Tensor]
) -> MiMoV2FlashForCausalLM:
    """Native model holding the independently dequantized weights."""
    model = MiMoV2FlashForCausalLM(config).float()
    model.load_state_dict(reference)
    return model.eval()


class TestFromPretrained:
    def test_dequantizes_every_weight_exactly(self, tiny_checkpoint):
        # Arrange
        directory, reference = tiny_checkpoint

        # Act
        model, info = AutoModelForCausalLM.from_pretrained(
            directory,
            dtype=torch.float32,
            attn_implementation="eager",
            output_loading_info=True,
        )

        # Assert
        assert type(model) is MiMoV2FlashForCausalLM
        assert info["missing_keys"] == set()
        assert info["unexpected_keys"] == set()
        assert info["mismatched_keys"] == set()
        assert getattr(model.config, "quantization_config", None) is None
        state = model.state_dict()
        assert state.keys() == reference.keys()
        for name, expected in reference.items():
            assert state[name].dtype == torch.float32, name
            assert torch.equal(state[name], expected), name

    def test_logits_match_reference_model(self, tiny_checkpoint, native_config):
        # Arrange
        directory, reference = tiny_checkpoint
        model = load_native(directory, torch.float32).eval()
        expected_model = reference_model(native_config, reference)

        # Act
        with torch.no_grad():
            logits = model(input_ids=INPUT_IDS).logits
            expected = expected_model(input_ids=INPUT_IDS).logits

        # Assert
        # Same weights and kernels; the tolerance only absorbs CPU reduction order.
        assert torch.allclose(logits, expected, rtol=1e-6, atol=1e-6)

    def test_bf16_load_keeps_only_the_correction_bias_in_fp32(self, tiny_checkpoint):
        directory, _ = tiny_checkpoint

        model = load_native(directory, torch.bfloat16)

        assert {
            name
            for name, tensor in model.state_dict().items()
            if tensor.dtype != torch.bfloat16
        } == {"model.layers.1.mlp.gate.e_score_correction_bias"}


class TestLoadShardedWeightsFromSafetensors:
    def test_meta_model_loads_the_same_weights_as_from_pretrained(
        self, tiny_checkpoint
    ):
        # Arrange
        directory, _ = tiny_checkpoint
        expected = load_native(directory, torch.bfloat16).state_dict()
        config = AutoConfig.from_pretrained(directory)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                config, dtype=torch.bfloat16, attn_implementation="eager"
            )
        model.to_empty(device="cpu")

        # Act
        _load_sharded_weights_from_safetensors(model)
        init_rope_buffers(model)

        # Assert
        state = model.state_dict()
        for name, tensor in expected.items():
            if name.endswith("e_score_correction_bias"):
                continue
            assert state[name].dtype == torch.bfloat16, name
            assert torch.equal(state[name], tensor), name

    def test_expert_route_reads_only_this_ranks_experts(
        self, tiny_checkpoint, native_config
    ):
        # Arrange
        directory, reference = tiny_checkpoint
        model = MiMoV2FlashForCausalLM(native_config)
        model.config.name_or_path = str(directory)
        key_files = {
            key: str(path)
            for path in directory.glob("*.safetensors")
            for key in safetensors_keys(path)
        }
        routes = converter_checkpoint_routes(
            key_files, _checkpoint_transform_groups(model)
        )
        target = "model.layers.1.mlp.experts.gate_up_proj"
        dest = torch.zeros(3, 64, 64)

        # Act
        _load_converter_route(
            routes[target],
            key_files,
            native_config,
            [(target, (slice(2, 5), slice(0, 64), slice(0, 64)), dest, 8)],
        )

        # Assert
        assert len(routes[target].sources) == 32
        assert torch.equal(dest, reference[target][2:5])


class TestPackedExperts:
    def test_loaded_experts_are_routed_packed_experts(self, tiny_checkpoint):
        directory, _ = tiny_checkpoint

        model = load_native(directory, torch.float32)

        experts = model.model.layers[1].mlp.experts
        assert is_routed_experts_module(experts)
        assert experts.gate_up_proj.shape == (8, 64, 64)
        assert experts.down_proj.shape == (8, 64, 32)


class TestLoraForwardBackward:
    def test_lora_adapters_get_finite_gradients(self, tiny_checkpoint):
        # Arrange
        directory, _ = tiny_checkpoint
        torch.manual_seed(0)
        model = get_peft_model(
            load_native(directory, torch.float32),
            LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
                init_lora_weights=False,
            ),
        )
        model.train()

        # Act
        loss = model(input_ids=INPUT_IDS, labels=INPUT_IDS).loss
        loss.backward()

        # Assert
        assert torch.isfinite(loss)
        lora_grads = {
            name: param.grad
            for name, param in model.named_parameters()
            if "lora_" in name
        }
        assert len(lora_grads) == 2 * 9
        for name, grad in lora_grads.items():
            assert grad is not None, name
            assert torch.isfinite(grad).all(), name
            assert grad.abs().sum() > 0, name
        assert all(
            param.grad is None
            for name, param in model.named_parameters()
            if "lora_" not in name
        )


@pytest.mark.usefixtures("pristine_moe_routers")
class TestRouterPatch:
    def test_patched_router_keeps_logits_bitwise_equal(self, tiny_checkpoint):
        # Arrange
        directory, _ = tiny_checkpoint
        model = load_native(directory, torch.float32).eval()
        with torch.no_grad():
            expected = model(input_ids=INPUT_IDS).logits

        # Act
        patch_moe_routers(model)
        with torch.no_grad():
            logits = model(input_ids=INPUT_IDS).logits

        # Assert
        assert torch.equal(logits, expected)


class TestRegisterMimoV2NativeLoading:
    def test_without_llm_dependencies_registers_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        module = load_without_llm_dependencies(mimo_v2, monkeypatch)

        for name in (
            "AutoConfig",
            "register_checkpoint_conversion_mapping",
            "MODEL_FOR_CAUSAL_LM_MAPPING",
            "MiMoV2FlashForCausalLM",
            "MiMoV2FlashPreTrainedModel",
            "install_mimo_v2_fp32_lse_flex_attention",
            "TOWER_KEY_PATTERNS",
            "mimo_v2_checkpoint_conversions",
            "MIMO_V2_MODEL_TYPE",
            "MiMoV2Config",
        ):
            assert getattr(module, name) is None, name
        assert module.register_mimo_v2_native_loading() is None


class TestInstallMimoV2NativeTraining:
    @pytest.mark.parametrize(
        ("install", "expected"),
        [(False, "False"), (True, "True")],
        ids=["import-keeps-hub-config", "install-registers-native-config"],
    )
    def test_registers_the_native_config_only_when_installed(
        self, install: bool, expected: str
    ):
        # A fresh interpreter: the package conftest registers the config in this one.
        script = (
            "import agilerl.architectures\n"
            "from agilerl.architectures.mimo_v2 import install_mimo_v2_native_training\n"
            "from agilerl.architectures.runtime import PatchRuntimeConfig\n"
            "from transformers.models.auto.configuration_auto import CONFIG_MAPPING\n"
            f"if {install}:\n"
            "    install_mimo_v2_native_training(PatchRuntimeConfig())\n"
            "print('mimo_v2' in CONFIG_MAPPING)\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )

        assert result.stdout.strip().splitlines()[-1] == expected

    def test_lets_the_native_class_build_with_flex_attention(
        self, native_config, monkeypatch: pytest.MonkeyPatch
    ):
        # Arrange
        monkeypatch.setattr(MiMoV2FlashPreTrainedModel, "_supports_flex_attn", False)

        # Act
        install_mimo_v2_native_training(PatchRuntimeConfig())
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                native_config, attn_implementation="flex_attention"
            )

        # Assert
        assert model.config._attn_implementation == "flex_attention"

    def test_native_class_rejects_flex_attention_without_it(
        self, native_config, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(MiMoV2FlashPreTrainedModel, "_supports_flex_attn", False)

        with pytest.raises(ValueError, match="flex"), torch.device("meta"):
            AutoModelForCausalLM.from_config(
                native_config, attn_implementation="flex_attention"
            )
