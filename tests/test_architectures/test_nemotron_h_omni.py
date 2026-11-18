# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Native Super-VL loading without checkpoint code (CPU, tiny weights)."""

from __future__ import annotations

import importlib

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from transformers.models.nemotron_h_omni.configuration_nemotron_h_omni import (
    NemotronH_Omni_Reasoning_V3_Config,
)
from transformers.models.nemotron_h_omni.modeling_nemotron_h_omni import (
    NemotronH_Omni_Reasoning_V3,
)
from transformers.models.radio.configuration_radio import RadioConfig

from agilerl.architectures import nemotron_h_omni
from agilerl.architectures.nemotron_h_omni import (
    register_nemotron_h_omni_native_loading,
)


def _tiny_config() -> NemotronH_Omni_Reasoning_V3_Config:
    return NemotronH_Omni_Reasoning_V3_Config(
        text_config=NemotronHConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            layers_block_type=["linear_attention", "full_attention"],
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
            n_groups=1,
            ssm_state_size=4,
            conv_kernel=2,
            expand=2,
            mamba_num_heads=2,
            tie_word_embeddings=False,
            num_nextn_predict_layers=1,
        ),
        vision_config=RadioConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=16,
            patch_size=8,
            video_temporal_patch_size=2,
        ),
        audio_config=None,
        projector_hidden_size=16,
        vision_hidden_size=32,
        image_token_id=5,
    )


def _tiny_model() -> torch.nn.Module:
    register_nemotron_h_omni_native_loading()
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(_tiny_config()).eval()


class TestRegisterNemotronHOmniNativeLoading:
    def test_registration_is_idempotent(self) -> None:
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping

        register_nemotron_h_omni_native_loading()
        wrapped = NemotronH_Omni_Reasoning_V3.get_image_features
        register_nemotron_h_omni_native_loading()

        assert NemotronH_Omni_Reasoning_V3.get_image_features is wrapped
        assert wrapped.__module__ == nemotron_h_omni.__name__
        entries = get_checkpoint_conversion_mapping("NemotronH_Omni_Reasoning_V3")
        sources = [
            pattern
            for transform in entries
            for pattern in (
                transform.source_patterns
                if isinstance(transform.source_patterns, list)
                else [transform.source_patterns]
            )
        ]
        assert sources.count(r"^vision_projector\.vision_final_layernorm\.") == 1

    def test_without_llm_dependencies_registers_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agilerl

        monkeypatch.setattr(agilerl, "HAS_LLM_DEPENDENCIES", False)
        try:
            module = importlib.reload(nemotron_h_omni)
            for name in (
                "get_checkpoint_conversion_mapping",
                "register_checkpoint_conversion_mapping",
                "WeightRenaming",
                "MODEL_FOR_CAUSAL_LM_MAPPING",
                "NemotronH_Omni_Reasoning_V3_Config",
                "NemotronH_Omni_Reasoning_V3",
                "NemotronH_Omni_Reasoning_V3PreTrainedModel",
            ):
                assert getattr(module, name) is None

            assert module.register_nemotron_h_omni_native_loading() is None
        finally:
            monkeypatch.undo()
            importlib.reload(nemotron_h_omni)


class TestNativeOmniCheckpointRoundtrip:
    def test_raw_layout_checkpoint_loads_strict(self, tmp_path) -> None:
        from transformers.core_model_loading import revert_weight_conversion

        register_nemotron_h_omni_native_loading()
        config = _tiny_config()
        native = AutoModelForCausalLM.from_config(config)

        raw_state = revert_weight_conversion(native, dict(native.state_dict()))
        assert any(key.startswith("mlp1.") for key in raw_state)
        # The four keys the native entry does not cover.
        norm_weight = native.vision_final_layernorm.weight.detach().clone()
        norm_bias = native.vision_final_layernorm.bias.detach().clone()
        raw_state["vision_projector.vision_final_layernorm.weight"] = norm_weight
        raw_state["vision_projector.vision_final_layernorm.bias"] = norm_bias
        raw_state["vision_model.radio_model.input_conditioner.norm_mean"] = torch.zeros(
            3
        )
        raw_state["vision_model.radio_model.input_conditioner.norm_std"] = torch.ones(3)
        raw_state["language_model.mtp.layers.0.hnorm.weight"] = torch.zeros(32)
        config.save_pretrained(tmp_path)
        save_file(raw_state, tmp_path / "model.safetensors")

        loaded, info = AutoModelForCausalLM.from_pretrained(
            tmp_path, trust_remote_code=False, output_loading_info=True
        )

        assert info["missing_keys"] == set()
        assert info["unexpected_keys"] == set()
        assert torch.equal(
            loaded.multi_modal_projector.linear_1.weight, raw_state["mlp1.1.weight"]
        )
        assert torch.equal(
            loaded.vision_final_layernorm.weight,
            raw_state["vision_projector.vision_final_layernorm.weight"],
        )


class TestPackedPatches:
    def test_unfolds_rows_into_row_major_patches_and_grids(self) -> None:
        # Arrange
        packed = torch.arange(2 * 2 * 4 * 3, dtype=torch.float32).reshape(16, 3)
        rows = packed.reshape(2, 2, 4, 3)

        # Act
        patches, image_grid_hw = nemotron_h_omni.packed_patches(rows)

        # Assert
        assert torch.equal(patches, packed)
        assert image_grid_hw.tolist() == [[2, 4], [2, 4]]


@pytest.mark.usefixtures("torch_kernels")
class TestNemotronHOmniGetImageFeatures:
    def test_patch_grid_rows_forward_matches_packed_patches(self) -> None:
        # Arrange
        model = _tiny_model()
        # One 2x2 grid of 8x8 RGB patches, which pixel-shuffles into one image token (id 5).
        rows = torch.randn(1, 2, 2, 3 * 8 * 8)
        patches, image_grid_hw = nemotron_h_omni.packed_patches(rows)
        input_ids = torch.tensor([[1, 5, 2, 3]])

        # Act
        with torch.no_grad():
            from_rows = model(input_ids=input_ids, pixel_values=rows).logits
            from_patches = model(
                input_ids=input_ids,
                pixel_values=patches,
                image_grid_hw=image_grid_hw,
            ).logits

        # Assert
        assert torch.equal(from_rows, from_patches)


class TestNemotronHOmniVisionFillerFeatures:
    def test_runs_the_tower_on_one_image(self) -> None:
        # Arrange
        model = _tiny_model()
        tower_inputs: list[torch.Tensor] = []
        model.vision_model.register_forward_hook(
            lambda _module, args, _output: tower_inputs.append(args[0])
        )

        # Act
        with torch.no_grad():
            features = model.vision_filler_features(torch.device("cpu"))

        # Assert
        assert len(tower_inputs) == 1
        assert [tuple(image.shape) for image in features.pooler_output] == [(1, 32)]
