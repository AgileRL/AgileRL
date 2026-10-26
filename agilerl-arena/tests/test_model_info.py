# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-model info, LoRA validation, and bundled configs."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agilerl.arena.models import model_info
from agilerl.arena.models.model_info import (
    ARCH_DENSE,
    ARCH_HYBRID,
    ARCH_HYBRID_MOE,
    ARCH_MOE,
    STATUS_DEPRECATED,
    STATUS_LIVE,
    SUPPORTED_MODEL_INFO,
    ModelInfo,
    supported_path,
)
from agilerl.arena.models.networks import (
    FinetuningNetworkSpec,
    LoraConfigDict,
    resolve_lora_defaults,
)


def finetuning_network(
    hub_id: str,
    target_modules: object = ("q_proj",),
    target_parameters: object = None,
    lora_r: int = 16,
    tower_target_modules: object = None,
) -> FinetuningNetworkSpec:
    """Finetuning section for a Hub id with explicit LoRA targets."""
    lora: dict[str, object] = {"target_modules": target_modules, "lora_r": lora_r}
    if target_parameters is not None:
        lora["target_parameters"] = target_parameters
    if tower_target_modules is not None:
        lora["tower_target_modules"] = tower_target_modules
    return FinetuningNetworkSpec.model_validate(
        {
            "pretrained_model_name_or_path": hub_id,
            "max_context_length": 512,
            "lora_config": lora,
        }
    )


@pytest.fixture
def no_dims_entry(monkeypatch: pytest.MonkeyPatch) -> ModelInfo:
    """Entry whose bundled ``lora_info`` names targets without dims."""
    bundled = {
        "config": {"model_type": "nemotron_h"},
        "inspected": {
            "num_params": None,
            "lora_info": {
                "q_proj": [{"kind": "module"}],
                "mixer.experts.up_proj": [{"kind": "parameter"}],
            },
        },
    }
    monkeypatch.setattr(model_info, "_bundled", lambda hub_id: bundled)
    return ModelInfo(hub_id="org/no-dims", architecture=ARCH_HYBRID_MOE)


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

    def test_super_vl_language_targets_carry_language_model_scope(self) -> None:
        entry = SUPPORTED_MODEL_INFO[
            "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
        ]
        language_targets = {
            key
            for key, group in entry.inspected["lora_info"].items()
            if group[0].get("scope", "").startswith("language_model.backbone.layers")
        }

        assert language_targets == {
            "down_proj",
            "in_proj",
            "k_proj",
            "mixer.experts.down_proj",
            "mixer.experts.up_proj",
            "o_proj",
            "out_proj",
            "q_proj",
            "up_proj",
            "v_proj",
        }
        assert entry.parameters == frozenset(
            {"mixer.experts.down_proj", "mixer.experts.up_proj"}
        )
        assert entry.num_params is None
        assert entry.lora_ranks == (1, 8, 16, 32, 64, 128)

    def test_super_vl_vision_and_projector_targets_have_no_dims(self) -> None:
        entry = SUPPORTED_MODEL_INFO[
            "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
        ]
        vision_targets = {
            "fc1",
            "fc2",
            "linear_1",
            "linear_2",
        }

        assert vision_targets <= entry.modules
        for key in vision_targets:
            (target,) = entry.inspected["lora_info"][key]
            assert "language_model" not in target.get("scope", "")
            assert "in_features" not in target
            assert "out_features" not in target
        vision_scopes = [
            target.get("scope", "") for target in entry.inspected["lora_info"]["q_proj"]
        ]

        assert "vision_model.encoder.layer.attention" in vision_scopes

    def test_super_vl_excludes_latent_projections(self) -> None:
        entry = SUPPORTED_MODEL_INFO[
            "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
        ]

        assert not {"fc1_latent_proj", "fc2_latent_proj"} & entry.modules

    def test_qwen3_6_lists_vision_and_experts(self) -> None:
        entry = SUPPORTED_MODEL_INFO["Qwen/Qwen3.6-35B-A3B"]

        assert {"qkv", "linear_fc1", "linear_fc2"} <= entry.modules
        assert "mlp.experts.gate_up_proj" in entry.parameters
        assert entry.lora_ranks == (1, 8, 16, 32)

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
    def test_supported_path_reads_bundled_json(self) -> None:
        entry = SUPPORTED_MODEL_INFO["Qwen/Qwen3-4B"]

        payload = json.loads(supported_path(entry.hub_id).read_text(encoding="utf-8"))

        assert payload == {"config": entry.config, "inspected": entry.inspected}

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

    def test_super_vl_in_proj_carries_language_dims(self) -> None:
        entry = SUPPORTED_MODEL_INFO[
            "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
        ]

        assert entry.lora_info["in_proj"] == [
            {
                "kind": "module",
                "scope": "language_model.backbone.layers.mixer",
                "type": "Linear",
                "in_features": 4096,
                "out_features": 18560,
                "count": 40,
            }
        ]

    def test_entry_without_dims_has_no_ranks(self, no_dims_entry: ModelInfo) -> None:
        assert no_dims_entry.lora_ranks is None
        assert no_dims_entry.num_params is None
        assert "mixer.experts.up_proj" in no_dims_entry.parameters

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
        assert architectures["Qwen/Qwen3.5-4B"] == ARCH_HYBRID
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


class TestModelInfoStatus:
    def test_listed_statuses(self) -> None:
        statuses = {
            hub_id: entry.status for hub_id, entry in SUPPORTED_MODEL_INFO.items()
        }
        deprecated = {
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen3-1.7B",
            "Qwen/Qwen3-4B",
            "ibm-granite/granite-4.0-micro",
            "ibm-granite/granite-4.0-micro-base",
            "ibm-granite/granite-4.0-h-tiny",
            "ibm-granite/granite-3.1-3b-a800m-instruct",
        }

        for hub_id, status in statuses.items():
            if hub_id in deprecated:
                assert status == STATUS_DEPRECATED, hub_id
            else:
                assert status == STATUS_LIVE, hub_id

    def test_defaults_to_live(self) -> None:
        assert ModelInfo(hub_id="org/x").status == STATUS_LIVE

    def test_serializes_as_lowercase_string(self) -> None:
        entry = SUPPORTED_MODEL_INFO[
            "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
        ]
        assert entry.model_dump(mode="json")["status"] == "live"

    def test_rejects_unknown_status(self) -> None:
        with pytest.raises(ValidationError, match="status"):
            ModelInfo.model_validate({"hub_id": "org/x", "status": "beta"})


class TestModelInfoModalitiesProperty:
    def test_defaults_to_text(self) -> None:
        assert ModelInfo(hub_id="org/x").modalities == ("text",)

    def test_entries_with_a_vision_tower_handle_vision(self) -> None:
        for hub_id, entry in SUPPORTED_MODEL_INFO.items():
            expected = (
                ("text", "vision") if "vision_config" in entry.config else ("text",)
            )

            assert entry.modalities == expected, hub_id

    @pytest.mark.parametrize(
        "hub_id",
        [
            "Qwen/Qwen3.5-0.8B",
            "Qwen/Qwen3.6-35B-A3B",
            "google/gemma-4-E4B-it",
            "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
        ],
    )
    def test_vl_entries_handle_text_and_vision(self, hub_id: str) -> None:
        assert SUPPORTED_MODEL_INFO[hub_id].modalities == ("text", "vision")

    def test_text_entries_keep_default(self) -> None:
        assert SUPPORTED_MODEL_INFO["Qwen/Qwen3-4B"].modalities == ("text",)

    def test_text_model_serializes_text_list(self) -> None:
        dumped = SUPPORTED_MODEL_INFO["Qwen/Qwen3-4B"].model_dump(mode="json")

        assert dumped["modalities"] == ["text"]

    def test_super_vl_serializes_both_modalities(self) -> None:
        dumped = SUPPORTED_MODEL_INFO[
            "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
        ].model_dump(mode="json")

        assert dumped["modalities"] == ["text", "vision"]

    def test_rejects_unknown_modality(self) -> None:
        with pytest.raises(ValidationError, match="modalities"):
            ModelInfo.model_validate({"hub_id": "org/x", "modalities": ["audio"]})

    def test_rejects_empty_modalities(self) -> None:
        with pytest.raises(ValidationError, match="modalities"):
            ModelInfo.model_validate({"hub_id": "org/x", "modalities": []})

    def test_rejects_duplicate_modalities(self) -> None:
        with pytest.raises(ValidationError, match="modalities"):
            ModelInfo.model_validate(
                {"hub_id": "org/x", "modalities": ["text", "text"]}
            )

    def test_rejects_vision_only(self) -> None:
        with pytest.raises(ValidationError, match="modalities"):
            ModelInfo.model_validate({"hub_id": "org/x", "modalities": ["vision"]})


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

    def test_unverified_ranks_skip_rank_check(
        self, monkeypatch: pytest.MonkeyPatch, no_dims_entry: ModelInfo
    ) -> None:
        monkeypatch.setitem(SUPPORTED_MODEL_INFO, no_dims_entry.hub_id, no_dims_entry)

        spec = finetuning_network(no_dims_entry.hub_id, lora_r=1024)

        assert spec.lora_config is not None

    def test_super_vl_accepts_vision_modules_as_tower_targets(self) -> None:
        spec = finetuning_network(
            "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
            ("q_proj",),
            lora_r=128,
            tower_target_modules=["q_proj", "linear_1"],
        )

        assert spec.lora_config is not None
        assert spec.lora_config.tower_target_modules == ["q_proj", "linear_1"]

    @pytest.mark.parametrize(
        ("hub_id", "tower_module"),
        [
            ("nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16", "fc1"),
            ("Qwen/Qwen3.5-2B", "qkv"),
            ("google/gemma-4-E4B-it", "linear"),
        ],
    )
    def test_tower_module_in_target_modules_raises(
        self, hub_id: str, tower_module: str
    ) -> None:
        with pytest.raises(
            ValidationError, match=rf"\['{tower_module}'\] are tower modules"
        ):
            finetuning_network(hub_id, ("q_proj", tower_module))

    def test_language_module_as_tower_target_raises(self) -> None:
        with pytest.raises(
            ValidationError, match=r"Unknown LoRA tower_target_modules \['q_proj'\]"
        ):
            finetuning_network(
                "Qwen/Qwen3.5-2B", ("q_proj",), tower_target_modules=["q_proj"]
            )

    def test_tower_targets_on_text_model_raise(self) -> None:
        with pytest.raises(ValidationError, match="valid tower modules: none"):
            finetuning_network(
                "Qwen/Qwen3-4B", ("q_proj",), tower_target_modules=["qkv"]
            )

    def test_super_vl_rejects_latent_projections(self) -> None:
        with pytest.raises(
            ValidationError, match=r"Unknown LoRA target_modules \['fc1_latent_proj'\]"
        ):
            finetuning_network(
                "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
                ("q_proj", "fc1_latent_proj"),
            )

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

    def test_explicit_dropout_is_zeroed_when_experts_auto_filled(self) -> None:
        spec = FinetuningNetworkSpec.model_validate(
            {
                "pretrained_model_name_or_path": "openai/gpt-oss-20b",
                "max_context_length": 512,
                "lora_config": {"lora_dropout": 0.1},
            }
        )

        assert spec.lora_config is not None
        assert spec.lora_config.target_parameters is not None
        assert spec.lora_config.lora_dropout == 0.0

    def test_dense_model_stays_without_parameters(self) -> None:
        spec = finetuning_network("Qwen/Qwen3-4B")

        assert spec.lora_config is not None
        assert spec.lora_config.target_parameters is None


VL_HUB_IDS = (
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3.8-27B",
    "google/gemma-4-E4B-it",
    "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
)


class TestModelInfoDefaultLoraModulesProperty:
    def test_vl_entries_are_exactly_the_checkpoints_with_towers(self) -> None:
        vl = {h for h, e in SUPPORTED_MODEL_INFO.items() if e.language_model_prefix}

        assert vl == set(VL_HUB_IDS)

    @pytest.mark.parametrize("hub_id", VL_HUB_IDS)
    def test_vl_default_is_every_module_scoped_under_the_language_model(
        self, hub_id: str
    ) -> None:
        entry = SUPPORTED_MODEL_INFO[hub_id]
        prefix = entry.language_model_prefix
        assert prefix is not None
        language = {
            key
            for key, group in entry.inspected["lora_info"].items()
            if group[0]["kind"] == "module"
            and any(item.get("scope", "").startswith(prefix) for item in group)
        }

        assert entry.default_lora_modules == language
        assert entry.default_lora_modules | entry.tower_lora_modules == entry.modules

    def test_super_vl_attention_names_are_language_and_tower_modules(self) -> None:
        entry = SUPPORTED_MODEL_INFO[
            "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
        ]
        attention = {"q_proj", "k_proj", "v_proj", "o_proj"}

        assert attention <= entry.default_lora_modules
        assert attention <= entry.tower_lora_modules
        assert entry.tower_lora_modules.isdisjoint({"in_proj", "out_proj"})

    def test_text_entry_defaults_to_every_module(self) -> None:
        entry = SUPPORTED_MODEL_INFO["Qwen/Qwen3-4B"]

        assert entry.default_lora_modules == entry.modules
        assert entry.tower_lora_modules == frozenset()

    def test_qwen3_5_default_skips_vision_modules(self) -> None:
        entry = SUPPORTED_MODEL_INFO["Qwen/Qwen3.5-2B"]

        assert entry.tower_lora_modules == {"qkv", "linear_fc1", "linear_fc2"}
        assert "in_proj_qkv" in entry.default_lora_modules

    def test_serializes_default_modules_sorted(self) -> None:
        dumped = SUPPORTED_MODEL_INFO["google/gemma-4-E4B-it"].model_dump()

        assert dumped["default_lora_modules"] == sorted(dumped["text_lora_modules"])
        assert "linear" not in dumped["default_lora_modules"]


class TestModelInfoVlFieldsValidation:
    def test_prefix_without_text_modules_raises(self) -> None:
        with pytest.raises(ValidationError, match="set together"):
            ModelInfo(hub_id="Qwen/Qwen3.5-2B", language_model_prefix="model.x")

    def test_unknown_text_module_raises(self) -> None:
        with pytest.raises(ValidationError, match=r"\['nope'\] are not LoRA modules"):
            ModelInfo(
                hub_id="Qwen/Qwen3.5-2B",
                language_model_prefix="model.language_model",
                text_lora_modules=frozenset({"q_proj", "nope"}),
            )


class TestResolveLoraDefaults:
    def test_packed_moe_model_adapts_every_expert(self) -> None:
        lora = LoraConfigDict()

        resolve_lora_defaults(lora, "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")

        assert lora.target_parameters == [
            "mixer.experts.down_proj",
            "mixer.experts.up_proj",
        ]
        assert lora.lora_dropout == 0.0

    @pytest.mark.parametrize(
        ("hub_id", "target_parameters"),
        [
            ("Qwen/Qwen3.6-35B-A3B", []),
            ("Qwen/Qwen3.6-35B-A3B", ["mlp.experts.down_proj"]),
            ("Qwen/Qwen3.5-2B", None),
            ("org/untracked", None),
        ],
    )
    def test_leaves_explicit_dense_and_untracked_configs(
        self, hub_id: str, target_parameters: list[str] | None
    ) -> None:
        lora = LoraConfigDict(target_parameters=target_parameters)

        resolve_lora_defaults(lora, hub_id)

        assert lora.target_parameters == target_parameters
        assert lora.lora_dropout == 0.05
