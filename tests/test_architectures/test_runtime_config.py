# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for model-type trainer and vLLM runtime configs."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from pydantic import ValidationError
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.configuration_utils import PretrainedConfig
from transformers.models.nemotron_h_omni.configuration_nemotron_h_omni import (
    NemotronH_Omni_Reasoning_V3_Config,
)

import agilerl
from agilerl import architectures
from agilerl.architectures import (
    family_context_parallel_plans,
    family_tensor_parallel_plans,
)
from agilerl.architectures import runtime as runtime_module
from agilerl.architectures.catalog import (
    FAMILIES,
    FAMILY_RUNTIME_CONFIGS,
    family_chat_template,
    family_runtime,
    model_type_trainer_class,
    model_type_vision,
    pretrained_model_type,
    sm80_attention_backend,
    trainer_model_class,
)
from agilerl.architectures.deepseek_v4.context_parallel import (
    CONTEXT_PARALLEL_PLAN as DEEPSEEK_V4_CONTEXT_PARALLEL_PLAN,
)
from agilerl.architectures.deepseek_v4.tensor_parallel import (
    TENSOR_PARALLEL_PLAN as DEEPSEEK_V4_TENSOR_PARALLEL_PLAN,
)
from agilerl.architectures.gemma4.tensor_parallel import (
    TENSOR_PARALLEL_PLAN as GEMMA4_TENSOR_PARALLEL_PLAN,
)
from agilerl.architectures.glm5_next.context_parallel import (
    CONTEXT_PARALLEL_PLAN as GLM5_NEXT_CONTEXT_PARALLEL_PLAN,
)
from agilerl.architectures.glm5_next.tensor_parallel import (
    TENSOR_PARALLEL_PLAN as GLM5_NEXT_TENSOR_PARALLEL_PLAN,
)
from agilerl.architectures.gptoss.tensor_parallel import (
    TENSOR_PARALLEL_PLAN as GPT_OSS_TENSOR_PARALLEL_PLAN,
)
from agilerl.architectures.granitemoe.tensor_parallel import (
    TENSOR_PARALLEL_PLAN as GRANITEMOE_TENSOR_PARALLEL_PLAN,
)
from agilerl.architectures.granitemoehybrid.context_parallel import (
    CONTEXT_PARALLEL_PLAN as GRANITEMOEHYBRID_CONTEXT_PARALLEL_PLAN,
)
from agilerl.architectures.granitemoehybrid.tensor_parallel import (
    TENSOR_PARALLEL_PLAN as GRANITEMOEHYBRID_TENSOR_PARALLEL_PLAN,
)
from agilerl.architectures.mimo_v2.configuration import MiMoV2Config
from agilerl.architectures.mimo_v2.patches import (
    install_mimo_v2_patches,
    register_mimo_v2_native_loading,
)
from agilerl.architectures.mimo_v2.tensor_parallel import (
    TENSOR_PARALLEL_PLAN as MIMO_V2_TENSOR_PARALLEL_PLAN,
)
from agilerl.architectures.nemotron_h.context_parallel import (
    CONTEXT_PARALLEL_PLAN as NEMOTRON_H_CONTEXT_PARALLEL_PLAN,
)
from agilerl.architectures.nemotron_h.patches import install_nemotron_h_patches
from agilerl.architectures.nemotron_h.tensor_parallel import (
    TENSOR_PARALLEL_PLAN as NEMOTRON_H_TENSOR_PARALLEL_PLAN,
)
from agilerl.architectures.nemotron_h_omni.vision import (
    nemotron_h_omni_vllm_hf_override,
)
from agilerl.architectures.qwen3.tensor_parallel import (
    TENSOR_PARALLEL_PLAN as QWEN3_TENSOR_PARALLEL_PLAN,
)
from agilerl.architectures.qwen3_5.context_parallel import (
    CONTEXT_PARALLEL_PLAN as QWEN3_5_CONTEXT_PARALLEL_PLAN,
)
from agilerl.architectures.qwen3_5.tensor_parallel import (
    TENSOR_PARALLEL_PLAN as QWEN3_5_TENSOR_PARALLEL_PLAN,
)
from agilerl.architectures.runtime import (
    ModelRuntimeConfig,
    ModelTypes,
    TensorParallelPlan,
    VllmRuntimeConfig,
    discover_families,
    runtime_configs_by_model_type,
)
from agilerl.arena.memory import formulas
from agilerl.arena.memory.manifest import FAMILY_MAX_NUM_BATCHED_TOKENS
from agilerl.arena.memory.specs import TENSOR_PARALLEL_MODEL_TYPES, ModelArch
from agilerl.arena.models.model_info import (
    LORA_EXCLUDED_MODULES,
    REASONING_PARSERS,
    SUPPORTED_MODEL_INFO,
)
from agilerl.distributed.tensor_parallel import apply_tensor_parallel
from agilerl.llm_envs.chat_formats import CHAT_FORMATS, DsmlFormat
from tests.rank_launch import run_ranks

NEMOTRON_VLLM_KWARGS = {
    "mamba_cache_mode": "align",
    "max_num_batched_tokens": 8192,
    "enable_prefix_caching": True,
    "prefix_cache_retention_interval": None,
}

SWA_MODEL_TYPES = ("gemma3", "gemma3_text", "gemma4", "gemma4_text")

TENSOR_PARALLEL_SPLIT_CASES = [
    ("deepseek-ai/DeepSeek-V4-Flash-0731", 2),
    ("deepseek-ai/DeepSeek-V4-Flash-0731", 4),
    ("zai-org/GLM-5.3-Flash", 2),
    ("zai-org/GLM-5.3-Flash", 4),
    ("XiaomiMiMo/MiMo-V2.6-Flash-RL", 2),
    ("XiaomiMiMo/MiMo-V2.6-Flash-RL", 4),
    # 14 query heads: the plan rejects tp=4.
    ("Qwen/Qwen2.5-0.5B-Instruct", 2),
    ("Qwen/Qwen3.5-0.8B", 2),
    ("Qwen/Qwen3.5-0.8B", 4),
    ("Qwen/Qwen3.5-2B", 2),
    ("Qwen/Qwen3.5-2B", 4),
    ("Qwen/Qwen3.6-35B-A3B", 2),
    ("Qwen/Qwen3.6-35B-A3B", 4),
    ("google/gemma-4-E4B-it", 2),
    ("google/gemma-4-E4B-it", 4),
    ("ibm-granite/granite-4.0-h-tiny", 2),
    ("ibm-granite/granite-4.0-h-tiny", 4),
    ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", 2),
    ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", 4),
]


def tensor_parallel_split_worker(
    _rank: int, world_size: int, model_id: str
) -> tuple[int, int]:
    """Full and per-rank elements of the meta model's parameters the family TP plan splits."""
    register_mimo_v2_native_loading()
    hf_config = dict(SUPPORTED_MODEL_INFO[model_id].config)
    model_type = hf_config["model_type"]
    trainer = FAMILY_RUNTIME_CONFIGS[model_type].trainer
    if trainer.config_class is AutoConfig:
        config = AutoConfig.for_model(**hf_config)
    else:
        config = trainer.config_class.from_dict(hf_config)
    with torch.device("meta"):
        model = model_type_trainer_class(model_type).from_config(config)
    apply_tensor_parallel(
        model, init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    )
    split = [
        param
        for param in model.parameters()
        if isinstance(param, DTensor) and param.to_local().numel() < param.numel()
    ]
    return (
        sum(param.numel() for param in split),
        sum(param.to_local().numel() for param in split),
    )


def stub_config_model_type(monkeypatch: pytest.MonkeyPatch, model_type: str) -> None:
    @classmethod
    def fake_get_config_dict(
        cls, pretrained_model_name_or_path: str, **kwargs: object
    ) -> tuple[dict[str, object], dict[str, object]]:
        return ({"model_type": model_type}, {})

    monkeypatch.setattr(
        "agilerl.architectures.catalog.PretrainedConfig.get_config_dict",
        fake_get_config_dict,
    )


class TestFamilyRuntimeConfigs:
    def test_catalog_keys(self) -> None:
        assert set(FAMILY_RUNTIME_CONFIGS) == {
            "deepseek_v4",
            "llama",
            "qwen2",
            "qwen3",
            "qwen3_moe",
            "granitemoe",
            "granitemoehybrid",
            "nemotron_h",
            "nemotron_h_omni",
            "gemma3",
            "gemma3_text",
            "gemma4",
            "gemma4_text",
            "glm5_next",
            "glm5_next_text",
            "gpt_oss",
            "mimo_v2",
            "qwen3_5",
            "qwen3_5_text",
            "qwen3_5_moe",
            "qwen3_5_moe_text",
        }

    def test_mimo_v2_loads_the_native_class_with_its_flex_patch(self) -> None:
        mimo = FAMILY_RUNTIME_CONFIGS["mimo_v2"]

        assert mimo.trainer.load_kwargs() == {"trust_remote_code": False}
        assert model_type_trainer_class("mimo_v2") is AutoModelForImageTextToText
        assert mimo.trainer.config_class is MiMoV2Config
        assert mimo.vllm.engine_kwargs() == {
            "trust_remote_code": True,
            "hf_overrides": {"architectures": ["MiMoV2OmniForCausalLM"]},
        }
        assert mimo.patch.install is install_mimo_v2_patches
        assert mimo.chat_format == "tool_call"
        assert mimo.reasoning_parser == "mimo"

    def test_lora_exclude_modules_match_the_arena_table(self) -> None:
        excluded = {
            model_type: runtime.lora_exclude_modules
            for model_type, runtime in FAMILY_RUNTIME_CONFIGS.items()
            if runtime.lora_exclude_modules
        }

        assert excluded == LORA_EXCLUDED_MODULES
        assert {"indexer.wk", "f_a_proj", "g_a_proj"} <= excluded["glm5_next"]
        assert "o_a_proj" in excluded["deepseek_v4"]
        assert excluded["mimo_v2"] == {"q_proj", "k_proj", "v_proj"}

    def test_nemotron_h_omni_keeps_nemotron_h_runtime_and_sets_the_vllm_architecture(
        self,
    ) -> None:
        base = FAMILY_RUNTIME_CONFIGS["nemotron_h"]
        omni = FAMILY_RUNTIME_CONFIGS["nemotron_h_omni"]
        assert omni.vllm.mamba_cache_mode == base.vllm.mamba_cache_mode
        assert omni.vllm.hf_overrides is nemotron_h_omni_vllm_hf_override
        assert base.vllm.hf_overrides is None
        assert omni.patch.install is install_nemotron_h_patches
        omni_vision = model_type_vision("nemotron_h_omni")
        assert omni_vision is not None
        assert {"q_proj", "fc1", "linear_1"} <= omni_vision.tower_connector_lora_targets
        assert omni_vision.tower_connector_lora_targets.isdisjoint(
            {"in_proj", "up_proj", "down_proj", "out_proj"}
        )
        assert model_type_vision("nemotron_h") is None

    def test_gemma4_uses_the_swa_trainer(self) -> None:
        gemma4 = FAMILY_RUNTIME_CONFIGS["gemma4"]
        gemma3 = FAMILY_RUNTIME_CONFIGS["gemma3"]
        assert gemma4.trainer == gemma3.trainer

    def test_every_arena_tower_lora_module_is_served(self) -> None:
        served = {}
        for hub_id, entry in SUPPORTED_MODEL_INFO.items():
            vision = model_type_vision(entry.config["model_type"])
            served[hub_id] = (
                frozenset() if vision is None else vision.tower_connector_lora_targets
            )

        unserved = {
            hub_id: sorted(entry.tower_lora_modules - served[hub_id])
            for hub_id, entry in SUPPORTED_MODEL_INFO.items()
        }

        assert {hub_id: names for hub_id, names in unserved.items() if names} == {}

    @pytest.mark.parametrize(
        "model_type",
        ["qwen3_5", "qwen3_5_moe", "gemma3", "gemma4", "glm5_next", "nemotron_h_omni"],
    )
    def test_vl_families_train_their_vl_class(self, model_type: str) -> None:
        assert model_type_trainer_class(model_type) is AutoModelForImageTextToText

    @pytest.mark.parametrize(
        "model_type",
        sorted(
            set(FAMILY_RUNTIME_CONFIGS)
            - {
                "qwen3_5",
                "qwen3_5_moe",
                "gemma3",
                "gemma4",
                "glm5_next",
                "mimo_v2",
                "nemotron_h_omni",
            }
        ),
    )
    def test_text_families_train_a_causal_lm(self, model_type: str) -> None:
        assert model_type_trainer_class(model_type) is AutoModelForCausalLM

    def test_nemotron_h_lookup(self) -> None:
        config = FAMILY_RUNTIME_CONFIGS["nemotron_h"]
        assert config.vllm.engine_kwargs() == NEMOTRON_VLLM_KWARGS
        assert config.trainer.load_kwargs() == {}
        assert config.patch.install is install_nemotron_h_patches

    def test_no_family_loads_checkpoint_code(self) -> None:
        omni = FAMILY_RUNTIME_CONFIGS["nemotron_h_omni"]

        assert omni.trainer.load_kwargs() == {"trust_remote_code": False}
        assert all(
            runtime.trainer.trust_remote_code is not True
            for runtime in FAMILY_RUNTIME_CONFIGS.values()
        )
        # vLLM has no mimo_v2 config class; it parses the Hub config code.
        assert {
            model_type
            for model_type, runtime in FAMILY_RUNTIME_CONFIGS.items()
            if runtime.vllm.trust_remote_code is True
        } == {"mimo_v2"}

    def test_nemotron_h_enables_prefix_caching(self) -> None:
        assert FAMILY_RUNTIME_CONFIGS["nemotron_h"].vllm.enable_prefix_caching is True

    def test_qwen3_5_lookup(self) -> None:
        runtime = FAMILY_RUNTIME_CONFIGS["qwen3_5"]
        assert runtime.vllm.mamba_cache_mode == "align"
        assert runtime.vllm.enable_prefix_caching is True
        vision = model_type_vision("qwen3_5")
        assert vision is not None
        assert vision.tower_connector_lora_targets == {
            "qkv",
            "linear_fc1",
            "linear_fc2",
        }
        assert runtime.patch.gdn is not None
        assert runtime.patch.gdn.clone_conv_output is True
        assert tuple(
            (entry.mixer, entry.text_model) for entry in runtime.patch.gdn.mixers
        ) == (
            (
                (
                    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe."
                    "Qwen3_5MoeGatedDeltaNet"
                ),
                (
                    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe."
                    "Qwen3_5MoeTextModel"
                ),
            ),
            (
                "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet",
                "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel",
            ),
        )

    def test_catalog_reasoning_parser_matches_model_info(self) -> None:
        catalog = {
            model_type: runtime.reasoning_parser
            for model_type, runtime in FAMILY_RUNTIME_CONFIGS.items()
            if runtime.reasoning_parser is not None
        }

        assert catalog == REASONING_PARSERS

    def test_every_supported_reasoning_model_names_its_parser(self) -> None:
        parsers = {
            info.config["model_type"]: FAMILY_RUNTIME_CONFIGS[
                info.config["model_type"]
            ].reasoning_parser
            for info in SUPPORTED_MODEL_INFO.values()
        }

        assert parsers == {
            "deepseek_v4": "deepseek_v4",
            "gemma4": "gemma4",
            "glm5_next": "glm47",
            "gpt_oss": "openai_gptoss",
            "granitemoe": None,
            "granitemoehybrid": None,
            "mimo_v2": "mimo",
            "nemotron_h": "nemotron_v3",
            "nemotron_h_omni": "nemotron_v3",
            "qwen2": None,
            "qwen3": "qwen3",
            "qwen3_5": "qwen3",
            "qwen3_5_moe": "qwen3",
        }

    @pytest.mark.parametrize("model_type", sorted(REASONING_PARSERS))
    def test_reasoning_parser_is_not_a_family_engine_kwarg(
        self, model_type: str
    ) -> None:
        kwargs = FAMILY_RUNTIME_CONFIGS[model_type].vllm.engine_kwargs()

        assert "reasoning_parser" not in kwargs

    def test_every_supported_model_names_a_registered_chat_format(
        self,
    ) -> None:
        model_types = {
            info.config["model_type"] for info in SUPPORTED_MODEL_INFO.values()
        }

        parsers = {
            model_type: FAMILY_RUNTIME_CONFIGS[model_type].chat_format
            for model_type in model_types
        }

        assert all(name in CHAT_FORMATS for name in parsers.values()), parsers

    def test_catalog_excludes_gemma_and_gemma2(self) -> None:
        assert "gemma" not in FAMILY_RUNTIME_CONFIGS
        assert "gemma2" not in FAMILY_RUNTIME_CONFIGS

    def test_nemotron_has_mamba_patches(self) -> None:
        patch = FAMILY_RUNTIME_CONFIGS["nemotron_h"].patch
        assert patch.install is install_nemotron_h_patches
        mamba = patch.mamba
        assert mamba is not None
        assert (
            mamba.mixer
            == "transformers.models.nemotron_h.modeling_nemotron_h.NemotronHMamba2Mixer"
        )
        assert mamba.fused_path is True
        assert mamba.stream_ordering is True


class TestVllmRuntimeConfigEngineKwargs:
    def test_dense_mamba_prefix_cache_retains_every_state(self) -> None:
        config = VllmRuntimeConfig(
            mamba_cache_mode="align", dense_mamba_prefix_cache=True
        )

        kwargs = config.engine_kwargs()

        assert kwargs == {
            "mamba_cache_mode": "align",
            "prefix_cache_retention_interval": None,
        }

    def test_unset_fields_add_no_kwargs(self) -> None:
        assert VllmRuntimeConfig(mamba_cache_mode="align").engine_kwargs() == {
            "mamba_cache_mode": "align"
        }

    @pytest.mark.parametrize("model_type", ["nemotron_h", "qwen3_5"])
    def test_hybrid_mamba_families_cache_densely(self, model_type: str) -> None:
        kwargs = FAMILY_RUNTIME_CONFIGS[model_type].vllm.engine_kwargs()

        assert kwargs["prefix_cache_retention_interval"] is None

    def test_sm80_attention_backend_is_not_an_engine_kwarg(self) -> None:
        config = VllmRuntimeConfig(sm80_attention_backend="FLASHINFER")

        assert config.engine_kwargs() == {}


class TestSm80AttentionBackend:
    @pytest.mark.parametrize(
        "hub_id",
        [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen3-4B",
            "ibm-granite/granite-3.1-3b-a800m-instruct",
            "ibm-granite/granite-4.0-micro",
        ],
    )
    def test_dense_attention_checkpoints_take_flashinfer(self, hub_id: str) -> None:
        assert sm80_attention_backend(hub_id) == "FLASHINFER"

    def test_qwen3_moe_takes_flashinfer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_config_model_type(monkeypatch, "qwen3_moe")

        assert sm80_attention_backend("Qwen/Qwen3-30B-A3B") == "FLASHINFER"

    @pytest.mark.parametrize(
        "hub_id",
        [
            "ibm-granite/granite-4.0-h-tiny",
            "google/gemma-4-E4B-it",
            "openai/gpt-oss-20b",
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.6-35B-A3B",
            "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        ],
    )
    def test_other_checkpoints_leave_the_backend_unset(self, hub_id: str) -> None:
        assert sm80_attention_backend(hub_id) is None

    def test_unknown_family_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_config_model_type(monkeypatch, "phi3")

        with pytest.raises(ValueError, match="'phi3' has no family"):
            sm80_attention_backend("microsoft/phi-3")


class TestFamilyMaxNumBatchedTokens:
    def test_arena_estimator_matches_the_catalog(self) -> None:
        catalog = {
            family: runtime.vllm.max_num_batched_tokens
            for family, runtime in FAMILY_RUNTIME_CONFIGS.items()
            if runtime.vllm.max_num_batched_tokens is not None
        }

        assert catalog == FAMILY_MAX_NUM_BATCHED_TOKENS


class TestFamilyTensorParallelPlans:
    def test_arena_estimator_matches_the_catalog(self) -> None:
        catalog = frozenset(
            family
            for family, runtime in FAMILY_RUNTIME_CONFIGS.items()
            if runtime.tensor_parallel_plan is not None
        )

        assert catalog == TENSOR_PARALLEL_MODEL_TYPES

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(
        ("model_id", "tensor_parallel"), TENSOR_PARALLEL_SPLIT_CASES
    )
    def test_arena_tensor_parallel_params_match_the_applied_plan(
        self, model_id: str, tensor_parallel: int
    ) -> None:
        # Arrange
        info = SUPPORTED_MODEL_INFO[model_id]
        arch = ModelArch.from_hf_config(info.config)
        expected = formulas.tensor_parallel_params(
            formulas.param_counts(arch, info.num_params), arch, tensor_parallel
        )
        if arch.tensor_parallel_tied_embedding:
            expected += formulas.token_embedding_params(arch)

        # Act
        splits = run_ranks(
            tensor_parallel_split_worker, tensor_parallel, model_id, timeout=600
        )

        # Assert
        assert [full for full, _ in splits] == [expected] * tensor_parallel
        assert sum(local for _, local in splits) == expected

    def test_tensor_parallel_plan_rejects_indivisible_query_heads(self) -> None:
        with pytest.raises(
            AssertionError, match=r"\(14\) must be divisible by tp \(4\)"
        ):
            run_ranks(tensor_parallel_split_worker, 4, "Qwen/Qwen2.5-0.5B-Instruct")

    def test_resolves_shared_nemotron_h_plan_once(self) -> None:
        plans = family_tensor_parallel_plans()

        assert plans == [
            DEEPSEEK_V4_TENSOR_PARALLEL_PLAN,
            GEMMA4_TENSOR_PARALLEL_PLAN,
            GLM5_NEXT_TENSOR_PARALLEL_PLAN,
            GPT_OSS_TENSOR_PARALLEL_PLAN,
            GRANITEMOE_TENSOR_PARALLEL_PLAN,
            GRANITEMOEHYBRID_TENSOR_PARALLEL_PLAN,
            MIMO_V2_TENSOR_PARALLEL_PLAN,
            NEMOTRON_H_TENSOR_PARALLEL_PLAN,
            QWEN3_TENSOR_PARALLEL_PLAN,
            QWEN3_5_TENSOR_PARALLEL_PLAN,
        ]

    def test_missing_plan_module_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        missing = ModelRuntimeConfig(
            name="missing_family",
            model_types=ModelTypes(),
            tensor_parallel_plan="agilerl.no_such_module:PLAN",
        )
        monkeypatch.setattr(architectures, "FAMILIES", (*FAMILIES, missing))

        with pytest.raises(ModuleNotFoundError, match=r"agilerl\.no_such_module"):
            family_tensor_parallel_plans()


class TestTensorParallelPlanDefaults:
    def test_restore_and_row_orders_default_to_no_ops(self) -> None:
        plan = TensorParallelPlan(shard=lambda _model, _mesh: 0)
        model = torch.nn.Linear(2, 2)
        weight = model.weight.detach().clone()

        restored = plan.restore(model, MagicMock())
        orders = plan.row_orders(model)

        assert restored is None
        assert orders == {}
        assert torch.equal(model.weight, weight)


class TestFamilyContextParallelPlans:
    def test_resolves_every_family_plan_in_family_order(self) -> None:
        plans = family_context_parallel_plans()

        assert plans == [
            DEEPSEEK_V4_CONTEXT_PARALLEL_PLAN,
            GLM5_NEXT_CONTEXT_PARALLEL_PLAN,
            GRANITEMOEHYBRID_CONTEXT_PARALLEL_PLAN,
            NEMOTRON_H_CONTEXT_PARALLEL_PLAN,
            QWEN3_5_CONTEXT_PARALLEL_PLAN,
        ]

    def test_missing_plan_module_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        missing = ModelRuntimeConfig(
            name="missing_family",
            model_types=ModelTypes(),
            context_parallel_plan="agilerl.no_such_module:PLAN",
        )
        monkeypatch.setattr(architectures, "FAMILIES", (*FAMILIES, missing))

        with pytest.raises(ModuleNotFoundError, match=r"agilerl\.no_such_module"):
            family_context_parallel_plans()


class TestFamilyRuntime:
    @pytest.mark.parametrize("model_type", ["nemotron_h", "nemotron_h_omni"])
    def test_nemotron_hub_id_uses_catalog(
        self, monkeypatch: pytest.MonkeyPatch, model_type: str
    ) -> None:
        stub_config_model_type(monkeypatch, model_type)
        config = family_runtime("nvidia/nemotron")
        assert config is FAMILY_RUNTIME_CONFIGS[model_type]

    @pytest.mark.parametrize("model_type", SWA_MODEL_TYPES)
    def test_swa_hub_id_uses_catalog(
        self, monkeypatch: pytest.MonkeyPatch, model_type: str
    ) -> None:
        stub_config_model_type(monkeypatch, model_type)
        config = family_runtime("google/gemma")
        assert config.trainer == FAMILY_RUNTIME_CONFIGS[model_type].trainer

    def test_gpt_oss_hub_id_uses_catalog(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_config_model_type(monkeypatch, "gpt_oss")
        config = family_runtime("openai/gpt-oss-20b")
        assert config.trainer == FAMILY_RUNTIME_CONFIGS["gpt_oss"].trainer

    def test_llama_hub_id_uses_its_family(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_config_model_type(monkeypatch, "llama")

        config = family_runtime("HuggingFaceTB/SmolLM2-135M-Instruct")

        assert config.name == "llama"

    @pytest.mark.parametrize("model_type", ["gemma", "gemma2", "phi3"])
    def test_unknown_hub_type_raises(
        self, monkeypatch: pytest.MonkeyPatch, model_type: str
    ) -> None:
        stub_config_model_type(monkeypatch, model_type)

        with pytest.raises(ValueError, match=f"'{model_type}' has no family"):
            family_runtime("some/model")

    def test_missing_config_json_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_missing(*args: object, **kwargs: object) -> None:
            msg = "missing config"
            raise OSError(msg)

        monkeypatch.setattr(
            "agilerl.architectures.catalog.PretrainedConfig.get_config_dict",
            raise_missing,
        )
        with pytest.raises(OSError, match="missing config"):
            family_runtime("nvidia/unlisted-model")

    @pytest.mark.parametrize(
        ("hub_id", "model_type"),
        [
            ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "nemotron_h"),
            ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "nemotron_h"),
            ("nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16", "nemotron_h_omni"),
        ],
    )
    def test_supported_id_resolves_from_bundle_without_hub(
        self, monkeypatch: pytest.MonkeyPatch, hub_id: str, model_type: str
    ) -> None:
        def raise_if_called(*args: object, **kwargs: object) -> None:
            msg = "hub lookup"
            raise AssertionError(msg)

        monkeypatch.setattr(
            "agilerl.architectures.catalog.PretrainedConfig.get_config_dict",
            raise_if_called,
        )

        config = family_runtime(hub_id)

        assert config is FAMILY_RUNTIME_CONFIGS[model_type]


class TestPretrainedModelType:
    def test_reads_config_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, object] = {}

        @classmethod
        def fake_get_config_dict(
            cls, pretrained_model_name_or_path: str, **kwargs: object
        ) -> tuple[dict[str, object], dict[str, object]]:
            seen.update(kwargs)
            return ({"model_type": "nemotron_h"}, {})

        monkeypatch.setattr(
            "agilerl.architectures.catalog.PretrainedConfig.get_config_dict",
            fake_get_config_dict,
        )
        assert pretrained_model_type("nvidia/unlisted-model") == "nemotron_h"
        assert "trust_remote_code" not in seen

    def test_missing_config_json_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_missing(*args: object, **kwargs: object) -> None:
            msg = "missing config"
            raise OSError(msg)

        monkeypatch.setattr(
            "agilerl.architectures.catalog.PretrainedConfig.get_config_dict",
            raise_missing,
        )
        with pytest.raises(OSError, match="missing config"):
            pretrained_model_type("nvidia/unlisted-model")

    def test_supported_id_reads_bundled_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_if_called(*args: object, **kwargs: object) -> None:
            msg = "hub lookup"
            raise AssertionError(msg)

        monkeypatch.setattr(
            "agilerl.architectures.catalog.PretrainedConfig.get_config_dict",
            raise_if_called,
        )

        assert pretrained_model_type("google/gemma-4-E4B-it") == "gemma4"


class TestRuntimeConfigsForbidExtra:
    def test_unknown_fields_are_rejected(self) -> None:
        from agilerl.architectures.runtime import (
            GdnMixerConfig,
            GdnPatchConfig,
            MambaPatchConfig,
            ModelRuntimeConfig,
            PatchRuntimeConfig,
            TrainerRuntimeConfig,
            VisionSupport,
            VllmRuntimeConfig,
        )

        cases = (
            (VllmRuntimeConfig, {}),
            (TrainerRuntimeConfig, {}),
            (MambaPatchConfig, {"mixer": "agilerl.architectures.nemotron_h.layers"}),
            (GdnMixerConfig, {"mixer": "pkg.GatedDeltaNet", "text_model": "pkg.Text"}),
            (
                GdnPatchConfig,
                {"mixers": [{"mixer": "pkg.GatedDeltaNet", "text_model": "pkg.Text"}]},
            ),
            (PatchRuntimeConfig, {}),
            (ModelRuntimeConfig, {"name": "family", "model_types": {}}),
            (ModelTypes, {}),
            (VisionSupport, {}),
        )
        for cls, payload in cases:
            with pytest.raises(ValidationError, match="extra_forbidden"):
                cls.model_validate({**payload, "__unknown__": True})


def write_family_package(root: Path, package: str, families: dict[str, str]) -> None:
    """Write ``package`` under ``root`` with one subpackage per ``families`` body."""
    (root / package).mkdir()
    (root / package / "__init__.py").write_text("")
    (root / package / "helpers.py").write_text("")
    for name, body in families.items():
        (root / package / name).mkdir()
        (root / package / name / "__init__.py").write_text(body)


def family_body(name: str) -> str:
    return (
        "from agilerl.architectures.runtime import ModelRuntimeConfig, ModelTypes\n"
        f"FAMILY = ModelRuntimeConfig(name={name!r}, "
        f"model_types=ModelTypes(text=({name!r},)))\n"
    )


class TestDiscoverFamilies:
    def test_returns_each_subpackage_family_in_name_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        write_family_package(
            tmp_path,
            "discover_ok",
            {"zeta": family_body("zeta"), "alpha": family_body("alpha")},
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        # Act
        families = discover_families("discover_ok")

        # Assert
        assert [family.name for family in families] == ["alpha", "zeta"]

    def test_subpackage_without_family_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_family_package(tmp_path, "discover_missing", {"bare": ""})
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises(
            TypeError, match=r"discover_missing\.bare must define FAMILY"
        ):
            discover_families("discover_missing")


class TestRuntimeConfigsByModelType:
    def test_maps_text_and_multimodal_types_to_their_family(self) -> None:
        family = ModelRuntimeConfig(
            name="vl", model_types=ModelTypes(text=("vl_text",), multimodal=("vl",))
        )

        configs = runtime_configs_by_model_type([family])

        assert configs == {"vl_text": family, "vl": family}

    def test_type_claimed_twice_raises(self) -> None:
        first = ModelRuntimeConfig(
            name="first", model_types=ModelTypes(text=("shared",))
        )
        second = ModelRuntimeConfig(
            name="second", model_types=ModelTypes(multimodal=("shared",))
        )

        with pytest.raises(ValueError, match="'shared' is claimed by first and second"):
            runtime_configs_by_model_type([first, second])


class TestTrainerModelClass:
    @pytest.mark.parametrize(
        ("hub_id", "expected"),
        [
            ("Qwen/Qwen3.5-2B", AutoModelForImageTextToText),
            ("Qwen/Qwen3.6-35B-A3B", AutoModelForImageTextToText),
            ("google/gemma-4-E4B-it", AutoModelForImageTextToText),
            (
                "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
                AutoModelForImageTextToText,
            ),
            ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", AutoModelForCausalLM),
            ("openai/gpt-oss-20b", AutoModelForCausalLM),
            ("zai-org/GLM-5.3-Flash", AutoModelForImageTextToText),
            ("deepseek-ai/DeepSeek-V4-Flash-0731", AutoModelForCausalLM),
            ("XiaomiMiMo/MiMo-V2.6-Flash-RL", AutoModelForImageTextToText),
        ],
    )
    def test_supported_ids_resolve_their_trainer_class(
        self, hub_id: str, expected: type
    ) -> None:
        assert trainer_model_class(hub_id) is expected

    def test_unknown_family_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_config_model_type(monkeypatch, "phi3")

        with pytest.raises(ValueError, match="'phi3' has no family"):
            trainer_model_class("some/model")


class TestFamilyChatTemplate:
    def test_deepseek_v4_resolves_the_dsml_template(self) -> None:
        template = family_chat_template("deepseek-ai/DeepSeek-V4-Flash-0731")

        assert template == DsmlFormat.CHAT_TEMPLATE

    def test_family_with_a_tokenizer_template_returns_none(self) -> None:
        assert family_chat_template("zai-org/GLM-5.3-Flash") is None

    def test_unknown_family_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_config_model_type(monkeypatch, "phi3")

        with pytest.raises(ValueError, match="'phi3' has no family"):
            family_chat_template("some/model")

    def test_only_dsml_families_carry_a_template(self) -> None:
        families = {
            model_type
            for model_type, runtime in FAMILY_RUNTIME_CONFIGS.items()
            if runtime.chat_format is not None
            and CHAT_FORMATS[runtime.chat_format].CHAT_TEMPLATE is not None
        }

        assert families == {"deepseek_v4"}


class TestNemotronHOmniVllmHfOverride:
    def test_native_config_gets_the_lora_architecture_and_vit_hidden_size(
        self,
    ) -> None:
        # Arrange
        config = NemotronH_Omni_Reasoning_V3_Config(vision_hidden_size=1024)

        # Act
        overridden = nemotron_h_omni_vllm_hf_override(config)

        # Assert
        assert overridden is config
        assert config.architectures == ["NemotronH_Super_Omni_Reasoning_V3"]
        assert config.vit_hidden_size == 1024

    def test_probe_config_without_a_vision_size_gets_only_the_architecture(
        self,
    ) -> None:
        config = PretrainedConfig()

        nemotron_h_omni_vllm_hf_override(config)

        assert config.architectures == ["NemotronH_Super_Omni_Reasoning_V3"]
        assert not hasattr(config, "vit_hidden_size")


class TestImportFallbacks:
    def test_reload_without_llm_dependencies_leaves_auto_classes_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(agilerl, "HAS_LLM_DEPENDENCIES", False)
        try:
            module = importlib.reload(runtime_module)

            assert module.AutoModelForCausalLM is None
            assert module.AutoModelForImageTextToText is None
            assert module.TrainerRuntimeConfig().config_class is None
        finally:
            monkeypatch.undo()
            importlib.reload(runtime_module)
