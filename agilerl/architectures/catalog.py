# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Family trainer, vLLM, and patch defaults keyed by Hugging Face ``model_type``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.architectures.deepseek_v4 import install_deepseek_v4_patches
from agilerl.architectures.gemma4 import install_gemma4_vision_patches
from agilerl.architectures.glm5_next import install_glm5_next_patches
from agilerl.architectures.granitemoe import install_granite_moe_patches
from agilerl.architectures.granitemoehybrid.packed import (
    install_granite_hybrid_patches,
)
from agilerl.architectures.mimo_v2 import MiMoV2Config, install_mimo_v2_native_training
from agilerl.architectures.nemotron_h.mamba import install_mamba_patches
from agilerl.architectures.qwen3_5.packed import install_qwen_gdn_patches
from agilerl.architectures.qwen3_5.vision import install_qwen3_5_vl_patches
from agilerl.architectures.runtime import (
    AutoModelForImageTextToText,
    GdnMixerConfig,
    GdnPatchConfig,
    MambaPatchConfig,
    ModelRuntimeConfig,
    PatchRuntimeConfig,
    TrainerModelClass,
    TrainerRuntimeConfig,
    VllmRuntimeConfig,
)
from agilerl.arena.models.model_info import (
    LORA_EXCLUDED_MODULES,
    REASONING_PARSERS,
    SUPPORTED_MODEL_INFO,
)
from agilerl.utils.patching import library_shim

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from transformers.configuration_utils import PretrainedConfig

NEMOTRON_H_RUNTIME_CONFIG = ModelRuntimeConfig(
    vllm=VllmRuntimeConfig(
        mamba_cache_mode="align",
        max_num_batched_tokens=8192,
        enable_prefix_caching=True,
        dense_mamba_prefix_cache=True,
    ),
    reasoning_parser=REASONING_PARSERS["nemotron_h"],
    patch=PatchRuntimeConfig(
        install=install_mamba_patches,
        mamba=MambaPatchConfig(
            mixer="transformers.models.nemotron_h.modeling_nemotron_h.NemotronHMamba2Mixer",
            block="transformers.models.nemotron_h.modeling_nemotron_h.NemotronHBlock",
        ),
    ),
    tensor_parallel_plan=(
        "agilerl.architectures.nemotron_h.tensor_parallel:"
        "NEMOTRON_H_TENSOR_PARALLEL_PLAN"
    ),
    context_parallel_plan=(
        "agilerl.architectures.nemotron_h.context_parallel:"
        "NEMOTRON_H_CONTEXT_PARALLEL_PLAN"
    ),
    chat_format="tool_call",
)


@library_shim("vllm", "==0.32.*")
def nemotron_h_omni_vllm_hf_override(config: object) -> object:
    """Point a native ``nemotron_h_omni`` config at NemotronHVLLoRA and restore ``vit_hidden_size``.

    :param config: Hugging Face config loaded from the checkpoint.
    :type config: object
    :return: The config vLLM's Nemotron VL model can build from.
    :rtype: object
    """
    fields = vars(config)
    # Serve through NemotronHVLLoRA, which adds tower and connector LoRA.
    fields["architectures"] = ["NemotronH_Super_Omni_Reasoning_V3"]
    # vLLM's Nemotron VL model reads the checkpoint's legacy name; the native
    # config stores it as vision_hidden_size. vLLM also probes with a dummy config.
    if "vision_hidden_size" in fields:
        fields["vit_hidden_size"] = fields["vision_hidden_size"]
    return config


NEMOTRON_H_OMNI_RUNTIME_CONFIG = NEMOTRON_H_RUNTIME_CONFIG.model_copy(
    update={
        # Native nemotron_h_omni; remote code stopped loading on 5.19.
        "trainer": TrainerRuntimeConfig(
            model_class=AutoModelForImageTextToText,
            trust_remote_code=False,
        ),
        "vllm": NEMOTRON_H_RUNTIME_CONFIG.vllm.model_copy(
            update={"hf_overrides": nemotron_h_omni_vllm_hf_override},
        ),
        "tower_connector_lora_targets": frozenset(
            {
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "fc1",
                "fc2",
                "patch_projection",
                "video_patch_projection",
                "linear_1",
                "linear_2",
            }
        ),
    },
)

GEMMA_SWA_TEXT_RUNTIME_CONFIG = ModelRuntimeConfig()

GEMMA_SWA_RUNTIME_CONFIG = ModelRuntimeConfig(
    trainer=TrainerRuntimeConfig(model_class=AutoModelForImageTextToText),
)

GEMMA4_TENSOR_PARALLEL_PLAN_PATH = (
    "agilerl.architectures.gemma4.tensor_parallel:GEMMA4_TENSOR_PARALLEL_PLAN"
)

GEMMA4_TEXT_RUNTIME_CONFIG = GEMMA_SWA_TEXT_RUNTIME_CONFIG.model_copy(
    update={
        "chat_format": "gemma4",
        "reasoning_parser": REASONING_PARSERS["gemma4_text"],
        "tensor_parallel_plan": GEMMA4_TENSOR_PARALLEL_PLAN_PATH,
    },
)

GEMMA4_RUNTIME_CONFIG = GEMMA_SWA_RUNTIME_CONFIG.model_copy(
    update={
        "chat_format": "gemma4",
        "reasoning_parser": REASONING_PARSERS["gemma4"],
        "tensor_parallel_plan": GEMMA4_TENSOR_PARALLEL_PLAN_PATH,
        "patch": PatchRuntimeConfig(install=install_gemma4_vision_patches),
        # vLLM swaps every tower linear for a LoRA layer and keeps HF names;
        # attention and MLP linears sit under clippable wrappers as ``linear``.
        "tower_connector_lora_targets": frozenset(
            {
                "linear",
                "input_proj",
                "embedding_projection",
                "input_proj_linear",
                "output_proj",
                "relative_k_proj",
            }
        ),
        "tower_modalities": {
            "vision_tower": frozenset({"image", "video"}),
            "embed_vision": frozenset({"image", "video"}),
            "audio_tower": frozenset({"audio"}),
            "embed_audio": frozenset({"audio"}),
        },
    },
)

GPT_OSS_RUNTIME_CONFIG = ModelRuntimeConfig(
    chat_format="harmony",
    tensor_parallel_plan=(
        "agilerl.architectures.gptoss.tensor_parallel:GPT_OSS_TENSOR_PARALLEL_PLAN"
    ),
    reasoning_parser=REASONING_PARSERS["gpt_oss"],
)

# Dense-attention families decode fastest with FlashInfer on sm80 (A100);
# checkpoints with Mamba layers opt out in sm80_attention_backend.
DENSE_ATTENTION_VLLM_CONFIG = VllmRuntimeConfig(sm80_attention_backend="FLASHINFER")
DENSE_TOOL_CALL_TAG_RUNTIME_CONFIG = ModelRuntimeConfig(
    vllm=DENSE_ATTENTION_VLLM_CONFIG,
    chat_format="tool_call",
)
QWEN2_RUNTIME_CONFIG = DENSE_TOOL_CALL_TAG_RUNTIME_CONFIG.model_copy(
    update={
        "tensor_parallel_plan": (
            "agilerl.architectures.qwen3.tensor_parallel:QWEN3_TENSOR_PARALLEL_PLAN"
        ),
    },
)
GRANITE3_RUNTIME_CONFIG = ModelRuntimeConfig(
    vllm=DENSE_ATTENTION_VLLM_CONFIG,
    patch=PatchRuntimeConfig(install=install_granite_moe_patches),
    chat_format="granite3",
    tensor_parallel_plan=(
        "agilerl.architectures.granitemoe.tensor_parallel:"
        "GRANITEMOE_TENSOR_PARALLEL_PLAN"
    ),
)
GRANITE_HYBRID_RUNTIME_CONFIG = DENSE_TOOL_CALL_TAG_RUNTIME_CONFIG.model_copy(
    update={
        "patch": PatchRuntimeConfig(install=install_granite_hybrid_patches),
        "context_parallel_plan": (
            "agilerl.architectures.granitemoehybrid.context_parallel:"
            "GRANITEMOEHYBRID_CONTEXT_PARALLEL_PLAN"
        ),
        "tensor_parallel_plan": (
            "agilerl.architectures.granitemoehybrid.tensor_parallel:"
            "GRANITEMOEHYBRID_TENSOR_PARALLEL_PLAN"
        ),
    },
)
QWEN3_MOE_RUNTIME_CONFIG = DENSE_TOOL_CALL_TAG_RUNTIME_CONFIG.model_copy(
    update={"reasoning_parser": REASONING_PARSERS["qwen3"]},
)
QWEN3_RUNTIME_CONFIG = QWEN3_MOE_RUNTIME_CONFIG.model_copy(
    update={
        "tensor_parallel_plan": (
            "agilerl.architectures.qwen3.tensor_parallel:QWEN3_TENSOR_PARALLEL_PLAN"
        ),
    },
)


# Linear-attention rejects mamba_cache_mode "all"; align needs prefix caching.
QWEN3_5_TEXT_RUNTIME_CONFIG = ModelRuntimeConfig(
    vllm=VllmRuntimeConfig(
        mamba_cache_mode="align",
        enable_prefix_caching=True,
        dense_mamba_prefix_cache=True,
    ),
    patch=PatchRuntimeConfig(
        install=install_qwen_gdn_patches,
        gdn=GdnPatchConfig(
            mixers=(
                GdnMixerConfig(
                    mixer=(
                        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe."
                        "Qwen3_5MoeGatedDeltaNet"
                    ),
                    text_model=(
                        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe."
                        "Qwen3_5MoeTextModel"
                    ),
                ),
                GdnMixerConfig(
                    mixer=(
                        "transformers.models.qwen3_5.modeling_qwen3_5."
                        "Qwen3_5GatedDeltaNet"
                    ),
                    text_model=(
                        "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel"
                    ),
                ),
            ),
        ),
    ),
    tensor_parallel_plan=(
        "agilerl.architectures.qwen3_5.tensor_parallel:QWEN3_5_TENSOR_PARALLEL_PLAN"
    ),
    context_parallel_plan=(
        "agilerl.architectures.qwen3_5.context_parallel:QWEN3_5_CONTEXT_PARALLEL_PLAN"
    ),
    chat_format="tool_call",
    reasoning_parser=REASONING_PARSERS["qwen3_5_text"],
)

# AutoModelForCausalLM maps qwen3_5 to a text-only class that drops model.visual.
QWEN3_5_RUNTIME_CONFIG = QWEN3_5_TEXT_RUNTIME_CONFIG.model_copy(
    update={
        "trainer": TrainerRuntimeConfig(model_class=AutoModelForImageTextToText),
        "patch": QWEN3_5_TEXT_RUNTIME_CONFIG.patch.model_copy(
            update={"install": install_qwen3_5_vl_patches}
        ),
        # The patch embedding ``proj`` is a Conv3d that vLLM serves without LoRA.
        "tower_connector_lora_targets": frozenset({"qkv", "linear_fc1", "linear_fc2"}),
        "tower_modalities": {"visual": frozenset({"image", "video"})},
    },
)

DEEPSEEK_V4_RUNTIME_CONFIG = ModelRuntimeConfig(
    chat_format="dsml",
    reasoning_parser=REASONING_PARSERS["deepseek_v4"],
    lora_exclude_modules=LORA_EXCLUDED_MODULES["deepseek_v4"],
    patch=PatchRuntimeConfig(install=install_deepseek_v4_patches),
    tensor_parallel_plan=(
        "agilerl.architectures.deepseek_v4.tensor_parallel:"
        "DEEPSEEK_V4_TENSOR_PARALLEL_PLAN"
    ),
    context_parallel_plan=(
        "agilerl.architectures.deepseek_v4.context_parallel:"
        "DEEPSEEK_V4_CONTEXT_PARALLEL_PLAN"
    ),
)

GLM5_NEXT_TEXT_RUNTIME_CONFIG = ModelRuntimeConfig(
    chat_format="glm47",
    reasoning_parser=REASONING_PARSERS["glm5_next_text"],
    lora_exclude_modules=LORA_EXCLUDED_MODULES["glm5_next_text"],
    patch=PatchRuntimeConfig(install=install_glm5_next_patches),
    tensor_parallel_plan=(
        "agilerl.architectures.glm5_next.tensor_parallel:GLM5_NEXT_TENSOR_PARALLEL_PLAN"
    ),
    context_parallel_plan=(
        "agilerl.architectures.glm5_next.context_parallel:"
        "GLM5_NEXT_CONTEXT_PARALLEL_PLAN"
    ),
)

GLM5_NEXT_RUNTIME_CONFIG = GLM5_NEXT_TEXT_RUNTIME_CONFIG.model_copy(
    update={
        "trainer": TrainerRuntimeConfig(model_class=AutoModelForImageTextToText),
        # The patch embedding ``proj`` is a Conv3d that vLLM serves without LoRA.
        "tower_connector_lora_targets": frozenset(
            {"qkv", "gate_proj", "up_proj", "down_proj"}
        ),
        "tower_modalities": {"visual": frozenset({"image", "video"})},
    },
)

MIMO_V2_RUNTIME_CONFIG = ModelRuntimeConfig(
    # The Hub auto_map names repo code: vLLM parses it, the trainer loads the
    # config with the native class.
    vllm=VllmRuntimeConfig(trust_remote_code=True),
    trainer=TrainerRuntimeConfig(trust_remote_code=False, config_class=MiMoV2Config),
    patch=PatchRuntimeConfig(install=install_mimo_v2_native_training),
    tensor_parallel_plan=(
        "agilerl.architectures.mimo_v2.tensor_parallel:MIMO_V2_TENSOR_PARALLEL_PLAN"
    ),
    chat_format="tool_call",
    reasoning_parser=REASONING_PARSERS["mimo_v2"],
    lora_exclude_modules=LORA_EXCLUDED_MODULES["mimo_v2"],
)

FAMILY_RUNTIME_CONFIGS: Mapping[str, ModelRuntimeConfig] = {
    "deepseek_v4": DEEPSEEK_V4_RUNTIME_CONFIG,
    "qwen2": QWEN2_RUNTIME_CONFIG,
    "qwen3": QWEN3_RUNTIME_CONFIG,
    "qwen3_moe": QWEN3_MOE_RUNTIME_CONFIG,
    "granitemoe": GRANITE3_RUNTIME_CONFIG,
    "granitemoehybrid": GRANITE_HYBRID_RUNTIME_CONFIG,
    "nemotron_h": NEMOTRON_H_RUNTIME_CONFIG,
    "nemotron_h_omni": NEMOTRON_H_OMNI_RUNTIME_CONFIG,
    "gemma3": GEMMA_SWA_RUNTIME_CONFIG,
    "gemma3_text": GEMMA_SWA_TEXT_RUNTIME_CONFIG,
    "gemma4": GEMMA4_RUNTIME_CONFIG,
    "gemma4_text": GEMMA4_TEXT_RUNTIME_CONFIG,
    "glm5_next": GLM5_NEXT_RUNTIME_CONFIG,
    "glm5_next_text": GLM5_NEXT_TEXT_RUNTIME_CONFIG,
    "gpt_oss": GPT_OSS_RUNTIME_CONFIG,
    "mimo_v2": MIMO_V2_RUNTIME_CONFIG,
    "qwen3_5": QWEN3_5_RUNTIME_CONFIG,
    "qwen3_5_text": QWEN3_5_TEXT_RUNTIME_CONFIG,
    "qwen3_5_moe": QWEN3_5_RUNTIME_CONFIG,
    "qwen3_5_moe_text": QWEN3_5_TEXT_RUNTIME_CONFIG,
}


def pretrained_config(model_name_or_path: str) -> dict[str, Any]:
    """Return ``config.json`` contents from a checkpoint id or local path.

    Supported ids read the bundled config instead of the Hub.

    :param model_name_or_path: Hugging Face model id or local path.
    :type model_name_or_path: str
    :return: The checkpoint's top-level config.
    :rtype: dict[str, Any]
    """
    entry = SUPPORTED_MODEL_INFO.get(model_name_or_path)
    if entry is not None:
        return entry.config
    config_dict, _ = PretrainedConfig.get_config_dict(model_name_or_path)
    return config_dict


def pretrained_model_type(model_name_or_path: str) -> str:
    """Return Hugging Face ``model_type`` from a checkpoint id or local path.

    Supported ids read the bundled config instead of the Hub.
    """
    return pretrained_config(model_name_or_path)["model_type"]


def family_runtime(model_name_or_path: str) -> ModelRuntimeConfig:
    """Return catalog runtime for a checkpoint id or local path."""
    return FAMILY_RUNTIME_CONFIGS.get(
        pretrained_model_type(model_name_or_path),
        ModelRuntimeConfig(),
    )


def sm80_attention_backend(model_name_or_path: str) -> str | None:
    """Return the async engine attention backend for this checkpoint on sm80 GPUs.

    :param model_name_or_path: Hugging Face model id or local path.
    :type model_name_or_path: str
    :return: The family's ``sm80_attention_backend``, or ``None`` when unset or
        the checkpoint has Mamba layers.
    :rtype: str | None
    """
    config = pretrained_config(model_name_or_path)
    # granitemoehybrid checkpoints range from attention-only to mostly Mamba.
    if "mamba" in config.get("layer_types", ()):
        return None
    runtime = FAMILY_RUNTIME_CONFIGS.get(config["model_type"], ModelRuntimeConfig())
    return runtime.vllm.sm80_attention_backend


def trainer_model_class(model_name_or_path: str) -> TrainerModelClass:
    """Return the auto class every trainer load of this checkpoint builds.

    :param model_name_or_path: Hugging Face model id or local path.
    :type model_name_or_path: str
    :return: ``AutoModelForCausalLM``, or ``AutoModelForImageTextToText`` for VL families.
    :rtype: TrainerModelClass
    """
    return family_runtime(model_name_or_path).trainer.model_class


def family_chat_template(model_name_or_path: str) -> str | None:
    """Return the family chat format's template for a tokenizer that ships none.

    :param model_name_or_path: Hugging Face model id or local path.
    :type model_name_or_path: str
    :return: The Jinja template, or ``None`` when the format has none.
    :rtype: str | None
    """
    # Import cycle: agilerl.llm_envs imports agilerl.utils.llm_utils, which
    # imports this package.
    from agilerl.llm_envs.chat_formats import CHAT_FORMATS

    chat_format = family_runtime(model_name_or_path).chat_format
    if chat_format is None:
        return None
    return CHAT_FORMATS[chat_format].CHAT_TEMPLATE
