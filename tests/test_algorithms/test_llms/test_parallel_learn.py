# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CISPO learn under cp, tp and ep matches a same-world FSDP reference for every family."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers import NemotronHConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
from transformers.models.granitemoe.configuration_granitemoe import GraniteMoeConfig
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeForCausalLM
from transformers.models.granitemoehybrid.configuration_granitemoehybrid import (
    GraniteMoeHybridConfig,
)
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridForCausalLM,
)
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration,
)

from tests.rank_launch import run_ranks
from tests.test_algorithms.test_llms.parallel_learn import (
    SEED,
    FamilyCase,
    expert_weight,
    learn_worker,
    linear_weight,
)
from tests.test_architectures.gemma4.parallel_helpers import model_config

HIDDEN = 64
HEAD_DIM = 16
ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
DENSE_TARGETS = (*ATTENTION_TARGETS, "gate_proj", "up_proj", "down_proj")
QWEN3_5_TARGETS = (
    *DENSE_TARGETS,
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "qkv",
    "proj",
    "linear_fc1",
    "linear_fc2",
)
# fp32 cases differ from the reference only in reduction order across tp * cp ranks.
FP32_ATOL = 1e-5
FP32_RTOL = 1e-4
# flash_attention_2, and MoE expert GEMMs on SM80, have no fp32 path, so these
# cases compute in bf16. On A100, tp and ep move bf16 LoRA steps by up to 2.7%
# of the largest step.
BF16_ATOL = 1e-3
BF16_RTOL = 3e-2


def seeded(model: type[nn.Module], config: object) -> nn.Module:
    torch.manual_seed(SEED)
    return model(config)


def qwen3_model() -> nn.Module:
    config = Qwen3Config(
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        vocab_size=32,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "flash_attention_2"
    return seeded(Qwen3ForCausalLM, config)


def qwen2_model() -> nn.Module:
    config = Qwen2Config(
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "flash_attention_2"
    model = seeded(Qwen2ForCausalLM, config)
    assert model.model.layers[0].self_attn.q_proj.bias is not None
    return model


def dense_probes(model: nn.Module) -> dict[str, torch.Tensor]:
    layers = model.model.layers
    return {
        "attention.q_proj": linear_weight(layers[0].self_attn.q_proj),
        "attention.o_proj": linear_weight(layers[0].self_attn.o_proj),
        "mlp.gate_proj": linear_weight(layers[1].mlp.gate_proj),
        "mlp.down_proj": linear_weight(layers[1].mlp.down_proj),
    }


def gemma4_model() -> nn.Module:
    return seeded(Gemma4ForConditionalGeneration, model_config())


def gemma4_probes(model: nn.Module) -> dict[str, torch.Tensor]:
    language = model.model.language_model
    layers = language.layers
    return {
        "token_embedding": language.embed_tokens.weight,
        "per_layer_embedding": language.embed_tokens_per_layer.weight,
        "sliding.q_proj": linear_weight(layers[0].self_attn.q_proj),
        "full.o_proj": linear_weight(layers[1].self_attn.o_proj),
        "kv_shared.q_proj": linear_weight(layers[2].self_attn.q_proj),
        "mlp.gate_proj": linear_weight(layers[0].mlp.gate_proj),
        "mlp.down_proj": linear_weight(layers[1].mlp.down_proj),
    }


def gpt_oss_model() -> nn.Module:
    config = GptOssConfig(
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=32,
        sliding_window=4,
        layer_types=["sliding_attention", "full_attention"],
        attention_bias=True,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "flex_attention"
    return seeded(GptOssForCausalLM, config)


def gpt_oss_probes(model: nn.Module) -> dict[str, torch.Tensor]:
    layers = model.model.layers
    return {
        "attention.q_proj": linear_weight(layers[0].self_attn.q_proj),
        "attention.sinks": layers[0].self_attn.sinks,
        "experts.gate_up_proj": expert_weight(layers[1].mlp.experts, "gate_up_proj"),
    }


def granitemoe_model() -> nn.Module:
    config = GraniteMoeConfig(
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=32,
        attention_multiplier=0.25,
        embedding_multiplier=12.0,
        residual_multiplier=0.22,
        logits_scaling=6.0,
        tie_word_embeddings=True,
        pad_token_id=0,
    )
    config._attn_implementation = "flash_attention_2"
    return seeded(GraniteMoeForCausalLM, config)


def granitemoe_probes(model: nn.Module) -> dict[str, torch.Tensor]:
    layers = model.model.layers
    return {
        "token_embedding": model.model.embed_tokens.weight,
        "attention.q_proj": linear_weight(layers[0].self_attn.q_proj),
        "attention.k_proj": linear_weight(layers[0].self_attn.k_proj),
        "attention.o_proj": linear_weight(layers[1].self_attn.o_proj),
        "experts.gate_up_proj": expert_weight(
            layers[1].block_sparse_moe.experts, "gate_up_proj"
        ),
    }


def granitemoehybrid_model() -> nn.Module:
    config = GraniteMoeHybridConfig(
        hidden_size=HIDDEN,
        intermediate_size=32,
        shared_intermediate_size=64,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=32,
        position_embedding_type="nope",
        mamba_n_heads=8,
        mamba_d_head=16,
        mamba_n_groups=1,
        mamba_d_state=16,
        mamba_chunk_size=64,
        attention_multiplier=0.25,
        embedding_multiplier=12.0,
        residual_multiplier=0.22,
        logits_scaling=6.0,
        tie_word_embeddings=True,
        pad_token_id=0,
    )
    config._attn_implementation = "flash_attention_2"
    return seeded(GraniteMoeHybridForCausalLM, config)


def granitemoehybrid_probes(model: nn.Module) -> dict[str, torch.Tensor]:
    layers = model.model.layers
    return {
        "token_embedding": model.model.embed_tokens.weight,
        "mamba.in_proj": linear_weight(layers[0].mamba.in_proj),
        "mamba.out_proj": linear_weight(layers[0].mamba.out_proj),
        "attention.q_proj": linear_weight(layers[1].self_attn.q_proj),
        "shared_mlp.input_linear": linear_weight(layers[1].shared_mlp.input_linear),
        "experts.gate_up_proj": expert_weight(
            layers[1].block_sparse_moe.experts, "gate_up_proj"
        ),
    }


def nemotron_h_model() -> nn.Module:
    config = NemotronHConfig(
        hidden_size=HIDDEN,
        vocab_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        mamba_num_heads=8,
        mamba_head_dim=HEAD_DIM,
        n_groups=4,
        ssm_state_size=16,
        chunk_size=64,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=64,
        intermediate_size=64,
        layers_block_type=["linear_attention", "full_attention", "moe", "mlp"],
        tie_word_embeddings=False,
    )
    config._attn_implementation = "flash_attention_2"
    return seeded(NemotronHForCausalLM, config)


def nemotron_h_probes(model: nn.Module) -> dict[str, torch.Tensor]:
    layers = model.model.layers
    return {
        "mamba.in_proj": linear_weight(layers[0].mixer.in_proj),
        "attention.q_proj": linear_weight(layers[1].mixer.q_proj),
        "shared_experts.up_proj": linear_weight(layers[2].mixer.shared_experts.up_proj),
        "experts.up_proj": expert_weight(layers[2].mixer.experts, "up_proj"),
        "mlp.up_proj": linear_weight(layers[3].mixer.up_proj),
    }


def qwen3_5_model() -> nn.Module:
    text = Qwen3_5TextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        linear_num_key_heads=2,
        # One row per value head. A 2-row local GEMM does not match a 4-row GEMM.
        linear_num_value_heads=8,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        vocab_size=32,
        layer_types=["linear_attention", "full_attention"],
        partial_rotary_factor=1.0,
    )
    vision = Qwen3_5VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        depth=1,
        out_hidden_size=16,
        spatial_merge_size=1,
        patch_size=2,
        temporal_patch_size=1,
        num_position_embeddings=16,
    )
    for config in (text, vision):
        config._attn_implementation = "eager"
    config = Qwen3_5Config(
        text_config=text, vision_config=vision, tie_word_embeddings=False
    )
    config._attn_implementation = "eager"
    return seeded(Qwen3_5ForConditionalGeneration, config)


def qwen3_5_probes(model: nn.Module) -> dict[str, torch.Tensor]:
    text = model.model.language_model
    mlp = text.layers[0].mlp
    return {
        "linear_attn.in_proj_a": linear_weight(text.layers[0].linear_attn.in_proj_a),
        "visual.attn.qkv": linear_weight(model.model.visual.blocks[0].attn.qkv),
        "mlp.gate_proj": linear_weight(mlp.gate_proj),
        "mlp.up_proj": linear_weight(mlp.up_proj),
        "mlp.down_proj": linear_weight(mlp.down_proj),
    }


def qwen3_5_moe_model() -> nn.Module:
    text = Qwen3_5MoeTextConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        linear_num_key_heads=2,
        linear_num_value_heads=8,
        linear_key_head_dim=HEAD_DIM,
        linear_value_head_dim=HEAD_DIM,
        linear_conv_kernel_dim=4,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=32,
        layer_types=["linear_attention", "full_attention"],
    )
    vision = Qwen3_5MoeVisionConfig(
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_heads=4,
        depth=1,
        out_hidden_size=HIDDEN,
        spatial_merge_size=1,
        patch_size=2,
        temporal_patch_size=1,
        num_position_embeddings=16,
    )
    config = Qwen3_5MoeConfig(
        text_config=text, vision_config=vision, tie_word_embeddings=False
    )
    config._attn_implementation = "flash_attention_2"
    return seeded(Qwen3_5MoeForConditionalGeneration, config)


def qwen3_5_moe_probes(model: nn.Module) -> dict[str, torch.Tensor]:
    text = model.model.language_model
    vision = model.model.visual
    return {
        "linear_attn.in_proj_qkv": linear_weight(
            text.layers[0].linear_attn.in_proj_qkv
        ),
        "self_attn.q_proj": linear_weight(text.layers[1].self_attn.q_proj),
        "shared_expert.gate_proj": linear_weight(
            text.layers[0].mlp.shared_expert.gate_proj
        ),
        "visual.attn.qkv": linear_weight(vision.blocks[0].attn.qkv),
        "visual.mlp.linear_fc1": linear_weight(vision.blocks[0].mlp.linear_fc1),
        "merger.linear_fc1": linear_weight(vision.merger.linear_fc1),
        "experts.gate_up_proj": expert_weight(
            text.layers[0].mlp.experts, "gate_up_proj"
        ),
    }


FAMILY_CASES = {
    "qwen3": FamilyCase(
        model=qwen3_model,
        target_modules=DENSE_TARGETS,
        target_parameters=(),
        probes=dense_probes,
        param_dtype="bfloat16",
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
        meshes=((2, 2, 1, 1), (2, 1, 2, 1), (4, 2, 1, 1)),
    ),
    "qwen2": FamilyCase(
        model=qwen2_model,
        target_modules=DENSE_TARGETS,
        target_parameters=(),
        probes=dense_probes,
        param_dtype="bfloat16",
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
        meshes=((2, 2, 1, 1), (2, 1, 2, 1), (4, 2, 1, 1)),
    ),
    "gemma4": FamilyCase(
        model=gemma4_model,
        target_modules=DENSE_TARGETS,
        target_parameters=(),
        probes=gemma4_probes,
        param_dtype="float32",
        atol=FP32_ATOL,
        rtol=FP32_RTOL,
        meshes=((2, 2, 1, 1), (2, 1, 2, 1), (4, 2, 1, 1), (4, 2, 2, 1)),
        strip_towers=True,
    ),
    "gpt_oss": FamilyCase(
        model=gpt_oss_model,
        target_modules=ATTENTION_TARGETS,
        target_parameters=("mlp.experts.gate_up_proj", "mlp.experts.down_proj"),
        probes=gpt_oss_probes,
        param_dtype="bfloat16",
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
        meshes=((2, 1, 1, 2), (2, 1, 2, 2), (2, 2, 1, 1), (2, 2, 1, 2), (4, 2, 1, 1)),
    ),
    "granitemoe": FamilyCase(
        model=granitemoe_model,
        target_modules=ATTENTION_TARGETS,
        target_parameters=(
            "block_sparse_moe.experts.gate_up_proj",
            "block_sparse_moe.experts.down_proj",
        ),
        probes=granitemoe_probes,
        param_dtype="bfloat16",
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
        meshes=(
            (2, 1, 1, 2),
            (2, 2, 1, 1),
            (2, 2, 1, 2),
            (4, 2, 1, 1),
            (4, 1, 2, 2),
            (4, 2, 2, 2),
        ),
    ),
    "granitemoehybrid": FamilyCase(
        model=granitemoehybrid_model,
        target_modules=(
            *ATTENTION_TARGETS,
            "in_proj",
            "out_proj",
            "input_linear",
            "output_linear",
        ),
        target_parameters=(
            "block_sparse_moe.experts.gate_up_proj",
            "block_sparse_moe.experts.down_proj",
        ),
        probes=granitemoehybrid_probes,
        param_dtype="bfloat16",
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
        meshes=(
            (2, 1, 1, 2),
            (2, 2, 1, 1),
            (2, 2, 1, 2),
            (4, 2, 1, 1),
            (4, 1, 2, 2),
            (4, 2, 2, 2),
        ),
    ),
    "nemotron_h": FamilyCase(
        model=nemotron_h_model,
        target_modules=("in_proj", *ATTENTION_TARGETS, "up_proj", "down_proj"),
        target_parameters=("mixer.experts.up_proj", "mixer.experts.down_proj"),
        probes=nemotron_h_probes,
        param_dtype="bfloat16",
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
        meshes=((2, 1, 2, 2), (2, 2, 1, 1), (2, 1, 1, 2), (4, 2, 1, 1)),
    ),
    "qwen3_5": FamilyCase(
        model=qwen3_5_model,
        target_modules=QWEN3_5_TARGETS,
        target_parameters=(),
        probes=qwen3_5_probes,
        param_dtype="float32",
        atol=FP32_ATOL,
        rtol=FP32_RTOL,
        meshes=((2, 1, 2, 1),),
    ),
    "qwen3_5_moe": FamilyCase(
        model=qwen3_5_moe_model,
        target_modules=QWEN3_5_TARGETS,
        target_parameters=("mlp.experts.gate_up_proj", "mlp.experts.down_proj"),
        probes=qwen3_5_moe_probes,
        param_dtype="bfloat16",
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
        meshes=((2, 1, 2, 2), (2, 2, 1, 1), (2, 1, 1, 2), (2, 2, 1, 2), (4, 2, 1, 1)),
        strip_towers=True,
    ),
}
LEARN_CASES = [
    pytest.param(
        family,
        world,
        cp,
        tp,
        ep,
        id=f"{family}-world{world}-cp{cp}-tp{tp}-ep{ep}",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available() or torch.cuda.device_count() < world,
            reason=f"needs {world} CUDA devices",
        ),
    )
    for family, case in FAMILY_CASES.items()
    for world, cp, tp, ep in case.meshes
]


@pytest.mark.gpu
class TestLearnWorker:
    @pytest.mark.parametrize(("family", "world_size", "cp", "tp", "ep"), LEARN_CASES)
    def test_post_step_lora_matches_fsdp_reference(
        self, family: str, world_size: int, cp: int, tp: int, ep: int
    ) -> None:
        # The agent broadcasts CPU objects, so ranks join gloo for CPU and nccl for CUDA.
        run_ranks(
            learn_worker,
            world_size,
            FAMILY_CASES[family],
            cp,
            tp,
            ep,
            backend="cpu:gloo,cuda:nccl",
            timeout=900,
        )
