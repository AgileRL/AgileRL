# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Formula-level tests: golden numbers on known geometries."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from agilerl.arena.memory import (
    block_activations,
    formulas,
    fsdp_gather,
    kv_cache,
    lora_formulas,
)
from agilerl.arena.memory.lora_placement import (
    lora_tensor_placement,
    split_moe_lora_recompute_bytes,
)
from agilerl.arena.memory.nccl import nccl_on_node_fraction, trainer_nccl_bytes
from agilerl.arena.memory.resolve.attention import FAMILY_ATTN_IMPLEMENTATIONS
from agilerl.arena.memory.resolve.chunk import resolve_chunk_rows
from agilerl.arena.memory.specs import (
    GiB,
    MiB,
    ModelArch,
    TrainingSettings,
    WeightVariant,
    has_routed_experts,
)
from agilerl.arena.models.fsdp import FSDPConfig
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO

ASSETS = Path(__file__).parent / "assets"

DSV4_FLASH = "deepseek-ai/DeepSeek-V4-Flash-0731"
GLM_FLASH = "zai-org/GLM-5.3-Flash"
MIMO_FLASH = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
PREQUANTIZED_CHECKPOINTS = (DSV4_FLASH, GLM_FLASH, MIMO_FLASH)

# Qwen2.5-0.5B-Instruct geometry (published config.json values).
QWEN_05B = ModelArch(
    n_layers=24,
    hidden_size=896,
    intermediate_size=4864,
    n_heads=14,
    n_kv_heads=2,
    head_dim=64,
    vocab_size=151936,
    tied_embeddings=True,
    attn_bias=True,
)


NEMOTRON_H = {
    "num_hidden_layers": 56,
    "hidden_size": 4480,
    "intermediate_size": 15680,
    "num_attention_heads": 40,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 131072,
    "hybrid_override_pattern": "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-",
    "ssm_state_size": 128,
    "conv_kernel": 4,
    "mamba_num_groups": 8,
    "mamba_num_heads": 128,
    "mamba_head_dim": 80,
    "mamba_ssm_cache_dtype": "float32",
}
# Falcon-H1 runs attention and SSM in parallel in every block, so both counts are n_layers.
FALCON_H1 = {
    "num_hidden_layers": 24,
    "hidden_size": 2048,
    "intermediate_size": 4608,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "vocab_size": 65537,
    "mamba_d_state": 256,
    "mamba_d_conv": 4,
    "mamba_n_groups": 1,
    "mamba_n_heads": 48,
    "mamba_d_head": 64,
    "mamba_d_ssm": 3072,
}

DECODER_TINY = {
    "num_hidden_layers": 8,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 256,
    "tie_word_embeddings": True,
}

# GLM-5.3-Flash shape: KDA and MLA layers alternate, a dense first FFN, a DSA
# indexer on each MLA layer, and two hyper-connection streams.
GLM_TINY = {
    "model_type": "glm5_next_text",
    "num_hidden_layers": 4,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "vocab_size": 256,
    "q_lora_rank": 32,
    "kv_lora_rank": 16,
    "qk_nope_head_dim": 8,
    "qk_rope_head_dim": 0,
    "v_head_dim": 8,
    "layer_types": ["linear_attention", "deepseek_sparse_attention"] * 2,
    "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 16,
    "n_shared_experts": 1,
    "index_n_heads": 2,
    "index_head_dim": 8,
    "indexer_types": ["full"] * 4,
    "hc_mult": 2,
    "linear_attn_config": {"num_heads": 2, "head_dim": 8, "short_conv_kernel_size": 4},
}
# DeepSeek-V4-Flash shape: shared-KV MQA with a low-rank query, a grouped
# output, one sliding, one CSA and one HCA layer (the fourth ratio is MTP's).
DSV4_TINY = {
    "model_type": "deepseek_v4",
    "num_hidden_layers": 3,
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "qk_rope_head_dim": 4,
    "q_lora_rank": 32,
    "o_groups": 2,
    "o_lora_rank": 8,
    "vocab_size": 256,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 16,
    "n_shared_experts": 1,
    "index_n_heads": 2,
    "index_head_dim": 8,
    "hc_mult": 2,
    "sliding_window": 8,
    "compress_ratios": [0, 4, 128, 0],
}
# MiMo-V2-Flash shape: a dense full-attention layer 0 with one KV head, two
# MoE sliding-window layers with two, values narrower than keys, and a vision
# tower the text-only classes drop.
MIMO_TINY = {
    "model_type": "mimo_v2",
    "num_hidden_layers": 3,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "swa_num_key_value_heads": 2,
    "head_dim": 16,
    "v_head_dim": 8,
    "hybrid_layer_pattern": [0, 1, 1],
    "moe_layer_freq": [0, 1, 1],
    "sliding_window": 8,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 16,
    "n_shared_experts": None,
    "vocab_size": 256,
    "vision_config": {"depth": 2, "hidden_size": 32, "num_heads": 2},
}


class TestHostBytesPerGpu:
    def test_node_ram_less_the_reserve_splits_evenly(self):
        assert formulas.host_bytes_per_gpu(165 * GiB, 4) == 157 * GiB // 4

    def test_node_under_the_reserve_leaves_nothing(self):
        assert formulas.host_bytes_per_gpu(4 * GiB, 1) == 0


class TestAttentionOutputGateParams:
    def test_biased_gate_counts_weight_and_bias_rows(self):
        # Arrange
        gated = QWEN_05B.model_copy(update={"attention_output_gate": True})
        counts = replace(formulas.param_counts(gated), unattributed=10**9)

        # Act
        params = formulas.attention_output_gate_params(counts, gated)

        # Assert: 24 layers * 14 heads * 64 head_dim rows, by 896 inputs plus a bias.
        assert params == 24 * 14 * 64 * 896 + 24 * 14 * 64


class TestPromptLogprobBytes:
    def test_holds_two_chunks_of_fp32_full_vocab_logits(self) -> None:
        assert formulas.prompt_logprob_bytes(QWEN_05B) == 2 * 1024 * 151936 * 4


class TestParamCounts:
    def test_repeat_and_equal_arch_match_a_fresh_count(self):
        # Arrange
        rebuilt = ModelArch.model_validate(QWEN_05B.model_dump())

        # Act
        first = formulas.param_counts(QWEN_05B, 494_032_768)
        again = formulas.param_counts(rebuilt, 494_032_768)
        formulas.param_counts.cache_clear()
        fresh = formulas.param_counts(QWEN_05B, 494_032_768)

        # Assert
        assert first == again == fresh
        assert formulas.param_counts(QWEN_05B) == replace(fresh, unattributed=0)

    def test_matches_published_total(self):
        counts = formulas.param_counts(QWEN_05B)
        # Qwen2.5-0.5B has ~494M parameters.
        assert counts.total == pytest.approx(494_000_000, rel=0.01)
        assert counts.lm_head == 0  # tied embeddings

    def test_untied_adds_lm_head(self):
        untied = QWEN_05B.model_copy(update={"tied_embeddings": False})
        delta = (
            formulas.param_counts(untied).total - formulas.param_counts(QWEN_05B).total
        )
        assert delta == QWEN_05B.vocab_size * QWEN_05B.hidden_size

    @pytest.mark.parametrize(
        "config", [NEMOTRON_H, FALCON_H1], ids=["nemotron-h", "falcon-h1"]
    )
    def test_hybrid_mamba_params_are_counted(self, config):
        counts = formulas.param_counts(ModelArch.from_hf_config(config))
        assert counts.mamba > 0

    def test_nemotron_h_mamba_param_share(self):
        counts = formulas.param_counts(ModelArch.from_hf_config(NEMOTRON_H))

        # ~4.0B Mamba params → ~7.4 GiB bf16.
        assert counts.mamba == pytest.approx(4.0e9, rel=0.05)
        assert counts.mamba * 2 == pytest.approx(7.4 * GiB, rel=0.05)

    def test_without_checkpoint_total_attributes_nothing(self):
        assert formulas.param_counts(QWEN_05B).unattributed == 0

    def test_reconciles_to_the_checkpoint_total(self):
        analytic = formulas.param_counts(QWEN_05B)
        exact = formulas.param_counts(QWEN_05B, analytic.total + 1_000_000)
        assert exact.unattributed == 1_000_000
        assert exact.total == analytic.total + 1_000_000

    def test_reconciles_when_geometry_over_counts(self):
        analytic = formulas.param_counts(QWEN_05B)
        exact = formulas.param_counts(QWEN_05B, analytic.total - 5_000)
        assert exact.unattributed == -5_000
        assert exact.total == analytic.total - 5_000

    def test_mla_and_kda_mix_matches_the_hf_build(self):
        # Arrange
        arch = ModelArch.from_hf_config(GLM_TINY)

        # Act
        counts = formulas.param_counts(arch)

        # Assert: an MLA layer holds q_a 64x32, q_b 32x32, kv_a 64x16, kv_b
        # 16x64, o 32x64 and two latent norms (7216); its indexer wq_b 32x16,
        # weights 64x2, wk and key-pool gate 64x8 each, LayerNorm 2x8 (1680).
        assert counts.attention == 2 * (7216 + 1680)
        # A KDA layer: q/k/v 64x16, gates 64x8 + 8x16 twice, beta 64x2, o
        # 16x64, conv 48x4, dt bias 16, A_log 2, output norm 8.
        assert counts.mamba == 2 * 5722
        # One dense FFN layer; three MoE layers with a router each.
        assert counts.mlp == 3 * 64 * 96 + 3 * 4 * 64
        assert counts.moe_experts == 3 * (4 * 3 * 64 * 16 + 3 * 64 * 16)
        # Two norms per layer, the final norm, and two hyper-connection
        # mixers per layer of an (8, 128) weight, 8 biases and 3 scales.
        assert counts.norms == 4 * 2 * 64 + 64 + 4 * 2 * (8 * 128 + 8 + 3)
        # The HF build adds the indexer's (4, 8) key-pool position table per layer.
        assert counts.total == 136_204 - 2 * 32

    def test_compressed_attention_matches_the_hf_build(self):
        # Arrange
        arch = ModelArch.from_hf_config(DSV4_TINY)

        # Act
        counts = formulas.param_counts(arch)

        # Assert: per layer q_a 64x32, q_b 32x64, kv 64x16, grouped o_a
        # 2x(32->8), o_b 16x64, two norms and four sinks (6708); a CSA
        # compressor (4240), an HCA compressor (4112), and the CSA layer's
        # indexer (2760).
        assert counts.attention == 3 * 6708 + 4240 + 4112 + 2760
        assert counts.mlp == 3 * 4 * 64
        assert counts.norms == 3 * 2 * 64 + 64 + 3 * 2 * (8 * 128 + 8 + 3)
        # The HF build adds the hyper-connection head.
        assert counts.total == 117_769 - (2 * 128 + 2 + 1)

    def test_single_stream_adds_no_hyper_connection_params(self):
        # Arrange
        arch = ModelArch.from_hf_config(GLM_TINY)
        single = arch.model_copy(update={"hc_mult": 1})

        # Act
        delta = formulas.param_counts(arch).norms - formulas.param_counts(single).norms

        # Assert
        assert delta == 4 * 2 * (8 * 128 + 8 + 3)

    def test_per_layer_kv_heads_and_narrow_values_match_the_hf_build(self):
        # Arrange
        arch = ModelArch.from_hf_config(MIMO_TINY)

        # Act
        counts = formulas.param_counts(arch)

        # Assert: q 64x64, k 64x16, v 64x8 and o 32x64 on the full layer;
        # k 64x32 and v 64x16 on each sliding layer.
        assert counts.attention == (4096 + 1024 + 512 + 2048) + 2 * (
            4096 + 2048 + 1024 + 2048
        )
        # One dense FFN layer; two MoE layers with a router each.
        assert counts.mlp == 3 * 64 * 96 + 2 * 4 * 64
        assert counts.moe_experts == 2 * 4 * 3 * 64 * 16
        assert counts.multimodal_towers == 0
        # The native build adds one sink logit per head on each sliding layer.
        assert counts.total == 102_856 - 2 * 4


class TestContextParallelShardLen:
    def test_attention_only_pads_the_action_frame_to_cp(self):
        assert formulas.context_parallel_shard_len(QWEN_05B, 4100, 4) == 1025

    def test_gated_delta_net_pads_to_scan_chunks_per_rank(self):
        arch = ModelArch.from_hf_config(QWEN35_08B.config)

        # 4099 action tokens pad to 33 chunks of 2 * 64.
        assert formulas.context_parallel_shard_len(arch, 4100, 2) == 33 * 64

    def test_flex_attention_pads_to_whole_blocks(self):
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["openai/gpt-oss-20b"].config
        )

        assert arch.flex_attention
        assert formulas.context_parallel_shard_len(arch, 4100, 4) == 33 * 128 // 4

    def test_one_rank_runs_the_whole_row(self):
        assert formulas.context_parallel_shard_len(QWEN_05B, 4100, 1) == 4100


class TestModelArchFromHfConfig:
    def test_moe_only_config_reads_expert_and_shared_expert_widths(self):
        # Arrange: Qwen3.6 has no dense intermediate_size; every layer is routed
        # experts plus one shared expert sized by shared_expert_intermediate_size.
        info = SUPPORTED_MODEL_INFO["Qwen/Qwen3.6-35B-A3B"]

        # Act
        arch = ModelArch.from_hf_config(info.config)
        counts = formulas.param_counts(arch, info.num_params)

        # Assert
        assert arch.n_experts == 256
        assert arch.expert_intermediate_size == 512
        assert arch.intermediate_size == 512
        assert arch.n_shared_experts == 1
        assert arch.shared_expert_intermediate_size == 512
        assert abs(counts.unattributed) / counts.total < 0.02

    def test_expert_pattern_layers_carry_the_routed_experts(self):
        # Arrange: Lightning's hybrid_override_pattern has 23 M, 6 * and 23 E.
        info = SUPPORTED_MODEL_INFO["nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"]

        # Act
        arch = ModelArch.from_hf_config(info.config)
        counts = formulas.param_counts(arch, info.num_params)

        # Assert
        assert arch.moe_layers == 23
        assert arch.mlp_layers == 0
        assert abs(counts.unattributed) / counts.total < 0.02

    def test_linear_attention_layers_are_recurrent_not_attention(self):
        # Arrange: Qwen3.8-27B lists 48 linear_attention (gated delta net) and
        # 16 full_attention layers; only the full ones hold a KV cache.
        config = SUPPORTED_MODEL_INFO["Qwen/Qwen3.8-27B"].config

        # Act
        arch = ModelArch.from_hf_config(config)

        # Assert
        assert arch.attention_layers == 16
        assert arch.n_mamba_layers == 48

    def test_transformers_layer_type_names_read_like_the_hub_ones(self):
        # ``mamba`` / ``attention`` and ``linear_attention`` / ``full_attention``
        # are the same layer kinds.
        hub = {**NEMOTRON_H_MOE, "layers_block_type": ["mamba", "moe", "attention"]}
        renamed = {
            **NEMOTRON_H_MOE,
            "layers_block_type": ["linear_attention", "moe", "full_attention"],
        }

        hub_arch = ModelArch.from_hf_config(hub)
        renamed_arch = ModelArch.from_hf_config(renamed)

        assert renamed_arch.layer_kinds == ("mamba", "moe", "attention")
        assert renamed_arch.layer_kinds == hub_arch.layer_kinds
        assert renamed_arch.attention_layers == hub_arch.attention_layers == 1

    @pytest.mark.parametrize(
        ("model_type", "eager"),
        [
            ("eager_family", True),
            ("deepseek_v4", False),
            ("glm5_next", False),
            ("qwen3", False),
        ],
    )
    def test_eager_attention_follows_the_family_backend(
        self, monkeypatch, model_type, eager
    ):
        monkeypatch.setitem(FAMILY_ATTN_IMPLEMENTATIONS, "eager_family", "eager")

        arch = ModelArch.from_hf_config({**DECODER_TINY, "model_type": model_type})

        assert arch.eager_attention is eager

    def test_tiny_llm_asset(self):
        config = json.loads((ASSETS / "tiny_llm" / "config.json").read_text())
        arch = ModelArch.from_hf_config(config)
        assert arch.n_layers == config["num_hidden_layers"]
        assert arch.hidden_size == config["hidden_size"]
        assert arch.vocab_size == config["vocab_size"]
        assert formulas.param_counts(arch).total > 0

    def test_hybrid_attention_layer_fraction_from_layer_types(self):
        # Hybrid attention: most layers windowed, a few global.
        arch = ModelArch.from_hf_config(
            {
                "num_hidden_layers": 35,
                "hidden_size": 1536,
                "intermediate_size": 6144,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "head_dim": 256,
                "vocab_size": 262144,
                "tie_word_embeddings": True,
                "sliding_window": 512,
                "layer_types": ["sliding_attention"] * 28 + ["full_attention"] * 7,
            }
        )
        assert arch.sliding_window == 512
        assert arch.sliding_window_layer_fraction == pytest.approx(0.8)
        # head_dim is read from the config (hidden // heads is 192).
        assert arch.head_dim == 256
        assert arch.n_kv_heads == 1  # MQA

        # Windowed layers cap KV growth at long context.
        windowed = kv_cache.kv_cache_demand_bytes(arch, "bf16", 8, 8192, 0)
        globl = kv_cache.kv_cache_demand_bytes(
            arch.model_copy(update={"sliding_window": None}), "bf16", 8, 8192, 0
        )
        assert windowed < globl / 3

    def test_sliding_window_pattern_integer_form(self):
        base = {
            "num_hidden_layers": 24,
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "vocab_size": 151936,
            "sliding_window": 4096,
            "sliding_window_pattern": 6,
        }
        arch = ModelArch.from_hf_config(base)
        assert arch.sliding_window_layer_fraction == pytest.approx(5 / 6)

    def test_hybrid_layer_mix_is_read_from_every_config_spelling(self):
        nemotron = ModelArch.from_hf_config(NEMOTRON_H)
        # 4 of 56 layers are attention.
        assert (nemotron.attention_layers, nemotron.n_mamba_layers) == (4, 27)
        assert nemotron.is_hybrid_ssm

        falcon = ModelArch.from_hf_config(FALCON_H1)
        assert (falcon.attention_layers, falcon.n_mamba_layers) == (24, 24)

        dense = QWEN_05B
        assert not dense.is_hybrid_ssm
        assert dense.attention_layers == dense.n_layers

    def test_nemotron_vl_reads_llm_config_with_depth_from_layer_types(self):
        # Arrange
        block_types = {"M": "mamba", "*": "attention", "-": "mlp"}
        text = {k: v for k, v in NEMOTRON_H.items() if k != "num_hidden_layers"}
        text["layers_block_type"] = [
            block_types[c] for c in NEMOTRON_H["hybrid_override_pattern"]
        ]
        vl_config = {"llm_config": text, "vision_config": {"hidden_size": 1280}}

        # Act
        arch = ModelArch.from_hf_config(vl_config)

        # Assert
        assert arch.n_layers == 56
        assert (arch.attention_layers, arch.n_mamba_layers) == (4, 27)
        assert arch.hidden_size == 4480

    def test_config_without_depth_is_rejected(self):
        text = {k: v for k, v in NEMOTRON_H.items() if k != "num_hidden_layers"}
        del text["hybrid_override_pattern"]

        with pytest.raises(KeyError, match="num_hidden_layers"):
            ModelArch.from_hf_config({"llm_config": text})

    def test_attention_only_layer_types_override_mamba_keys(self):
        # Granite 4.0 Micro: a granitemoehybrid config with mamba_* keys whose
        # layer_types list only attention layers.
        granite_micro = {
            "num_hidden_layers": 40,
            "hidden_size": 2560,
            "intermediate_size": 8192,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "vocab_size": 100352,
            "layer_types": ["attention"] * 40,
            "mamba_d_state": 256,
            "mamba_n_heads": 128,
            "mamba_d_head": 32,
        }

        arch = ModelArch.from_hf_config(granite_micro)

        assert (arch.attention_layers, arch.n_mamba_layers) == (40, 0)
        assert not arch.is_hybrid_ssm
        assert formulas.param_counts(arch).mamba == 0

    def test_layer_types_with_mamba_set_the_hybrid_mix(self):
        arch = ModelArch.from_hf_config(
            {**DECODER_TINY, "layer_types": ["mamba"] * 6 + ["attention"] * 2}
        )

        assert (arch.attention_layers, arch.n_mamba_layers) == (2, 6)
        assert arch.is_hybrid_ssm

    def test_full_attention_interval_splits_the_stack(self):
        arch = ModelArch.from_hf_config({**DECODER_TINY, "full_attention_interval": 4})

        assert (arch.attention_layers, arch.n_mamba_layers) == (2, 6)

    def test_block_exclusive_ffn_mix_from_layers_block_type(self):
        arch = ModelArch.from_hf_config(
            {
                **DECODER_TINY,
                "layers_block_type": ["attention", "mlp", "moe", "mamba"] * 2,
            }
        )

        assert (arch.n_mlp_layers, arch.n_moe_layers) == (2, 2)
        assert arch.moe_layers == 2
        assert arch.block_exclusive_layers

    def test_gated_delta_net_state_geometry(self):
        arch = ModelArch.from_hf_config(
            {
                **DECODER_TINY,
                "layer_types": ["full_attention", "linear"] * 4,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 64,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "linear_conv_kernel_dim": 4,
                "chunk_size": 64,
            }
        )

        assert (arch.attention_layers, arch.n_mamba_layers) == (4, 4)
        assert arch.mamba_d_state == 128
        assert arch.mamba_d_head == 64
        assert arch.mamba_n_heads == 32
        assert arch.mamba_d_conv == 4
        assert arch.mamba_chunk_size == 64
        assert arch.mamba_conv_dim == 2 * 16 * 128 + 32 * 64
        assert arch.mamba_gated_delta
        assert arch.sublayer_transients
        assert not arch.block_exclusive_layers

    def test_one_mixer_per_layer_mamba_stack_peaks_in_one_sublayer(self):
        arch = ModelArch.from_hf_config(
            {
                **DECODER_TINY,
                "layer_types": ["attention", "mamba", "mamba", "mamba"] * 2,
            }
        )

        assert (arch.attention_layers, arch.n_mamba_layers) == (2, 6)
        assert not arch.mamba_gated_delta
        assert arch.sublayer_transients

    @pytest.mark.parametrize(
        ("model_id", "dtype"),
        [
            ("ibm-granite/granite-4.0-h-tiny", "bf16"),
            ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "fp32"),
            ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "fp32"),
            ("Qwen/Qwen3.5-2B", "fp32"),
        ],
    )
    def test_ssm_state_dtype_follows_vllm(self, model_id, dtype):
        arch = ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[model_id].config)

        assert arch.mamba_ssm_state_dtype == dtype

    def test_gpt_oss_experts_keep_the_clamped_swiglu_intermediates(self):
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["openai/gpt-oss-20b"].config
        )
        routed = 1024 * arch.n_experts_per_tok
        inter = arch.expert_intermediate_size or arch.intermediate_size

        forward = formulas.routed_expert_bytes(arch, 1, 1024, 2.0)
        backward = formulas.routed_expert_bytes(arch, 1, 1024, 2.0, backward=True)

        assert arch.transposed_experts
        assert forward == routed * (2 * arch.expert_width + 8 * inter) * 2
        assert backward == routed * (2 * arch.expert_width + 9 * inter) * 2

    def test_gpt_oss_contracted_lora_is_not_capped_by_the_chunk(self):
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["openai/gpt-oss-20b"].config
        )
        sorted_path = arch.model_copy(update={"transposed_experts": False})
        common = {
            "rows": 1,
            "seq_len": 65536,
            "act_bytes": 2.0,
            "chunk_bytes": 512 * MiB,
            "expert_lora": True,
            "contracted": True,
            "ep": 1,
        }
        routed = 65536 * arch.n_experts_per_tok
        inter = arch.expert_intermediate_size or arch.intermediate_size
        widest = max(arch.expert_width, inter) * 2

        transposed = formulas.routed_chunk_bytes(arch, **common)
        capped = formulas.routed_chunk_bytes(sorted_path, **common)

        assert transposed == formulas.CONTRACTED_CHUNK_LIVE_MULTIPLE * routed * widest
        assert capped == formulas.CONTRACTED_CHUNK_LIVE_MULTIPLE * 512 * MiB

    def test_parallel_mixer_stack_sums_its_sublayers(self):
        arch = ModelArch.from_hf_config(FALCON_H1)

        assert not arch.sublayer_transients

    def test_counts_vision_encoder_params(self):
        empty = ModelArch.from_hf_config(
            {**DECODER_TINY, "vision_config": {"num_hidden_layers": 2}}
        )
        towered = ModelArch.from_hf_config(
            {
                **DECODER_TINY,
                "vision_config": {
                    "hidden_size": 128,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "head_dim": 32,
                    "num_key_value_heads": 2,
                    "intermediate_size": 256,
                    "position_embedding_size": 16,
                },
            }
        )

        attn = 128 * 4 * 32 + 2 * 128 * 2 * 32 + 4 * 32 * 128
        per_layer = attn + 2 * 128 * 256 + 2 * 128
        assert empty.multimodal_tower_params == 0
        assert towered.multimodal_tower_params == 2 * per_layer + 16 * 128

    def test_mla_caches_one_latent_head(self):
        arch = ModelArch.from_hf_config(GLM_TINY)

        assert (arch.n_kv_heads, arch.head_dim, arch.kv_lora_rank) == (1, 8, 16)
        assert (arch.attention_layers, arch.n_mamba_layers) == (2, 2)
        assert arch.mamba_kda
        assert arch.n_indexed_layers == 2

    def test_dense_mlp_layer_types_leave_the_rest_routed(self):
        arch = ModelArch.from_hf_config(GLM_TINY)

        assert (arch.dense_ffn_layers, arch.mlp_layers, arch.moe_layers) == (1, 1, 3)

    def test_first_k_dense_replace_and_moe_layer_freq_pick_dense_layers(self):
        # Arrange: layers 0 and 1 come before the first MoE layer; of the
        # rest, only even layers route experts.
        config = {
            **DECODER_TINY,
            "num_hidden_layers": 6,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "first_k_dense_replace": 2,
            "moe_layer_freq": 2,
        }

        # Act
        arch = ModelArch.from_hf_config(config)

        # Assert
        assert (arch.dense_ffn_layers, arch.mlp_layers, arch.moe_layers) == (4, 4, 2)

    def test_compress_ratios_count_csa_and_hca_layers_of_the_stack(self):
        arch = ModelArch.from_hf_config(DSV4_TINY)

        assert (arch.n_csa_layers, arch.n_hca_layers, arch.n_indexed_layers) == (
            1,
            1,
            1,
        )
        assert arch.shared_kv
        assert arch.sliding_window_layer_fraction == 1.0

    def test_unknown_compress_ratio_raises(self):
        config = {**DSV4_TINY, "compress_ratios": [0, 8, 4]}

        with pytest.raises(
            ValueError, match=r"compress_ratios has unknown entries \[8\]"
        ):
            ModelArch.from_hf_config(config)

    def test_hybrid_layer_pattern_sizes_sliding_and_full_kv_heads(self):
        arch = ModelArch.from_hf_config(MIMO_TINY)

        assert (arch.n_kv_heads, arch.global_n_kv_heads) == (2, 1)
        assert arch.sliding_window_layer_fraction == pytest.approx(2 / 3)
        assert arch.mean_kv_heads == pytest.approx(5 / 3)

    def test_per_layer_moe_layer_freq_marks_dense_layers(self):
        arch = ModelArch.from_hf_config(MIMO_TINY)

        assert (arch.dense_ffn_layers, arch.mlp_layers, arch.moe_layers) == (1, 1, 2)

    def test_text_only_family_counts_no_tower(self):
        arch = ModelArch.from_hf_config(MIMO_TINY)
        towered = ModelArch.from_hf_config({**MIMO_TINY, "model_type": "qwen3"})

        assert (arch.multimodal_tower_params, arch.vision_hidden_size) == (0, 0)
        assert towered.multimodal_tower_params > 0

    def test_ue8m0_fp8_with_fp4_experts_stores_scales_per_block(self):
        config = SUPPORTED_MODEL_INFO[DSV4_FLASH].config

        stored = ModelArch.from_hf_config(config).checkpoint_quantization

        assert stored is not None
        assert stored.linear_bytes == 1 + 1 / (128 * 128)
        assert stored.expert_bytes == 0.5 + 1 / 32
        assert stored.unquantized_attention == frozenset()
        assert stored.unquantized_mixer == frozenset()

    def test_fp32_scaled_fp8_keeps_unconverted_mla_and_kda_modules(self):
        # Arrange
        config = SUPPORTED_MODEL_INFO[GLM_FLASH].config

        # Act
        stored = ModelArch.from_hf_config(config).checkpoint_quantization

        # Assert
        assert stored is not None
        assert stored.linear_bytes == stored.expert_bytes == 1 + 4 / (128 * 128)
        assert "kv_b_proj" in stored.unquantized_attention
        assert "o_proj" not in stored.unquantized_attention
        assert set(formulas.KDA_LINEAR_NAMES) <= stored.unquantized_mixer

    def test_mxfp4_experts_keep_the_unconverted_output_projection(self):
        config = SUPPORTED_MODEL_INFO[MIMO_FLASH].config

        stored = ModelArch.from_hf_config(config).checkpoint_quantization

        assert stored is not None
        assert stored.linear_bytes == 1 + 4 / (128 * 128)
        assert stored.expert_bytes == 0.5 + 1 / 32
        assert stored.unquantized_attention == frozenset({"o_proj"})

    def test_only_prequantized_catalog_checkpoints_carry_quantization(self):
        quantized = {
            hub_id
            for hub_id, info in SUPPORTED_MODEL_INFO.items()
            if ModelArch.from_hf_config(info.config).checkpoint_quantization is not None
        }

        assert quantized == set(PREQUANTIZED_CHECKPOINTS)

    def test_per_tensor_fp8_stores_one_byte(self):
        config = {**DSV4_TINY, "quantization_config": {"quant_method": "fp8"}}

        stored = ModelArch.from_hf_config(config).checkpoint_quantization

        assert stored is not None
        assert stored.linear_bytes == stored.expert_bytes == 1.0

    def test_non_fp8_quant_method_is_unparsed(self):
        config = {**DSV4_TINY, "quantization_config": {"quant_method": "mxfp4"}}

        assert ModelArch.from_hf_config(config).checkpoint_quantization is None

    def test_unconverted_modules_on_draft_layers_are_skipped(self):
        # Arrange
        draft_layer = DSV4_TINY["num_hidden_layers"]
        quant = {
            "quant_method": "fp8",
            "modules_to_not_convert": [f"model.layers.{draft_layer}.self_attn.wo_a"],
        }

        # Act
        stored = ModelArch.from_hf_config(
            {**DSV4_TINY, "quantization_config": quant}
        ).checkpoint_quantization

        # Assert
        assert stored is not None
        assert stored.unquantized_attention == frozenset()


class TestHasRoutedExperts:
    @pytest.mark.parametrize(
        ("hub_id", "moe"),
        [
            ("Qwen/Qwen3-4B", False),
            ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", True),
            ("Qwen/Qwen3.6-35B-A3B", True),
        ],
    )
    def test_matches_the_checkpoint_arch(self, hub_id, moe):
        config = SUPPORTED_MODEL_INFO[hub_id].config

        assert has_routed_experts(config) is moe
        assert ModelArch.from_hf_config(config).is_moe is moe

    def test_config_without_expert_keys_is_dense(self):
        assert has_routed_experts({"model_type": "llama"}) is False

    def test_one_expert_is_dense(self):
        assert has_routed_experts({"num_local_experts": 1}) is False


class TestModelArchProperties:
    def test_double_wide_mlp_scales_mean_and_peak_width(self):
        arch = QWEN_05B.model_copy(
            update={"double_wide_mlp": True, "n_kv_shared_layers": 12}
        )

        assert arch.mlp_width_factor == (24 + 12) / 24
        assert arch.peak_mlp_width_factor == 2.0
        assert QWEN_05B.mlp_width_factor == 1.0

    def test_peak_qkv_dim_uses_global_head_dim(self):
        arch = QWEN_05B.model_copy(update={"global_head_dim": 128})

        assert arch.peak_qkv_dim == (14 + 2 * 2) * 128
        assert QWEN_05B.peak_qkv_dim == (14 + 2 * 2) * 64

    def test_mean_qkv_dim_without_a_wider_global_head_is_the_head_width(self):
        assert QWEN_05B.mean_qkv_dim == (14 + 2 * 2) * 64

    def test_mean_qkv_dim_under_mla_expands_keys_and_values_per_head(self):
        arch = QWEN_05B.model_copy(update={"kv_lora_rank": 512})

        assert arch.mean_qkv_dim == 14 * (2 * 64 + 64)

    def test_mean_kv_heads_without_global_kv_heads_is_n_kv_heads(self):
        assert QWEN_05B.mean_kv_heads == 2

    def test_moe_layers_follows_n_moe_layers(self):
        arch = MOE_TINY.model_copy(update={"n_moe_layers": 1})

        assert arch.moe_layers == 1
        assert MOE_TINY.moe_layers == MOE_TINY.n_layers

    def test_attention_params_scale_with_global_head_dim(self):
        arch = ModelArch.from_hf_config(
            {
                "num_hidden_layers": 35,
                "hidden_size": 1536,
                "intermediate_size": 6144,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "head_dim": 256,
                "global_head_dim": 512,
                "vocab_size": 262144,
                "tie_word_embeddings": True,
                "sliding_window": 512,
                "layer_types": ["sliding_attention"] * 28 + ["full_attention"] * 7,
            }
        )
        h, dh = arch.hidden_size, arch.head_dim
        q_dim, kv_dim = arch.n_heads * dh, arch.n_kv_heads * dh
        base = h * q_dim + 2 * h * kv_dim + q_dim * h
        sliding = arch.model_copy(update={"global_head_dim": None})

        assert formulas.attention_params_per_layer(sliding) == base
        assert formulas.attention_params_per_layer(arch) == int(
            base * arch.mean_qkv_dim / (q_dim + 2 * kv_dim)
        )


GRANITE_H_TINY = ModelArch.from_hf_config(
    SUPPORTED_MODEL_INFO["ibm-granite/granite-4.0-h-tiny"].config
)
GRANITE_MICRO = ModelArch.from_hf_config(
    SUPPORTED_MODEL_INFO["ibm-granite/granite-4.0-micro"].config
)


class TestGraniteSharedMlp:
    def test_hybrid_moe_reads_the_shared_mlp_beside_routed_experts(self):
        assert GRANITE_H_TINY.n_shared_experts == 1
        assert GRANITE_H_TINY.shared_expert_intermediate_size == 1024

    def test_dense_granite_keeps_its_mlp_as_the_dense_mlp(self):
        assert GRANITE_MICRO.n_shared_experts == 0
        assert GRANITE_MICRO.shared_expert_intermediate_size is None
        assert GRANITE_MICRO.intermediate_size == 8192


class TestMambaSharedGroupRowsParams:
    def test_one_group_over_tp_keeps_every_b_c_row(self):
        # 36 layers x 2 x 128 state rows x (1536 hidden + 4 conv taps + 1 bias).
        assert formulas.mamba_shared_group_rows_params(GRANITE_H_TINY, 2) == (
            36 * 256 * 1541
        )

    def test_groups_that_split_over_tp_keep_nothing(self):
        nemotron = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"].config
        )

        assert formulas.mamba_shared_group_rows_params(nemotron, 2) == 0

    def test_no_tensor_parallel_keeps_nothing(self):
        assert formulas.mamba_shared_group_rows_params(GRANITE_H_TINY, 1) == 0


class TestMambaSavedWidth:
    def test_fused_kernel_keeps_in_proj_and_scan_outputs(self):
        # 48 heads x 64 = 3072 inner, conv dim 3328.
        assert block_activations.mamba_saved_width(GRANITE_H_TINY, lora=False) == (
            2 * 3072 + 3328 + 48
        )

    def test_lora_adds_the_in_proj_and_out_proj_inputs(self):
        frozen = block_activations.mamba_saved_width(GRANITE_H_TINY, lora=False)

        assert block_activations.mamba_saved_width(GRANITE_H_TINY, lora=True) == (
            frozen + 1536 + 3072
        )

    def test_models_without_mamba_save_nothing(self):
        assert block_activations.mamba_saved_width(QWEN_05B, lora=True) == 0


class TestStoredBlockBytes:
    def test_expert_lora_keeps_only_a_score_and_index_per_slot(self):
        # Arrange
        arch = ModelArch.from_hf_config(GRANITE_MOE)
        slots = arch.n_experts_per_tok
        inter = arch.intermediate_size
        tokens = 4096

        # Act
        lora = block_activations.stored_block_bytes(arch, 1, tokens, 2.0, True, False)
        frozen = block_activations.stored_block_bytes(
            arch, 1, tokens, 2.0, False, False
        )

        # Assert
        held_by_frozen = slots * (2 * inter + arch.expert_width) * 2
        assert frozen - lora == pytest.approx(
            tokens * (held_by_frozen - slots * 12), abs=1
        )

    def test_bytes_scale_with_tokens(self):
        arch = ModelArch.from_hf_config(GRANITE_MOE)

        one = block_activations.stored_block_bytes(arch, 1, 2048, 2.0, True, False)
        four = block_activations.stored_block_bytes(arch, 4, 2048, 2.0, True, False)

        assert four == pytest.approx(4 * one, abs=4)

    def test_expert_parallel_keeps_measured_rows_per_slot(self):
        # Arrange
        arch = ModelArch.from_hf_config(GRANITE_MOE)
        tokens = 4096
        row_bytes = arch.n_experts_per_tok * arch.expert_width * 2

        # Act
        local = block_activations.stored_block_bytes(arch, 1, tokens, 2.0, True, False)
        lora_ep = block_activations.stored_block_bytes(
            arch, 1, tokens, 2.0, True, False, 2
        )
        frozen_ep = block_activations.stored_block_bytes(
            arch, 1, tokens, 2.0, False, False, 2
        )
        frozen = block_activations.stored_block_bytes(
            arch, 1, tokens, 2.0, False, False
        )

        # Assert
        assert lora_ep - local == pytest.approx(tokens * 4 * row_bytes, abs=1)
        assert frozen_ep - frozen == pytest.approx(tokens * row_bytes, abs=1)

    def test_expert_parallel_rows_split_across_tensor_parallel_ranks(self):
        # Arrange
        arch = ModelArch.from_hf_config(GRANITE_MOE)
        tokens = 4096
        local = block_activations.stored_block_bytes(arch, 1, tokens, 2.0, True, False)

        # Act
        ep = block_activations.stored_block_bytes(arch, 1, tokens, 2.0, True, False, 2)
        ep_tp = block_activations.stored_block_bytes(
            arch, 1, tokens, 2.0, True, False, 2, 2
        )

        # Assert
        assert ep_tp - local == pytest.approx((ep - local) / 2, abs=1)

    def test_hybrid_prices_mamba_lora_on_mamba_layers_only(self):
        # Arrange
        tokens = 1024
        lora_width = 1536 + 3072

        # Act
        lora = block_activations.stored_block_bytes(
            GRANITE_H_TINY, 1, tokens, 2.0, True, True
        )
        frozen = block_activations.stored_block_bytes(
            GRANITE_H_TINY, 1, tokens, 2.0, True, False
        )

        # Assert: 36 of 40 layers are Mamba.
        assert lora - frozen == pytest.approx(tokens * 36 / 40 * lora_width * 2, abs=1)

    def test_qk_norm_keeps_the_fp32_q_norm_input_of_every_attention_layer(self):
        # Arrange
        arch = ModelArch.from_hf_config(GRANITE_MOE)
        normed = arch.model_copy(update={"qk_norm": True})
        tokens = 1024
        q_dim = arch.n_heads * arch.head_dim

        # Act
        plain = block_activations.stored_block_bytes(arch, 1, tokens, 2.0, True, False)
        with_norm = block_activations.stored_block_bytes(
            normed, 1, tokens, 2.0, True, False
        )

        # Assert: every Granite MoE layer holds attention.
        assert arch.attention_layers == arch.n_layers
        assert with_norm - plain == pytest.approx(tokens * q_dim * 4, abs=1)

    def test_prices_recomputing_moe_stacks_only(self):
        assert block_activations.prices_stored_tensors(
            ModelArch.from_hf_config(GRANITE_MOE)
        )
        assert block_activations.prices_stored_tensors(GRANITE_H_TINY)
        assert not block_activations.prices_stored_tensors(
            ModelArch.from_hf_config(NEMOTRON_H_MOE)
        )
        assert not block_activations.prices_stored_tensors(QWEN_05B)
        assert not block_activations.prices_stored_tensors(GRANITE_MICRO)


class TestLoraParamCount:
    def test_hand_computed(self):
        arch = ModelArch(
            n_layers=1,
            hidden_size=8,
            intermediate_size=16,
            n_heads=2,
            n_kv_heads=1,
            head_dim=4,
            vocab_size=100,
        )
        # q: 8->8, k: 8->4, v: 8->4, o: 8->8, gate: 8->16, up: 8->16, down: 16->8
        rank = 2
        expected = rank * ((8 + 8) + (8 + 4) + (8 + 4) + (8 + 8) + 3 * (8 + 16))
        assert lora_formulas.lora_param_count(arch, rank) == expected
        # Attention-only scope drops the MLP terms.
        attn_only = rank * ((8 + 8) + (8 + 4) + (8 + 4) + (8 + 8))
        assert lora_formulas.lora_param_count(arch, rank, "attention-only") == attn_only

    def test_follows_hybrid_layer_mix(self):
        # Arrange
        nemotron = ModelArch.from_hf_config(NEMOTRON_H)
        rank = 8
        h, dh = nemotron.hidden_size, nemotron.head_dim
        q_dim = nemotron.n_heads * dh
        kv_dim = nemotron.n_kv_heads * dh
        attn = (h + q_dim) + 2 * (h + kv_dim) + (q_dim + h)
        mlp_matrices = 3 if nemotron.gated_mlp else 2
        mlp = mlp_matrices * (h + nemotron.intermediate_size)
        d_inner = nemotron.mamba_n_heads * nemotron.mamba_d_head
        mamba = (h + d_inner + nemotron.mamba_conv_dim + nemotron.mamba_n_heads) + (
            d_inner + h
        )
        assert (
            nemotron.attention_layers,
            nemotron.n_mamba_layers,
            nemotron.mlp_layers,
        ) == (
            4,
            27,
            25,
        )

        # Act
        attn_only = lora_formulas.lora_param_count(nemotron, rank, "attention-only")
        all_linear = lora_formulas.lora_param_count(nemotron, rank)

        # Assert: attention adapters on the 4 attention layers.
        assert attn_only == rank * attn * 4
        assert rank * attn * nemotron.n_layers == 14 * attn_only
        assert all_linear == rank * (
            attn * 4 + mlp * nemotron.mlp_layers + mamba * nemotron.n_mamba_layers
        )

    def test_scales_with_global_head_dim(self):
        arch = ModelArch.from_hf_config(
            {
                "num_hidden_layers": 35,
                "hidden_size": 1536,
                "intermediate_size": 6144,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "head_dim": 256,
                "global_head_dim": 512,
                "vocab_size": 262144,
                "tie_word_embeddings": True,
                "sliding_window": 512,
                "layer_types": ["sliding_attention"] * 28 + ["full_attention"] * 7,
            }
        )
        rank = 4

        # Act
        mixed = lora_formulas.lora_param_count(arch, rank, "attention-only")
        sliding = lora_formulas.lora_param_count(
            arch.model_copy(update={"global_head_dim": None}), rank, "attention-only"
        )

        # Assert: 28 sliding layers at q 2048 / kv 256, 7 full at q 4096 / kv 512.
        assert mixed == rank * (28 * 10752 + 7 * 15360)
        assert sliding == rank * 35 * 10752

    def test_gemma4_e4b_matches_its_inspected_language_linears(self):
        # Arrange
        arch = ModelArch.from_hf_config(GEMMA4_E4B.config)

        # Act
        all_linear = lora_formulas.lora_param_count(arch, 1)
        attention = lora_formulas.lora_param_count(arch, 1, "attention-only")

        # Assert: 18 KV-shared layers (15 sliding, 3 full) have no k/v; the
        # per-layer-input gate, projection and model projection add 249,856.
        assert attention == 567_296
        assert all_linear == 2_429_952

    def test_qwen36_shared_expert_adds_gated_mlp_adapters(self):
        # Arrange
        arch = ModelArch.from_hf_config(QWEN36.config)
        no_shared = arch.model_copy(update={"n_shared_experts": 0})

        # Act
        shared = lora_formulas.lora_param_count(
            arch, 16
        ) - lora_formulas.lora_param_count(no_shared, 16)

        # Assert: 40 layers of gate/up 2048->512 and down 512->2048.
        assert shared == 16 * 40 * 3 * (2048 + 512)

    def test_granite_h_tiny_matches_its_inspected_linears(self):
        # Arrange
        no_shared = GRANITE_H_TINY.model_copy(update={"n_shared_experts": 0})

        # Act
        total = lora_formulas.lora_param_count(GRANITE_H_TINY, 1)
        shared = total - lora_formulas.lora_param_count(no_shared, 1)

        # Assert: shared_mlp input_linear 1536->2048 and output_linear 1024->1536.
        assert shared == 40 * ((1536 + 2048) + (1024 + 1536))
        assert total == 740_032

    def test_qwen36_matches_its_inspected_language_linears(self):
        # Arrange
        arch = ModelArch.from_hf_config(QWEN36.config)
        ungated = arch.model_copy(update={"attention_output_gate": False})

        # Act
        total = lora_formulas.lora_param_count(arch, 1)
        gate = lora_formulas.lora_param_count(
            arch, 1, "attention-only"
        ) - lora_formulas.lora_param_count(ungated, 1, "attention-only")

        # Assert: q_proj 2048->8192 carries the output gate on 10 attention
        # layers; 30 GDN layers wrap in_proj_qkv 2048->8192, in_proj_z
        # 2048->4096, in_proj_b/a 2048->32 and out_proj 4096->2048.
        assert gate == 10 * 4096
        assert total == 1_322_880

    def test_granite_micro_fuses_its_dense_gate_up(self):
        # Act
        total = lora_formulas.lora_param_count(GRANITE_MICRO, 1)

        # Assert: 40 layers of q/o 2560<->2560, k/v 2560->512, input_linear
        # 2560->16384 and output_linear 8192->2560.
        assert total == 40 * (10240 + 6144 + (2560 + 16384) + (8192 + 2560))
        assert total == 1_843_200


class TestTowerLoraParamCount:
    def test_qwen36_tower_and_merger(self):
        arch = ModelArch.from_hf_config(QWEN36.config)

        # 27 blocks of qkv 1152->3456 and fc1/fc2 1152<->4304, merger 4608->4608->2048.
        assert lora_formulas.tower_lora_param_count(arch, 1, QWEN_TOWER_TARGETS) == (
            434_912
        )

    def test_qwen36_charges_only_the_named_targets(self):
        arch = ModelArch.from_hf_config(QWEN36.config)

        # 27 blocks of qkv 1152->3456.
        assert lora_formulas.tower_lora_param_count(arch, 1, {"qkv"}) == 124_416

    def test_gemma4_e4b_vision_and_audio_towers(self):
        arch = ModelArch.from_hf_config(GEMMA4_E4B.config)

        # Vision: 16 blocks of q/k/v/o 768 and gate/up/down 768<->3072, patch
        # 768->768, 768->2560 (287,488). Audio: 12 conformer layers of two
        # 1024<->4096 feed-forwards, light conv 1024->2048 and 1024->1024,
        # q/k/v/post and relative_k 1024->1024, subsample 1024->1024, output
        # 1024->1536 and embedder 1536->2560 (438,784).
        assert lora_formulas.tower_lora_param_count(arch, 1, GEMMA4_TOWER_TARGETS) == (
            287_488 + 438_784
        )

    def test_gemma4_e4b_audio_only_targets_skip_the_vision_tower(self):
        arch = ModelArch.from_hf_config(GEMMA4_E4B.config)

        # Act
        audio = lora_formulas.tower_lora_param_count(
            arch, 1, {"relative_k_proj", "input_proj_linear", "output_proj"}
        )

        # Assert: 12 relative_k 1024->1024, subsample 1024->1024, output 1024->1536.
        assert audio == 12 * 2048 + 2048 + 2560

    def test_gemma4_without_an_audio_tower_charges_only_vision(self):
        arch = ModelArch.from_hf_config({**GEMMA4_E4B.config, "audio_config": None})

        # Vision tower only (287,488); the audio terms drop out.
        assert lora_formulas.tower_lora_param_count(arch, 1, GEMMA4_TOWER_TARGETS) == (
            287_488
        )

    def test_super_vl_encoder_patch_embedders_and_projector(self):
        arch = ModelArch.from_hf_config(SUPER_VL.config)

        # 32 blocks of q/k/v/o 1280 and fc1/fc2 1280<->5120, patch 768->1280,
        # video patch 1536->1280, projector 5120->20480->4096.
        assert lora_formulas.tower_lora_param_count(arch, 1, RADIO_TOWER_TARGETS) == (
            792_320
        )

    def test_text_model_has_no_tower_adapters(self):
        assert (
            lora_formulas.tower_lora_param_count(QWEN_05B, 16, QWEN_TOWER_TARGETS) == 0
        )

    def test_mla_and_kda_skip_the_excluded_gate_projections(self):
        # Arrange
        arch = ModelArch.from_hf_config(GLM_TINY)
        # q_a, q_b, kv_a, kv_b, o per MLA layer.
        mla = (64 + 32) + (32 + 32) + (64 + 16) + (16 + 64) + (32 + 64)
        # q, k, v, f_b, beta, g_b, o per KDA layer.
        kda = 3 * (64 + 16) + 2 * (8 + 16) + (64 + 2) + (16 + 64)
        # gate, up and down of the shared expert per MoE layer.
        shared = 3 * (64 + 16)
        # f_a and g_a per KDA layer.
        excluded = 2 * (64 + 8)

        # Act
        attn_only = lora_formulas.lora_param_count(arch, 1, "attention-only")
        all_linear = lora_formulas.lora_param_count(arch, 1)
        engine = lora_formulas.engine_lora_param_count(arch, 1)

        # Assert
        assert attn_only == 2 * mla
        assert all_linear == 2 * mla + 3 * (64 + 96) + 3 * shared + 2 * kda
        assert engine == all_linear + 2 * excluded

    def test_grouped_output_wraps_only_the_mixing_linear(self):
        arch = ModelArch.from_hf_config(DSV4_TINY)

        # q_a, q_b, the shared kv, o_b, and the shared expert's gate, up and
        # down per layer.
        expected = 3 * ((64 + 32) + (32 + 64) + (64 + 16) + (16 + 64) + 3 * (64 + 16))
        assert lora_formulas.lora_param_count(arch, 1) == expected

    def test_values_and_kv_heads_take_each_layer_width(self):
        # Arrange: the adapter wraps o_proj only; the engine wraps q, k, v too.
        arch = ModelArch.from_hf_config(MIMO_TINY)
        full = (64 + 64) + (64 + 16) + (64 + 8) + (32 + 64)
        sliding = (64 + 64) + (64 + 32) + (64 + 16) + (32 + 64)
        output = 32 + 64

        # Act
        attn_only = lora_formulas.lora_param_count(arch, 1, "attention-only")
        all_linear = lora_formulas.lora_param_count(arch, 1)
        engine = lora_formulas.engine_lora_param_count(arch, 1)

        # Assert
        assert attn_only == 3 * output
        assert all_linear == 3 * output + 3 * (64 + 96)
        assert engine == full + 2 * sliding + 3 * (64 + 96)

    @pytest.mark.parametrize(
        "config", [{**DECODER_TINY, "model_type": "qwen3"}, NEMOTRON_H]
    )
    def test_engine_count_matches_the_adapter_without_exclusions(self, config):
        arch = ModelArch.from_hf_config(config)

        assert arch.lora_excluded_modules == frozenset()
        assert lora_formulas.engine_lora_param_count(arch, 16) == (
            lora_formulas.lora_param_count(arch, 16)
        )

    def test_unwrapped_deepseek_v4_exclusions_leave_the_count_unchanged(self):
        arch = ModelArch.from_hf_config(DSV4_TINY)

        assert "o_a_proj" in arch.lora_excluded_modules
        assert lora_formulas.engine_lora_param_count(arch, 1) == (
            lora_formulas.lora_param_count(arch, 1)
        )


class TestResolveChunkRows:
    def test_targets_256mib_logit_tile(self):
        # 256 MiB / (151936 vocab * 4 bytes) = 441 rows.
        assert resolve_chunk_rows(151936) == 441
        assert resolve_chunk_rows(1000) == 4096  # clamped high
        assert resolve_chunk_rows(10_000_000) == 128  # clamped low


class TestSupportRowsBytes:
    def test_a_narrow_support_fills_one_vocab_tile_with_head_rows(self):
        # 441 * 151936 // (20 * 896) = 3739 tokens of 20 rows; bf16 plus fp32.
        assert formulas.support_rows_bytes(QWEN_05B, 441, 20, 2.0) == (
            3739 * 20 * 896 * 6
        )

    def test_an_fp32_head_holds_one_copy(self):
        assert formulas.support_rows_bytes(QWEN_05B, 441, 20, 4.0) == (
            3739 * 20 * 896 * 4
        )

    def test_a_wide_support_takes_fewer_tokens_than_the_tile(self):
        # 441 * 151936 // (1000 * 896) = 74 tokens of 1000 rows.
        assert formulas.support_rows_bytes(QWEN_05B, 441, 1000, 2.0) == (
            74 * 1000 * 896 * 6
        )

    def test_a_vocab_wide_support_takes_one_token(self):
        assert formulas.support_rows_bytes(QWEN_05B, 441, 151936, 2.0) == (
            151936 * 896 * 6
        )


class TestLogitTerms:
    def test_loss_holds_two_fp32_tiles_and_no_head_copy(self):
        settings = TrainingSettings()

        logits, nograd = formulas.logit_terms(QWEN_05B, settings, 2.0)

        rows = resolve_chunk_rows(QWEN_05B.vocab_size)
        assert logits == rows * QWEN_05B.vocab_size * 8
        assert nograd == rows * QWEN_05B.vocab_size * 4

    def test_support_width_replaces_vocab_tiles(self):
        settings = TrainingSettings(sampling_support_width=20)

        logits, nograd = formulas.logit_terms(QWEN_05B, settings, 2.0)

        rows = resolve_chunk_rows(QWEN_05B.vocab_size)
        support = formulas.support_rows_bytes(QWEN_05B, rows, 20, 2.0)
        assert logits == support
        assert nograd == support


class TestResolveMaxNumBatchedTokens:
    def test_unset_takes_the_async_engine_default(self):
        assert kv_cache.resolve_max_num_batched_tokens(8, 32768) == 8192

    def test_explicit_value_wins(self):
        assert kv_cache.resolve_max_num_batched_tokens(8, 1024, explicit=4096) == 4096

    def test_caps_at_every_sequence_at_full_context(self):
        assert kv_cache.resolve_max_num_batched_tokens(1, 1024) == 1024
        assert kv_cache.resolve_max_num_batched_tokens(2, 512, explicit=4096) == 1024


class TestKvCacheBytesPerToken:
    def test_gqa(self):
        # 2 (K+V) * 24 layers * 2 kv-heads * 64 head-dim * 2 bytes.
        assert kv_cache.kv_cache_bytes_per_token(QWEN_05B, "bf16") == 12288

    def test_counts_only_attention_layers(self):
        nemotron = ModelArch.from_hf_config(NEMOTRON_H)
        per_token = kv_cache.kv_cache_bytes_per_token(nemotron, "bf16")
        assert per_token == 2 * 4 * nemotron.n_kv_heads * nemotron.head_dim * 2

    def test_uses_global_head_dim_on_full_attention_layers(self):
        # Arrange
        arch = ModelArch.from_hf_config(
            {
                "num_hidden_layers": 35,
                "hidden_size": 1536,
                "intermediate_size": 6144,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "head_dim": 256,
                "global_head_dim": 512,
                "vocab_size": 262144,
                "tie_word_embeddings": True,
                "sliding_window": 512,
                "layer_types": ["sliding_attention"] * 28 + ["full_attention"] * 7,
            }
        )
        kv_bytes = 2
        storing = 35
        windowed = 0.8
        mean_dh = windowed * 256 + (1.0 - windowed) * 512

        # Act
        per_token = kv_cache.kv_cache_bytes_per_token(arch, "bf16")
        demand = kv_cache.kv_cache_demand_bytes(arch, "bf16", 8, 8192, 0)

        # Assert: full-attention KV at the 512-wide head; a 512-token window
        # holds 511 tokens in 32 blocks, plus one.
        assert per_token == pytest.approx(2 * storing * 1 * mean_dh * kv_bytes)
        assert demand == int(
            8
            * 2
            * storing
            * 1
            * kv_bytes
            * (windowed * 33 * 16 * 256 + (1.0 - windowed) * 8192 * 512)
        )
        sliding_only = kv_cache.kv_cache_bytes_per_token(
            arch.model_copy(update={"global_head_dim": None}), "bf16"
        )
        assert per_token == pytest.approx(sliding_only * mean_dh / 256)

        # No sliding window: every storing layer uses the full-attention head width.
        full = arch.model_copy(update={"sliding_window": None})
        assert kv_cache.kv_cache_bytes_per_token(full, "bf16") == (
            2 * storing * 1 * 512 * 2
        )
        assert kv_cache.kv_cache_demand_bytes(full, "bf16", 8, 8192, 0) == int(
            8 * 8192 * 2 * storing * 1 * 2 * 512
        )

    def test_mla_stores_the_latent_and_indexer_keys(self):
        arch = ModelArch.from_hf_config(GLM_TINY)

        # Two MLA layers cache a 16-wide latent and 8-wide indexer keys.
        assert kv_cache.kv_cache_bytes_per_token(arch, "bf16") == 2 * (16 + 8) * 2

    def test_compressed_layers_add_their_entries_to_every_window(self):
        arch = ModelArch.from_hf_config(DSV4_TINY)

        # One shared 16-wide KV per layer, one CSA entry per 4 tokens, one
        # HCA entry per 128, and CSA-rate 8-wide indexer keys.
        expected = (3 * 16 + 16 / 4 + 16 / 128 + 8 / 4) * 2
        assert kv_cache.kv_cache_bytes_per_token(arch, "bf16") == expected

    def test_values_cache_at_their_own_width(self):
        arch = ModelArch.from_hf_config(MIMO_TINY)

        # A 16-wide key and 8-wide value per KV head: one head on the full
        # layer, two on each sliding layer.
        expected = (1 * (16 + 8) + 2 * 2 * (16 + 8)) * 2
        assert kv_cache.kv_cache_bytes_per_token(arch, "bf16") == pytest.approx(
            expected
        )


class TestKvCacheDemandBytes:
    def test_sliding_window_caps_growth(self):
        windowed = QWEN_05B.model_copy(update={"sliding_window": 1024})
        full = kv_cache.kv_cache_demand_bytes(QWEN_05B, "bf16", 8, 8192, 0)
        capped = kv_cache.kv_cache_demand_bytes(windowed, "bf16", 8, 8192, 0)
        # 1023 tokens round to 64 blocks, plus one: 1040 of 8192 tokens.
        assert capped == full * 1040 // 8192

    def test_sliding_window_holds_the_in_flight_tokens(self):
        # gpt-oss on vLLM 0.32: a 128-token window with two 2048-token steps
        # in flight holds 265 blocks per sequence.
        windowed = QWEN_05B.model_copy(update={"sliding_window": 128})
        full = kv_cache.kv_cache_demand_bytes(QWEN_05B, "bf16", 1, 16384, 0)

        held = kv_cache.kv_cache_demand_bytes(windowed, "bf16", 1, 16384, 2 * 2048)
        capped = kv_cache.kv_cache_demand_bytes(windowed, "bf16", 1, 2048, 2 * 2048)

        assert held == full * 265 * 16 // 16384
        # A 2048-token sequence caps the window at 128 blocks, plus one.
        assert capped == full * 129 * 16 // 16384

    def test_compressed_entries_grow_past_the_window(self):
        # Arrange
        arch = ModelArch.from_hf_config(DSV4_TINY)

        # Act
        demand = kv_cache.kv_cache_demand_bytes(arch, "bf16", 2, 1024, 0)

        # Assert: a 7-token window holds one block plus one in each of three
        # layers; compressed entries and indexer keys cover all 1024 tokens.
        window = 2 * 3 * 16 * 2 * 32
        compressed = 2 * 1024 * (16 / 4 + 16 / 128 + 8 / 4) * 2
        assert demand == window + compressed


class TestWeightBytes:
    def test_variants(self):
        counts = formulas.param_counts(QWEN_05B)
        dense = formulas.weight_bytes(counts, "bf16", WeightVariant())
        assert dense == int(counts.total * 2)

        towered = replace(counts, multimodal_towers=1_000_000)
        full = formulas.weight_bytes(towered, "bf16", WeightVariant())
        stripped = formulas.weight_bytes(
            towered, "bf16", WeightVariant(name="stripped", stripped_multimodal=True)
        )
        assert full - stripped == 1_000_000 * 2

    def test_carries_the_reconciliation_at_checkpoint_dtype(self):
        analytic = formulas.param_counts(QWEN_05B)
        exact = formulas.param_counts(QWEN_05B, analytic.total + 1_000_000)
        delta = formulas.weight_bytes(exact, "bf16", WeightVariant()) - (
            formulas.weight_bytes(analytic, "bf16", WeightVariant())
        )
        assert delta == 1_000_000 * 2


class TestEngineWeightBytes:
    def test_unquantized_matches_weight_bytes(self):
        counts = formulas.param_counts(QWEN_05B)

        stored = formulas.engine_weight_bytes(
            QWEN_05B, counts, "bf16", WeightVariant(), "none"
        )

        assert stored == formulas.weight_bytes(counts, "bf16", WeightVariant())

    def test_fp8_halves_linear_weights_only(self):
        # Arrange
        counts = formulas.param_counts(QWEN_05B)
        linear = counts.attention + counts.mlp + counts.moe_experts + counts.mamba

        # Act
        stored = formulas.engine_weight_bytes(
            QWEN_05B, counts, "bf16", WeightVariant(), "fp8"
        )

        # Assert
        assert stored == (counts.total - linear) * 2 + linear
        assert counts.embedding > 0

    def test_fp8_halves_expert_and_mamba_weights(self):
        counts = replace(
            formulas.param_counts(QWEN_05B), moe_experts=1_000_000, mamba=500_000
        )
        base = replace(counts, moe_experts=0, mamba=0)

        delta = formulas.engine_weight_bytes(
            QWEN_05B, counts, "bf16", WeightVariant(), "fp8"
        ) - formulas.engine_weight_bytes(QWEN_05B, base, "bf16", WeightVariant(), "fp8")

        assert delta == 1_500_000

    # Hub safetensors bytes less the MTP draft layers vLLM leaves unloaded and
    # the towers the engine variant strips, over TP 8.
    @pytest.mark.parametrize(
        ("hub_id", "gib_per_gpu"),
        [(DSV4_FLASH, 18.17), (GLM_FLASH, 37.22), (MIMO_FLASH, 19.81)],
    )
    def test_prequantized_checkpoint_matches_its_stored_bytes(
        self, hub_id, gib_per_gpu
    ):
        # Arrange
        info = SUPPORTED_MODEL_INFO[hub_id]
        arch = ModelArch.from_hf_config(info.config)
        counts = formulas.param_counts(arch, info.inspected["num_params"])
        engine = WeightVariant(name="engine", stripped_multimodal=True)

        # Act
        stored = formulas.engine_weight_bytes(arch, counts, "bf16", engine, "none")

        # Assert
        assert stored / 8 / GiB == pytest.approx(gib_per_gpu, rel=0.02)

    @pytest.mark.parametrize("hub_id", PREQUANTIZED_CHECKPOINTS)
    def test_online_fp8_leaves_a_prequantized_checkpoint_as_stored(self, hub_id):
        info = SUPPORTED_MODEL_INFO[hub_id]
        arch = ModelArch.from_hf_config(info.config)
        counts = formulas.param_counts(arch, info.inspected["num_params"])

        stored = formulas.engine_weight_bytes(
            arch, counts, "bf16", WeightVariant(), "none"
        )

        assert stored == formulas.engine_weight_bytes(
            arch, counts, "bf16", WeightVariant(), "fp8"
        )

    @pytest.mark.parametrize("hub_id", [GLM_FLASH, MIMO_FLASH])
    def test_unconverted_modules_stay_at_checkpoint_dtype(self, hub_id):
        # Arrange
        info = SUPPORTED_MODEL_INFO[hub_id]
        arch = ModelArch.from_hf_config(info.config)
        counts = formulas.param_counts(arch, info.inspected["num_params"])
        stored = arch.checkpoint_quantization
        assert stored is not None
        all_converted = arch.model_copy(
            update={
                "checkpoint_quantization": stored.model_copy(
                    update={
                        "unquantized_attention": frozenset(),
                        "unquantized_mixer": frozenset(),
                    }
                )
            }
        )

        # Act
        kept = formulas.engine_weight_bytes(
            arch, counts, "bf16", WeightVariant(), "none"
        )
        converted = formulas.engine_weight_bytes(
            all_converted, counts, "bf16", WeightVariant(), "none"
        )

        # Assert
        assert kept > converted

    @pytest.mark.parametrize(
        "hub_id",
        [h for h in SUPPORTED_MODEL_INFO if h not in PREQUANTIZED_CHECKPOINTS],
    )
    def test_bf16_checkpoint_serves_its_full_weights(self, hub_id):
        info = SUPPORTED_MODEL_INFO[hub_id]
        arch = ModelArch.from_hf_config(info.config)
        counts = formulas.param_counts(arch, info.inspected["num_params"])

        stored = formulas.engine_weight_bytes(
            arch, counts, "bf16", WeightVariant(), "none"
        )

        assert stored == formulas.weight_bytes(counts, "bf16", WeightVariant())


# Granite 3.1 3B-A800M geometry; the byte counts below are the live tensors a
# torch allocator snapshot showed at one rank's 16k-token backward peak.
GRANITE_MOE = {
    "model_type": "granitemoe",
    "num_hidden_layers": 32,
    "hidden_size": 1536,
    "intermediate_size": 512,
    "num_attention_heads": 24,
    "num_key_value_heads": 8,
    "vocab_size": 49155,
    "num_local_experts": 40,
    "num_experts_per_tok": 8,
}
# Qwen3-4B geometry, same snapshot source.
QWEN3_4B = {
    "model_type": "qwen3",
    "num_hidden_layers": 36,
    "hidden_size": 2560,
    "intermediate_size": 9728,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
}


# Nemotron 3.5 Lightning 30B-A3B geometry, first seven layers: MoE layers
# alternate with Mamba and attention and never sit side by side.
NEMOTRON_H_MOE = {
    "model_type": "nemotron_h",
    "num_hidden_layers": 7,
    "hidden_size": 2688,
    "intermediate_size": 1856,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "vocab_size": 131072,
    "n_routed_experts": 128,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 1856,
    "moe_shared_expert_intermediate_size": 3712,
    "n_shared_experts": 1,
    "mlp_hidden_act": "relu2",
    "ssm_state_size": 128,
    "conv_kernel": 4,
    "n_groups": 8,
    "mamba_num_heads": 64,
    "mamba_head_dim": 64,
    "chunk_size": 128,
    "tie_word_embeddings": False,
    "layers_block_type": [
        "mamba",
        "moe",
        "mamba",
        "moe",
        "mamba",
        "attention",
        "moe",
    ],
}


class TestRoutedExpertBytes:
    def test_chunked_routed_backward_keeps_inputs_outputs_and_ungated_activation(self):
        # Arrange
        arch = ModelArch.from_hf_config(NEMOTRON_H_MOE)
        routed = 16384 * 6

        # Act
        backward = formulas.routed_expert_bytes(arch, 1, 16384, 2.0, backward=True)
        forward = formulas.routed_expert_bytes(arch, 1, 16384, 2.0)

        # Assert: the gathered input and expert output (hidden each), the up
        # projection and its relu2 output (inter each); a no-grad pass keeps
        # only the gathered input. Both add the fp32 per-token layer output.
        combined = 16384 * 2688 * 4
        assert arch.chunked_routed_experts
        assert not arch.gated_mlp
        assert backward == routed * (2 * 2688 + 2 * 1856) * 2 + combined
        assert forward == routed * 2688 * 2 + combined

    def test_gated_chunked_routed_backward_keeps_gate_and_up(self):
        qwen3_moe = {
            "model_type": "qwen3_moe",
            "num_hidden_layers": 4,
            "hidden_size": 64,
            "intermediate_size": 128,
            "moe_intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 128,
            "num_experts": 8,
            "num_experts_per_tok": 2,
        }
        arch = ModelArch.from_hf_config(qwen3_moe)

        backward = formulas.routed_expert_bytes(arch, 1, 1024, 2.0, backward=True)

        assert arch.chunked_routed_experts
        assert backward == 1024 * 2 * (2 * 64 + 3 * 32) * 2 + 1024 * 64 * 4

    def test_matches_the_grouped_gemm_backward_workspace(self):
        arch = ModelArch.from_hf_config(GRANITE_MOE)

        backward = formulas.routed_expert_bytes(arch, 1, 16384, 2.0, backward=True)

        # Granite's sorted experts: gather + expert output (hidden each),
        # gate/up, activation, its grad.
        assert not arch.chunked_routed_experts
        assert backward == 16384 * 8 * (2 * 1536 + 4 * 512) * 2
        assert backward == 1.25 * GiB

    def test_forward_drops_the_activation_gradient_and_dense_is_zero(self):
        arch = ModelArch.from_hf_config(GRANITE_MOE)

        forward = formulas.routed_expert_bytes(arch, 1, 16384, 2.0)

        assert forward == 16384 * 8 * (2 * 1536 + 3 * 512) * 2
        assert formulas.routed_expert_bytes(QWEN_05B, 1, 16384, 2.0) == 0

    def test_shared_experts_stay_out_of_the_routed_workspace(self):
        arch = ModelArch.from_hf_config(GRANITE_MOE)
        shared = arch.model_copy(update={"n_shared_experts": 2})

        assert formulas.routed_expert_bytes(
            shared, 1, 1024, 2.0
        ) == formulas.routed_expert_bytes(arch, 1, 1024, 2.0)

    def test_engine_forward_uses_active_expert_width(self):
        arch = ModelArch.from_hf_config(GRANITE_MOE)
        engine = block_activations.block_recompute_bytes(
            arch, 1, 1024, 2.0, engine_forward=True
        )
        trainer = block_activations.block_recompute_bytes(arch, 1, 1024, 2.0)

        assert engine != trainer
        assert engine > 0


class TestSavedRouterTopkBytes:
    def test_charges_fp32_scores_and_int64_indices_per_moe_layer(self):
        arch = ModelArch.from_hf_config(GRANITE_MOE)

        saved = formulas.saved_router_topk_bytes(arch, 1, 16384)

        # 32 MoE layers x 16k tokens x top-8, 4 + 8 bytes per routed slot.
        assert saved == 32 * 16384 * 8 * 12
        assert saved == 48 * MiB

    def test_block_exclusive_stack_charges_only_moe_layers(self):
        arch = ModelArch.from_hf_config(
            {
                **NEMOTRON_H_MOE,
                "num_hidden_layers": 3,
                "layers_block_type": ["moe", "attention", "moe"],
            }
        )

        saved = formulas.saved_router_topk_bytes(arch, 2, 1024)

        assert saved == 2 * 2 * 1024 * 6 * 12

    def test_dense_model_saves_nothing(self):
        assert formulas.saved_router_topk_bytes(QWEN_05B, 1, 16384) == 0


class TestMoeCollectivesSaveable:
    def test_expert_parallel_moe_is_saveable(self):
        arch = ModelArch.from_hf_config(GRANITE_MOE)

        assert formulas.moe_collectives_saveable(arch, 2, 1)
        assert not formulas.moe_collectives_saveable(arch, 1, 2)

    def test_tensor_parallel_needs_a_saved_shared_expert_or_latent(self):
        qwen = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["Qwen/Qwen3.6-35B-A3B"].config
        )
        latent = ModelArch.from_hf_config({**NEMOTRON_H_MOE, "moe_latent_size": 1024})
        trailing = ModelArch.from_hf_config(NEMOTRON_H_MOE)
        beside = ModelArch.from_hf_config(
            {**GRANITE_MOE, "shared_intermediate_size": 1024}
        )

        assert formulas.moe_collectives_saveable(qwen, 1, 2)
        assert not formulas.moe_collectives_saveable(qwen, 1, 1)
        assert formulas.moe_collectives_saveable(latent, 1, 2)
        assert not formulas.moe_collectives_saveable(trailing, 1, 2)
        assert not formulas.moe_collectives_saveable(beside, 1, 2)

    def test_dense_model_is_not_saveable(self):
        assert not formulas.moe_collectives_saveable(QWEN_05B, 2, 2)


class TestSavedMoeCollectiveBytes:
    def test_expert_parallel_saves_dispatched_and_combined_rows(self):
        arch = ModelArch.from_hf_config(NEMOTRON_H_MOE)

        saved = formulas.saved_moe_collective_bytes(arch, 1, 16384, 2.0, 8, 1, 1)

        # 16k tokens x top-6 routed rows, dispatched and combined, at hidden 2688.
        assert saved == 2 * 16384 * 6 * 2688 * 2
        assert saved == 1008 * MiB

    def test_tensor_parallel_adds_row_gather_and_skips_trailing_shared_expert(self):
        arch = ModelArch.from_hf_config(NEMOTRON_H_MOE)

        saved = formulas.saved_moe_collective_bytes(arch, 1, 16384, 2.0, 2, 2, 2)

        # Half the routed rows each way and both ranks' 8k-row spans. The
        # recompute stops before the trailing shared expert's reduce.
        assert not arch.shared_expert_reduce_saved
        assert saved == (2 * 16384 * 6 // 2 + 2 * 8192) * 2688 * 2
        assert saved == 588 * MiB

    def test_gated_shared_expert_saves_its_reduce(self):
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["Qwen/Qwen3.6-35B-A3B"].config
        )

        saved = formulas.saved_moe_collective_bytes(arch, 1, 16384, 2.0, 1, 1, 2)

        # The sigmoid gate reads the shared expert's 16k-token reduce at hidden 2048.
        assert arch.shared_expert_reduce_saved
        assert saved == 16384 * 2048 * 2
        assert saved == 64 * MiB

    def test_latent_moe_saves_both_latent_projection_collectives(self):
        arch = ModelArch.from_hf_config({**NEMOTRON_H_MOE, "moe_latent_size": 1024})

        saved = formulas.saved_moe_collective_bytes(arch, 1, 16384, 2.0, 1, 1, 2)
        routed = formulas.saved_moe_collective_bytes(arch, 1, 16384, 2.0, 2, 1, 1)

        # Latent input gather and hidden output reduce.
        assert saved == 16384 * (1024 + 2688) * 2
        # Routed rows travel at the latent width.
        assert routed == 2 * 16384 * 6 * 1024 * 2

    def test_shared_mlp_beside_the_experts_saves_nothing(self):
        arch = ModelArch.from_hf_config(
            {**GRANITE_MOE, "shared_intermediate_size": 1024}
        )

        saved = formulas.saved_moe_collective_bytes(arch, 1, 16384, 2.0, 1, 1, 2)

        assert arch.n_shared_experts == 1
        assert not arch.shared_expert_reduce_saved
        assert saved == 0

    def test_dense_model_and_unsplit_moe_save_nothing(self):
        arch = ModelArch.from_hf_config(NEMOTRON_H_MOE)

        assert (
            formulas.saved_moe_collective_bytes(QWEN_05B, 1, 16384, 2.0, 2, 2, 2) == 0
        )
        assert formulas.saved_moe_collective_bytes(arch, 1, 16384, 2.0, 1, 1, 1) == 0


# Nemotron 3 Nano 4B Mamba-2 geometry. At one rank's 16k-token FSDP backward
# peak a torch allocator snapshot held 3.63 GiB in one Mamba layer's scan.
NEMOTRON_NANO_MAMBA = {
    "model_type": "nemotron_h",
    "num_hidden_layers": 42,
    "hidden_size": 3136,
    "intermediate_size": 12544,
    "num_attention_heads": 40,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 131072,
    "hybrid_override_pattern": "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-",
    "ssm_state_size": 128,
    "conv_kernel": 4,
    "n_groups": 8,
    "mamba_num_heads": 96,
    "mamba_head_dim": 80,
    "chunk_size": 256,
}


class TestMambaBlockBytes:
    def test_backward_matches_the_scan_snapshot(self):
        arch = ModelArch.from_hf_config(NEMOTRON_NANO_MAMBA)

        backward = block_activations.mamba_block_bytes(
            arch, 1, 16384, 2.0, backward=True
        )

        assert arch.mamba_chunk_size == 256
        assert backward == pytest.approx(3.63 * GiB, rel=0.03)

    def test_forward_is_smaller_and_attention_only_is_zero(self):
        arch = ModelArch.from_hf_config(NEMOTRON_NANO_MAMBA)

        forward = block_activations.mamba_block_bytes(arch, 1, 16384, 2.0)
        backward = block_activations.mamba_block_bytes(
            arch, 1, 16384, 2.0, backward=True
        )

        assert 0 < forward < backward
        assert block_activations.mamba_block_bytes(QWEN_05B, 1, 16384, 2.0) == 0

    def test_unfused_scan_adds_backward_bytes_per_inner_channel(self):
        arch = ModelArch.from_hf_config(NEMOTRON_NANO_MAMBA)
        d_inner = arch.mamba_n_heads * arch.mamba_d_head

        fused = block_activations.mamba_block_bytes(arch, 1, 16384, 2.0, backward=True)
        unfused = block_activations.mamba_block_bytes(
            arch, 1, 16384, 2.0, backward=True, unfused_scan=True
        )
        forward = block_activations.mamba_block_bytes(
            arch, 1, 16384, 2.0, unfused_scan=True
        )

        assert (
            unfused - fused
            == 16384 * d_inner * block_activations.UNFUSED_MAMBA_SCAN_BYTES
        )
        assert forward == block_activations.mamba_block_bytes(arch, 1, 16384, 2.0)

    def test_block_exclusive_recompute_is_the_widest_layer(self):
        arch = ModelArch.from_hf_config(NEMOTRON_NANO_MAMBA)

        per_layer = {
            kind: block_activations.block_recompute_bytes(
                arch, 1, 16384, 2.0, backward=True, layer=kind
            )
            for kind in formulas.block_kinds(arch)
        }
        widest = block_activations.block_recompute_bytes(
            arch, 1, 16384, 2.0, backward=True
        )

        assert set(per_layer) == {"attention", "ffn", "mamba"}
        assert per_layer["mamba"] == block_activations.mamba_block_bytes(
            arch, 1, 16384, 2.0, backward=True
        )
        assert widest == max(per_layer.values())

    def test_parallel_hybrid_block_adds_the_mamba_scan(self):
        falcon = ModelArch.from_hf_config(FALCON_H1)
        dense = falcon.model_copy(update={"n_mamba_layers": 0})

        with_scan = block_activations.block_recompute_bytes(falcon, 1, 1024, 2.0)
        without = block_activations.block_recompute_bytes(dense, 1, 1024, 2.0)

        assert with_scan - without == block_activations.mamba_block_bytes(
            falcon, 1, 1024, 2.0
        )
        assert not falcon.mamba_gated_delta


class TestGatedDeltaBlockBytes:
    def test_forward_matches_the_fla_kernel_snapshot(self):
        # Qwen3.5-0.8B no-grad forward at 16k: 31744 bf16 elements per token.
        arch = ModelArch.from_hf_config(QWEN35_08B.config)

        forward = block_activations.gated_delta_block_bytes(arch, 1, 16384, 2.0)

        assert forward == 16384 * 31744 * 2
        assert forward == 992 * MiB

    def test_backward_adds_the_in_proj_output_and_gate(self):
        arch = ModelArch.from_hf_config(QWEN35_08B.config)

        forward = block_activations.gated_delta_block_bytes(arch, 2, 4096, 2.0)
        backward = block_activations.gated_delta_block_bytes(
            arch, 2, 4096, 2.0, backward=True
        )

        assert backward - forward == 2 * 4096 * (6144 + 2048) * 2

    def test_mamba2_and_attention_only_stacks_are_zero(self):
        mamba2 = ModelArch.from_hf_config(NEMOTRON_NANO_MAMBA)

        assert block_activations.gated_delta_block_bytes(mamba2, 1, 16384, 2.0) == 0
        assert block_activations.gated_delta_block_bytes(QWEN_05B, 1, 16384, 2.0) == 0
        assert block_activations.gated_delta_saved_bytes(mamba2, 1, 16384, 2.0) == 0
        assert block_activations.gated_delta_saved_bytes(QWEN_05B, 1, 16384, 2.0) == 0

    def test_forward_charges_q_and_k_at_the_key_head_count(self):
        # Qwen3.6-35B-A3B: 16 key heads, 32 value heads, so q/k are half as wide as v.
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["Qwen/Qwen3.6-35B-A3B"].config
        )

        forward = block_activations.gated_delta_block_bytes(arch, 1, 1024, 2.0)

        assert block_activations.gated_delta_qk_width(arch) == 2048
        assert (
            forward == 1024 * (8192 + 4 * 2048 + 4096 + 5 * 4096 + 32 * 64 + 8192) * 2
        )
        assert forward == 100 * MiB

    def test_saved_set_matches_the_recompute_snapshot(self):
        # Qwen3.6-35B-A3B FSDP 32k: GDN tensors live in the MoE backward with
        # fla's fused gated RMSNorm and the l2-normed q and k at the key-head count.
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["Qwen/Qwen3.6-35B-A3B"].config
        )

        saved = block_activations.gated_delta_saved_bytes(arch, 1, 32768, 2.0)

        assert saved == 32768 * ((8192 + 2 * 2048 + 4 * 4096 + 32 * 64) * 2 + 32 * 4)
        assert saved == 1924 * MiB

    def test_tensor_parallel_keeps_each_ranks_heads(self):
        # Arrange: Qwen3.6-35B-A3B at tp=2 keeps 8 key and 16 value heads per rank.
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["Qwen/Qwen3.6-35B-A3B"].config
        )

        # Act
        forward = block_activations.gated_delta_block_bytes(
            arch, 1, 1024, 2.0, False, 2
        )
        saved = block_activations.gated_delta_saved_bytes(arch, 1, 32768, 2.0, 2)

        # Assert
        assert block_activations.gated_delta_qk_width(arch, 2) == 1024
        assert (
            forward == 1024 * (4096 + 4 * 1024 + 2048 + 5 * 2048 + 16 * 64 + 4096) * 2
        )
        assert forward == 50 * MiB
        assert saved == 32768 * ((4096 + 2 * 1024 + 4 * 2048 + 16 * 64) * 2 + 16 * 4)

    def test_tensor_parallel_that_does_not_split_the_heads_keeps_them_whole(self):
        arch = ModelArch.from_hf_config(QWEN35_08B.config)

        split = block_activations.gated_delta_block_bytes(arch, 1, 1024, 2.0, True, 3)

        assert split == block_activations.gated_delta_block_bytes(
            arch, 1, 1024, 2.0, True
        )

    def test_block_recompute_sizes_the_recurrent_layer_per_rank(self):
        # Arrange
        arch = ModelArch.from_hf_config(QWEN35_08B.config)
        tokens = 16384

        # Act
        forward = block_activations.block_recompute_bytes(
            arch, 1, tokens, 2.0, layer="mamba", tensor_parallel=2
        )

        # Assert
        assert forward == tokens * 4 * 1024 * 2 + 496 * MiB
        assert (
            forward
            == tokens * 4 * 1024 * 2
            + block_activations.gated_delta_block_bytes(
                arch, 1, tokens, 2.0, tensor_parallel=2
            )
        )

    def test_forward_peaks_in_the_recurrent_sub_layer(self):
        # Arrange
        arch = ModelArch.from_hf_config(QWEN35_08B.config)
        tokens = 16384

        # Act
        forward = block_activations.block_recompute_bytes(arch, 1, tokens, 2.0)
        nograd = block_activations.nograd_forward_bytes(arch, 1, tokens, 2.0)

        # Assert
        assert forward == tokens * 4 * 1024 * 2 + 992 * MiB
        assert nograd == forward + tokens * 1024 * 2
        assert nograd == pytest.approx(1.125 * GiB, rel=0.001)

    def test_backward_keeps_the_recurrent_tensors_through_the_mlp(self):
        # Arrange
        arch = ModelArch.from_hf_config(QWEN35_08B.config)
        tokens = 16384

        # Act
        backward = block_activations.block_recompute_bytes(
            arch, 1, tokens, 2.0, backward=True
        )
        mamba = block_activations.block_recompute_bytes(
            arch, 1, tokens, 2.0, backward=True, layer="mamba"
        )
        mlp = block_activations.block_recompute_bytes(
            arch, 1, tokens, 2.0, backward=True, layer="ffn"
        )

        # Assert: two residual copies, the gated MLP's four tensors, the GDN's saved set.
        assert backward == tokens * (2 * 1024 + 4 * 3584) * 2 + 800 * MiB
        assert backward == mamba
        assert mlp == tokens * (2 * 1024 + 4 * 3584) * 2

    def test_moe_backward_keeps_the_recurrent_tensors_through_the_experts(self):
        # Arrange
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["Qwen/Qwen3.6-35B-A3B"].config
        )
        tokens = 32768
        routed = formulas.routed_expert_bytes(arch, 1, tokens, 2.0, backward=True)

        # Act
        mamba = block_activations.block_recompute_bytes(
            arch, 1, tokens, 2.0, backward=True, layer="mamba"
        )

        # Assert: residual copies, the GDN's saved set, the shared expert, routed experts.
        assert routed == 3072 * MiB
        assert mamba == tokens * (2 * 2048 + 4 * 512) * 2 + 1924 * MiB + routed

    def test_backward_keeps_the_kernel_peak_when_the_mlp_is_narrow(self):
        # Arrange
        arch = ModelArch.from_hf_config(QWEN35_08B.config).model_copy(
            update={"intermediate_size": 256}
        )
        tokens = 16384

        # Act
        mamba = block_activations.block_recompute_bytes(
            arch, 1, tokens, 2.0, backward=True, layer="mamba"
        )

        # Assert: the kernel's backward outgrows the saved set plus a narrow MLP.
        assert mamba == tokens * 2 * 1024 * 2 + 1248 * MiB

    def test_block_exclusive_stack_keeps_one_sub_layer(self):
        arch = ModelArch.from_hf_config(QWEN35_08B.config).model_copy(
            update={"n_mlp_layers": 24}
        )
        tokens = 16384

        backward = block_activations.block_recompute_bytes(
            arch, 1, tokens, 2.0, backward=True
        )

        assert backward == tokens * 2 * 1024 * 2 + 1248 * MiB

    def test_lora_dropout_charges_the_recurrent_sub_layer_and_mlp(self):
        arch = ModelArch.from_hf_config(QWEN35_08B.config)

        dropout = formulas.lora_dropout_bytes(arch, 1, 16384, 2.0)
        mamba = formulas.lora_dropout_bytes(arch, 1, 16384, 2.0, layer="mamba")

        assert mamba == 16384 * (4 * 1024 + 2048 + 2 * 1024 + 3584) * 3
        assert dropout == mamba


class TestBlockRecomputeBackwardPeak:
    def test_rows_scale_the_block_linearly(self):
        one = block_activations.block_recompute_bytes(
            QWEN_05B, 1, 1024, 2.0, backward=True
        )
        four = block_activations.block_recompute_bytes(
            QWEN_05B, 4, 1024, 2.0, backward=True
        )

        assert four == 4 * one

    def test_gated_backward_keeps_two_residuals_and_four_mlp_tensors(self):
        # Arrange
        seq_len = 16384
        arch = QWEN_05B

        # Act
        backward = block_activations.block_recompute_bytes(
            arch, 1, seq_len, 2.0, backward=True
        )

        # Assert: gate, up, fused SwiGLU output, and one gradient.
        assert (
            backward
            == seq_len
            * (2 * arch.hidden_size + arch.peak_qkv_dim + 4 * arch.intermediate_size)
            * 2
        )

    def test_ungated_backward_doubles_the_two_mlp_matrices(self):
        seq_len = 1024
        arch = QWEN_05B.model_copy(update={"gated_mlp": False})

        forward = block_activations.block_recompute_bytes(arch, 1, seq_len, 2.0)
        backward = block_activations.block_recompute_bytes(
            arch, 1, seq_len, 2.0, backward=True
        )

        assert (
            forward
            == seq_len
            * (4 * arch.hidden_size + arch.peak_qkv_dim + 2 * arch.intermediate_size)
            * 2
        )
        assert (
            backward
            == seq_len
            * (2 * arch.hidden_size + arch.peak_qkv_dim + 4 * arch.intermediate_size)
            * 2
        )

    def test_forward_keeps_four_residuals_and_the_undoubled_mlp(self):
        seq_len = 16384
        arch = QWEN_05B

        forward = block_activations.block_recompute_bytes(arch, 1, seq_len, 2.0)

        assert (
            forward
            == seq_len
            * (4 * arch.hidden_size + arch.peak_qkv_dim + 3 * arch.intermediate_size)
            * 2
        )

    def test_qwen3_backward_adds_one_fp32_query(self):
        seq_len = 16384
        arch = ModelArch.from_hf_config(QWEN3_4B)
        without = arch.model_copy(update={"qk_norm": False})

        backward = block_activations.block_recompute_bytes(
            arch, 1, seq_len, 2.0, backward=True
        )
        base = block_activations.block_recompute_bytes(
            without, 1, seq_len, 2.0, backward=True
        )

        assert arch.qk_norm
        assert not without.qk_norm
        assert backward - base == seq_len * arch.n_heads * arch.head_dim * 4

    def test_deepseek_v4_backward_adds_one_fp32_query_per_rank(self):
        # Arrange
        arch = ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[DSV4_FLASH].config)
        without = arch.model_copy(update={"qk_norm": False})

        # Act
        backward = block_activations.block_recompute_bytes(
            arch, 1, 4096, 2.0, backward=True, tensor_parallel=4
        )
        base = block_activations.block_recompute_bytes(
            without, 1, 4096, 2.0, backward=True, tensor_parallel=4
        )

        # Assert: q_b_norm upcasts 16 of 64 512-wide query heads per tp4 rank.
        assert arch.qk_norm
        assert backward - base == 4096 * 16 * 512 * 4

    def test_qwen2_has_no_query_key_norm(self):
        qwen2 = dict(QWEN3_4B)
        qwen2["model_type"] = "qwen2"

        assert not ModelArch.from_hf_config(qwen2).qk_norm

    def test_qwen3_5_text_has_query_key_norm(self):
        qwen3_5 = dict(QWEN3_4B)
        qwen3_5["model_type"] = "qwen3_5_text"

        assert ModelArch.from_hf_config(qwen3_5).qk_norm


class TestBlockRecomputeNormInputs:
    def test_stock_rms_norm_backward_keeps_two_fp32_norm_inputs(self):
        # Arrange
        fused = ModelArch.from_hf_config(GRANITE_MOE)
        granite = fused.model_copy(update={"fp32_norm_inputs": True})

        # Act
        stock = block_activations.block_recompute_bytes(
            granite, 1, 65536, 2.0, backward=True
        )
        liger = block_activations.block_recompute_bytes(
            fused, 1, 65536, 2.0, backward=True
        )

        # Assert: a 64k-token stock-norm Granite MoE snapshot held two 0.375 GiB fp32 copies.
        assert stock - liger == 2 * 65536 * 1536 * 4
        assert stock - liger == 0.75 * GiB

    def test_liger_families_and_forward_pay_nothing(self):
        fused = ModelArch.from_hf_config(GRANITE_MOE)
        granite = fused.model_copy(update={"fp32_norm_inputs": True})

        assert not fused.fp32_norm_inputs
        assert not ModelArch.from_hf_config(QWEN3_4B).fp32_norm_inputs
        assert not ModelArch.from_hf_config(NEMOTRON_NANO_MAMBA).fp32_norm_inputs
        assert block_activations.block_recompute_bytes(
            granite, 1, 1024, 2.0
        ) == block_activations.block_recompute_bytes(fused, 1, 1024, 2.0)

    def test_block_exclusive_layer_keeps_one_norm_input(self):
        # A non-gated MLP keeps the stock MLP term out of the difference.
        nano = ModelArch.from_hf_config(NEMOTRON_NANO_MAMBA).model_copy(
            update={"gated_mlp": False}
        )
        stock = nano.model_copy(update={"fp32_norm_inputs": True})

        added = {
            kind: block_activations.block_recompute_bytes(
                stock, 1, 4096, 2.0, backward=True, layer=kind
            )
            - block_activations.block_recompute_bytes(
                nano, 1, 4096, 2.0, backward=True, layer=kind
            )
            for kind in formulas.block_kinds(nano)
        }

        assert added == dict.fromkeys(("attention", "ffn", "mamba"), 4096 * 3136 * 4)

    def test_gemma4_keeps_five_fp32_norm_inputs(self):
        # Arrange
        gemma = ModelArch.from_hf_config(GEMMA4_E4B.config)
        fused = gemma.model_copy(update={"fp32_norm_inputs": False})

        # Act
        stock = block_activations.block_recompute_bytes(
            gemma, 1, 16384, 2.0, backward=True
        )
        liger = block_activations.block_recompute_bytes(
            fused, 1, 16384, 2.0, backward=True
        )

        # Assert: four sandwich norms and the per-layer-input norm.
        assert gemma.fp32_norm_inputs
        assert gemma.norms_per_block == 5
        assert stock - liger == 5 * 16384 * 2560 * 4


class TestBlockRecomputeEagerScores:
    @pytest.mark.parametrize("backward", [True, False])
    def test_eager_attention_holds_three_score_tensors(self, backward):
        # Arrange
        eager = QWEN_05B.model_copy(update={"eager_attention": True})

        # Act
        with_scores = block_activations.block_recompute_bytes(
            eager, 2, 4096, 2.0, backward=backward
        )
        tiled = block_activations.block_recompute_bytes(
            QWEN_05B, 2, 4096, 2.0, backward=backward
        )

        # Assert: 3 copies of 2 rows x 14 heads x 4096 x 4096 bf16 scores.
        assert with_scores - tiled == 3 * 2 * 14 * 4096 * 4096 * 2

    def test_only_the_attention_sub_layer_pays(self):
        # Arrange
        nano = ModelArch.from_hf_config(NEMOTRON_NANO_MAMBA)
        eager = nano.model_copy(update={"eager_attention": True})

        # Act
        added = {
            kind: block_activations.block_recompute_bytes(
                eager, 1, 2048, 2.0, backward=True, layer=kind
            )
            - block_activations.block_recompute_bytes(
                nano, 1, 2048, 2.0, backward=True, layer=kind
            )
            for kind in formulas.block_kinds(nano)
        }

        # Assert
        assert added == {"attention": 3 * 40 * 2048 * 2048 * 2, "ffn": 0, "mamba": 0}

    def test_engine_forward_pays_nothing(self):
        eager = QWEN_05B.model_copy(update={"eager_attention": True})

        assert block_activations.block_recompute_bytes(
            eager, 1, 4096, 2.0, engine_forward=True
        ) == block_activations.block_recompute_bytes(
            QWEN_05B, 1, 4096, 2.0, engine_forward=True
        )


class TestCompressedEntryBytes:
    def test_entries_past_topk_hold_a_mask_byte_and_head_scores(self):
        arch = ModelArch.from_hf_config({**DSV4_TINY, "index_topk": 128})

        n_bytes = block_activations.compressed_entry_bytes(arch, 1, 1024, 1024)

        # 256 entries per query: a bool mask, and fp32 scores of 2 indexer
        # heads plus their sum.
        assert n_bytes == 1024 * 256 * (1 + 3 * 4)

    def test_topk_covering_every_entry_holds_nothing_per_query(self):
        arch = ModelArch.from_hf_config({**DSV4_TINY, "index_topk": 512})

        assert block_activations.compressed_entry_bytes(arch, 2, 1024, 1024) == 0

    def test_long_documents_score_queries_in_capped_chunks(self):
        # Arrange
        arch = ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[DSV4_FLASH].config)

        # Act
        n_bytes = block_activations.compressed_entry_bytes(arch, 1, 65536, 65536)

        # Assert: 16384 entries; 2^28 scores cover 256 queries of 64 heads,
        # each with the fp32 head sum.
        assert n_bytes == 65536 * 16384 + 256 * 16384 * 65 * 4

    def test_packed_documents_score_one_document_but_mask_the_whole_row(self):
        arch = ModelArch.from_hf_config({**DSV4_TINY, "index_topk": 128})

        n_bytes = block_activations.compressed_entry_bytes(arch, 1, 1536, 768)

        # A bool per query and each of the row's 384 entries; one 768-token
        # document's 192 entries scored at once.
        assert n_bytes == 1536 * 384 + 768 * 192 * 3 * 4

    def test_packed_documents_within_topk_hold_nothing_past_a_long_row(self):
        arch = ModelArch.from_hf_config({**DSV4_TINY, "index_topk": 256})

        assert block_activations.compressed_entry_bytes(arch, 1, 1536, 768) == 0

    def test_model_without_csa_layers_pays_nothing(self):
        assert block_activations.compressed_entry_bytes(QWEN_05B, 1, 4096, 4096) == 0

    def test_context_parallel_queries_score_every_shards_entries(self):
        arch = ModelArch.from_hf_config({**DSV4_TINY, "index_topk": 128})

        n_bytes = block_activations.compressed_entry_bytes(arch, 1, 512, 256, 2)

        # 512 local queries over the 256 entries of the whole 1024 tokens,
        # which context parallel treats as one document.
        assert n_bytes == 512 * 256 * (1 + 3 * 4)


class TestContextParallelKeyBytes:
    def test_deepseek_v4_gathers_sliding_kv_entries_and_indexer_keys(self):
        # Arrange
        arch = ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[DSV4_FLASH].config)
        # Sliding KV 512 and CSA entries 512 / 4, three copies; indexer keys
        # 128 / 4. The last 7 tokens of each shard's CSA KV and gate
        # projections 2048 and indexer projections 4 * 128.
        keys = 512 + 128
        boundary = 2 * 7 * (2048 + 512)

        # Act
        forward = block_activations.context_parallel_key_bytes(
            arch, 2, 4096, 2.0, context_parallel=2
        )
        backward = block_activations.context_parallel_key_bytes(
            arch, 2, 4096, 2.0, backward=True, context_parallel=2
        )

        # Assert
        assert forward == 2 * (4096 * (3 * keys + 32) + boundary) * 2
        assert backward == 2 * (4096 * (3 * keys + 2 * 32) + boundary) * 2

    def test_deepseek_v4_skips_the_indexer_when_topk_keeps_every_entry(self):
        arch = ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[DSV4_FLASH].config)

        n_bytes = block_activations.context_parallel_key_bytes(
            arch, 1, 2048, 2.0, context_parallel=2
        )

        assert n_bytes == (2048 * 3 * (512 + 128) + 2 * 7 * 2048) * 2

    def test_hca_gathers_boundary_tokens_only_when_a_window_straddles_shards(self):
        # Arrange
        hca = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO[DSV4_FLASH].config
        ).model_copy(update={"n_csa_layers": 0})
        keys = 512 + 512 / 128

        # Act
        aligned = block_activations.context_parallel_key_bytes(
            hca, 1, 4096, 2.0, context_parallel=2
        )
        straddling = block_activations.context_parallel_key_bytes(
            hca, 1, 2000, 2.0, context_parallel=2
        )

        # Assert: 2048-token shards end on a 128-token window; 1000-token
        # shards send their last 127 tokens of KV and gate projections 1024.
        assert aligned == 4096 * 3 * keys * 2
        assert straddling == (2000 * 3 * keys + 2 * 127 * 1024) * 2

    def test_glm_gathers_the_latent_and_expands_local_heads(self):
        # Arrange
        arch = ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[GLM_FLASH].config)
        # Latent 512; 64 / 2 heads of key 256 and value 256.
        with_grad = 512 + 32 * (256 + 256)

        # Act
        forward = block_activations.context_parallel_key_bytes(
            arch, 1, 8192, 2.0, tensor_parallel=2
        )
        backward = block_activations.context_parallel_key_bytes(
            arch, 1, 8192, 2.0, tensor_parallel=2, backward=True, context_parallel=2
        )

        # Assert
        # Packed no-grad indexer keys: key, gate and validity.
        assert forward == 8192 * (with_grad + 2 * 128 + 1) * 2
        # Each tensor and its gradient, the reduce-scatter's stacked latent, and
        # the half of it this rank receives.
        assert backward == 8192 * (2 * with_grad + 512 + 512 / 2) * 2

    def test_uncompressed_key_gather_holds_the_whole_sequence_kv(self):
        # Arrange
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO[DSV4_FLASH].config
        ).model_copy(update={"n_csa_layers": 0, "n_hca_layers": 0})

        # Act
        forward = block_activations.context_parallel_key_bytes(
            arch, 1, 4096, 2.0, context_parallel=2
        )
        backward = block_activations.context_parallel_key_bytes(
            arch, 1, 4096, 2.0, backward=True, context_parallel=2
        )

        # Assert: one 512-wide KV head; the backward holds four copies.
        assert forward == 4096 * 512 * 2
        assert backward == 4096 * 4 * 512 * 2

    def test_ulysses_family_pays_nothing(self):
        arch = ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[MIMO_FLASH].config)

        assert block_activations.context_parallel_key_bytes(arch, 1, 8192, 2.0) == 0


class TestPooledIndexBytes:
    def test_local_queries_score_the_pools_of_every_shard(self):
        arch = ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[GLM_FLASH].config)

        n_bytes = block_activations.pooled_index_bytes(arch, 2, 1024, 2)

        # 2048 keys in 512 pools: three fp32 scores per indexer head and pool,
        # plus a visibility byte per key.
        assert n_bytes == 2 * 1024 * (3 * 32 * 512 * 4 + 2048)

    def test_top_k_mask_holds_one_bool_per_key(self):
        # Arrange
        glm = ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[GLM_FLASH].config)
        narrow = glm.model_copy(update={"index_n_heads": 1, "index_kpool": 64})

        # Act
        n_bytes = block_activations.pooled_index_bytes(narrow, 1, 2048)

        # Assert: 32 pools of one head's three fp32 scores, and the bool mask
        # over 2048 keys.
        assert n_bytes == 2048 * (3 * 32 * 4 + 2048)

    def test_family_without_kpool_pays_nothing(self):
        arch = ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[DSV4_FLASH].config)

        assert block_activations.pooled_index_bytes(arch, 1, 4096, 2) == 0


class TestBlockRecomputeContextParallel:
    def test_key_gather_block_adds_the_whole_sequence_key_side(self):
        # Arrange
        arch = ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[GLM_FLASH].config)
        ulysses = arch.model_copy(update={"context_parallel_key_gather": False})
        local_kv = formulas.qkv_width_per_rank(arch, 1) - arch.n_heads * arch.head_dim

        # Act
        gathered = block_activations.block_recompute_bytes(
            arch, 1, 2048, 2.0, layer="attention", context_parallel=2
        )
        local = block_activations.block_recompute_bytes(
            ulysses, 1, 2048, 2.0, layer="attention", context_parallel=2
        )

        # Assert
        assert gathered - local == block_activations.context_parallel_key_bytes(
            arch, 1, 4096, 2.0
        ) - (2048 * local_kv * 2)


class TestBlockRecomputeCompressedEntries:
    @pytest.mark.parametrize("backward", [True, False])
    def test_block_holds_the_indexer_selection_of_its_documents(self, backward):
        # Arrange
        arch = ModelArch.from_hf_config({**DSV4_TINY, "index_topk": 128})
        without_csa = arch.model_copy(update={"n_csa_layers": 0})

        # Act
        with_entries = block_activations.block_recompute_bytes(
            arch, 1, 1536, 2.0, backward=backward, document_len=768
        )
        without_entries = block_activations.block_recompute_bytes(
            without_csa, 1, 1536, 2.0, backward=backward, document_len=768
        )

        # Assert
        assert with_entries - without_entries == 1536 * 384 + 768 * 192 * 3 * 4

    def test_document_len_defaults_to_the_sequence(self):
        arch = ModelArch.from_hf_config({**DSV4_TINY, "index_topk": 512})

        assert block_activations.block_recompute_bytes(
            arch, 1, 1024, 2.0, backward=True
        ) == block_activations.block_recompute_bytes(
            arch, 1, 1024, 2.0, backward=True, document_len=1024
        )


class TestBlockRecomputeStockGatedMlp:
    def test_stock_mlp_backward_keeps_the_activated_gate(self):
        # Arrange
        fused = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["ibm-granite/granite-4.0-micro"].config
        )
        granite = fused.model_copy(update={"stock_gated_mlp": True})

        # Act
        stock = block_activations.block_recompute_bytes(
            granite, 1, 32768, 2.0, backward=True
        )
        liger = block_activations.block_recompute_bytes(
            fused, 1, 32768, 2.0, backward=True
        )

        # Assert: a 32k-token stock-MLP granite-4.0-micro snapshot held a 0.5 GiB SiLU output.
        assert stock - liger == 32768 * 8192 * 2
        assert stock - liger == 0.5 * GiB

    def test_forward_and_liger_families_pay_nothing(self):
        fused = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["ibm-granite/granite-4.0-micro"].config
        )
        granite = fused.model_copy(update={"stock_gated_mlp": True})

        assert not fused.stock_gated_mlp
        assert not ModelArch.from_hf_config(QWEN3_4B).stock_gated_mlp
        assert block_activations.block_recompute_bytes(
            granite, 1, 1024, 2.0
        ) == block_activations.block_recompute_bytes(fused, 1, 1024, 2.0)


class TestBlockRecomputeEngineHyperConnections:
    def test_engine_forward_carries_the_streams_and_their_fp32_mix(self):
        # Arrange
        streams = QWEN_05B.model_copy(update={"hc_mult": 4})

        # Act
        mixed = block_activations.block_recompute_bytes(
            streams, 1, 8192, 2.0, engine_forward=True
        )
        plain = block_activations.block_recompute_bytes(
            QWEN_05B, 1, 8192, 2.0, engine_forward=True
        )

        # Assert: 3 extra streams in and out in bf16, 4 streams in fp32.
        h = QWEN_05B.hidden_size
        assert mixed - plain == 8192 * (2 * 3 * h * 2 + 4 * h * 4)

    def test_training_forward_ignores_the_engine_mix(self):
        streams = QWEN_05B.model_copy(update={"hc_mult": 4})

        assert block_activations.block_recompute_bytes(
            streams, 1, 1024, 2.0
        ) == block_activations.block_recompute_bytes(QWEN_05B, 1, 1024, 2.0)


class TestBlockRecomputeTrainerHyperConnections:
    def test_backward_keeps_every_stream(self):
        # Arrange
        streams = QWEN_05B.model_copy(update={"hc_mult": 4})

        # Act
        mixed = block_activations.block_recompute_bytes(
            streams, 1, 8192, 2.0, backward=True
        )
        plain = block_activations.block_recompute_bytes(
            QWEN_05B, 1, 8192, 2.0, backward=True
        )

        # Assert: 3 extra bf16 streams in the input and mid-block residuals;
        # the mixer's fp32 copy fits under the block's other transients.
        h = QWEN_05B.hidden_size
        assert mixed - plain == 8192 * (2 * 3 * h * 2)

    def test_wide_streams_peak_in_the_mixer_backward(self):
        # Arrange
        streams = QWEN_05B.model_copy(update={"hc_mult": 16})

        # Act
        block = block_activations.block_recompute_bytes(
            streams, 1, 8192, 2.0, backward=True
        )

        # Assert: both bf16 residuals of 16 streams, and the mixer's fp32
        # copy of all 16.
        h = QWEN_05B.hidden_size
        assert block == 8192 * (2 * 16 * h * 2 + 16 * h * 4)

    def test_sublayer_backward_holds_no_fp32_mix_beside_the_ffn(self):
        # Arrange
        arch = ModelArch.from_hf_config(GLM_TINY)
        single = arch.model_copy(update={"hc_mult": 1})

        # Act
        mixed = block_activations.block_recompute_bytes(
            arch, 1, 1024, 2.0, backward=True, layer="ffn"
        )
        plain = block_activations.block_recompute_bytes(
            single, 1, 1024, 2.0, backward=True, layer="ffn"
        )

        # Assert: one extra 64-wide bf16 stream in both residuals.
        assert arch.sublayer_transients
        assert mixed - plain == 1024 * (2 * 64 * 2)

    def test_wide_sublayer_streams_peak_in_the_mixer_backward(self):
        # Arrange
        arch = ModelArch.from_hf_config(GLM_TINY).model_copy(update={"hc_mult": 8})

        # Act
        ffn = block_activations.block_recompute_bytes(
            arch, 1, 1024, 2.0, backward=True, layer="ffn"
        )

        # Assert: both bf16 residuals of 8 64-wide streams, and the mixer's
        # fp32 copy of all 8.
        assert ffn == 1024 * (2 * 512 * 2 + 512 * 4)

    def test_deepseek_v4_flash_block_at_tp4(self):
        # Arrange
        arch = ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[DSV4_FLASH].config)
        plain = arch.model_copy(update={"hc_mult": 1, "qk_norm": False})

        # Act
        block = block_activations.block_recompute_bytes(
            arch, 1, 4096, 2.0, backward=True, tensor_parallel=4, document_len=768
        )
        base = block_activations.block_recompute_bytes(
            plain, 1, 4096, 2.0, backward=True, tensor_parallel=4, document_len=768
        )

        # Assert: 4096 packed tokens hold 3 extra 4096-wide bf16 streams twice
        # and 16 fp32 512-wide query heads.
        assert block - base == 4096 * (2 * 3 * 4096 * 2 + 16 * 512 * 4)
        assert block - base == 0.3125 * GiB


class TestLoraDropoutBytes:
    def test_matches_the_fp32_and_bf16_adapter_snapshots(self):
        arch = ModelArch.from_hf_config(QWEN3_4B)
        width = 3 * 2560 + 32 * 128 + 2 * 2560 + 9728

        fp32 = formulas.lora_dropout_bytes(arch, 1, 16384, 4.0)
        bf16 = formulas.lora_dropout_bytes(arch, 1, 16384, 2.0)

        # Dropped-out input at the adapter dtype plus a one-byte mask.
        assert fp32 == 16384 * width * 5
        assert bf16 == 16384 * width * 3
        assert fp32 == pytest.approx(2.031 * GiB, rel=0.01)
        assert bf16 == pytest.approx(1.219 * GiB, rel=0.01)

    def test_attention_only_scope_skips_the_mlp_inputs(self):
        arch = ModelArch.from_hf_config(QWEN3_4B)

        attention = formulas.lora_dropout_bytes(arch, 1, 1024, 2.0, "attention-only")

        assert attention == 1024 * (3 * 2560 + 32 * 128) * 3

    def test_excluded_qkv_keep_no_dropped_inputs(self):
        arch = ModelArch.from_hf_config(MIMO_TINY)

        attention = formulas.lora_dropout_bytes(arch, 1, 1024, 2.0, "attention-only")

        # o_proj alone reads the four 16-wide heads.
        assert attention == 1024 * 4 * 16 * 3

    def test_block_exclusive_stack_charges_its_widest_layer(self):
        # Nemotron-H: one mixer or MLP per layer; the MLP layer is widest.
        arch = ModelArch.from_hf_config(NEMOTRON_H)
        h, inter = NEMOTRON_H["hidden_size"], NEMOTRON_H["intermediate_size"]
        mlp_width = (2 * h if arch.gated_mlp else h) + inter

        dropout = formulas.lora_dropout_bytes(arch, 1, 1024, 4.0)

        assert arch.block_exclusive_layers
        assert dropout == 1024 * mlp_width * 5

    def test_parallel_hybrid_includes_mamba_in_proj(self):
        arch = ModelArch.from_hf_config(FALCON_H1)
        h = arch.hidden_size
        attention = 3 * h + arch.n_heads * arch.head_dim
        mlp = (2 * h if arch.gated_mlp else h) + arch.intermediate_size
        mamba = h

        dropout = formulas.lora_dropout_bytes(arch, 1, 16384, 4.0)

        assert not arch.block_exclusive_layers
        assert dropout == 16384 * (attention + mlp + mamba) * 5
        assert dropout - 16384 * (attention + mlp) * 5 == 16384 * mamba * 5

    def test_low_rank_attention_and_kda_read_their_own_linears(self):
        # Arrange
        arch = ModelArch.from_hf_config(GLM_TINY)
        # q_a, q_b, kv_a, kv_b and o inputs per MLA layer.
        mla = 64 + 32 + 64 + 16 + 32
        # q, k, v, f_b, beta, g_b and o inputs per KDA layer; f_a and g_a
        # stay unwrapped.
        kda = 3 * 64 + 8 + 64 + 8 + 16
        mlp = 2 * 64 + 96

        # Act
        attention = formulas.lora_dropout_bytes(arch, 1, 1024, 2.0, layer="attention")
        mamba = formulas.lora_dropout_bytes(arch, 1, 1024, 2.0, layer="mamba")

        # Assert: each mixer also charges the MLP beside it.
        assert attention == 1024 * (mla + mlp) * 3
        assert mamba == 1024 * (kda + mlp) * 3


class TestAllocatorReserveBytes:
    def test_is_a_markup_on_allocated_bytes(self):
        assert formulas.allocator_reserve_bytes(0) == 0
        assert formulas.allocator_reserve_bytes(100 * MiB) == pytest.approx(7 * MiB)
        # Floor at zero.
        assert formulas.allocator_reserve_bytes(-1 * MiB) == 0

    def test_fsdp_compute_stream_reserves_less_than_flat_data_parallel(self):
        data_parallel = formulas.allocator_reserve_bytes(100 * MiB, n_ranks=4)
        fsdp = formulas.allocator_reserve_bytes(100 * MiB, n_ranks=4, sharded=True)

        assert data_parallel == pytest.approx(5.4 * MiB)
        assert fsdp == pytest.approx(2 * MiB)

    def test_mixed_tensor_sizes_reserve_more_under_fsdp(self):
        mixed = formulas.allocator_reserve_bytes(
            100 * MiB, n_ranks=4, sharded=True, mixed_sizes=True
        )
        unsharded = formulas.allocator_reserve_bytes(
            100 * MiB, n_ranks=4, mixed_sizes=True
        )

        assert mixed == pytest.approx(6.4 * MiB)
        assert unsharded == pytest.approx(5.4 * MiB)


class TestRecommendationFits:
    def test_keeps_the_underprediction_share_of_usable_memory_free(self):
        usable = 10_000
        limit = int(usable * (1 - formulas.MAX_UNDERPREDICTION))

        assert formulas.recommendation_fits(limit, usable)
        assert not formulas.recommendation_fits(limit + 1, usable)


class TestMambaStateBytes:
    def test_charges_padded_pages_over_padded_groups(self):
        arch = ModelArch.from_hf_config(NEMOTRON_H)

        state = kv_cache.mamba_state_bytes(arch, 16)

        # 27 Mamba layers pad to 7 groups of 4; each page pads to 1312 tokens
        # of attention KV; Mamba-2 holds 3 blocks per sequence.
        assert state == 7 * 4 * 1312 * 4096 * 16 * 3
        assert kv_cache.mamba_page_bytes(QWEN_05B) == 0
        assert kv_cache.mamba_state_bytes(QWEN_05B, 16) == 0

    def test_gated_delta_holds_two_blocks_and_mamba2_three(self):
        gated = ModelArch.from_hf_config(
            {
                **DECODER_TINY,
                "layer_types": ["full_attention", "linear"] * 4,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 64,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "linear_conv_kernel_dim": 4,
            }
        )

        assert kv_cache.mamba_state_blocks_per_seq(gated) == 2
        assert (
            kv_cache.mamba_state_blocks_per_seq(ModelArch.from_hf_config(NEMOTRON_H))
            == 3
        )


class TestHybridKvGroups:
    def test_groups_take_the_smaller_layer_count(self):
        arch = ModelArch.from_hf_config(NEMOTRON_H)

        assert kv_cache.hybrid_kv_groups(arch) == (4, 1, 7)

    def test_groups_take_the_larger_count_when_close(self):
        falcon = ModelArch.from_hf_config(FALCON_H1)
        close = falcon.model_copy(
            update={"n_attention_layers": 12, "n_mamba_layers": 13}
        )

        assert kv_cache.hybrid_kv_groups(falcon) == (24, 1, 1)
        assert kv_cache.hybrid_kv_groups(close) == (13, 1, 1)


class TestHybridKvDemandBytes:
    def test_each_sequence_rounds_up_to_whole_blocks(self):
        arch = ModelArch.from_hf_config(NEMOTRON_H)
        page = kv_cache.hybrid_page_bytes(arch)

        one_token = kv_cache.hybrid_kv_demand_bytes(arch, "bf16", 2, 1)
        one_block = kv_cache.hybrid_kv_demand_bytes(arch, "bf16", 2, 1312)
        past_block = kv_cache.hybrid_kv_demand_bytes(arch, "bf16", 2, 1313)

        assert one_token == one_block == 2 * 4 * page
        assert past_block == 2 * one_block

    def test_indexer_keys_grow_with_every_token(self):
        # Arrange
        arch = ModelArch.from_hf_config(GLM_TINY)
        unindexed = arch.model_copy(update={"n_indexed_layers": 0})

        # Act
        indexed = kv_cache.hybrid_kv_demand_bytes(arch, "bf16", 2, 1024)
        paged = kv_cache.hybrid_kv_demand_bytes(unindexed, "bf16", 2, 1024)

        # Assert: two sequences of 1024 tokens, two layers of 8-wide keys.
        assert indexed - paged == 2 * 1024 * 2 * 8 * 2

    def test_tensor_parallel_shrinks_the_page(self):
        arch = ModelArch.from_hf_config(NEMOTRON_H)

        assert kv_cache.hybrid_page_bytes(arch, tensor_parallel=2) == (
            kv_cache.aligned_kv_block_size(arch, tensor_parallel=2)
            * kv_cache.attention_token_bytes(arch, tensor_parallel=2)
        )
        assert kv_cache.attention_token_bytes(arch, tensor_parallel=2) == (
            kv_cache.attention_token_bytes(arch) // 2
        )


class TestMambaPagePerRank:
    def test_tp_one_matches_the_unsharded_page(self):
        arch = ModelArch.from_hf_config(NEMOTRON_H)

        assert kv_cache.mamba_page_per_rank(arch) == kv_cache.mamba_page_bytes(arch)
        assert kv_cache.mamba_page_per_rank(QWEN_05B, tensor_parallel=4) == 0

    def test_heads_must_divide_tensor_parallel(self):
        arch = ModelArch.from_hf_config(NEMOTRON_H)

        with pytest.raises(
            ValueError,
            match=r"mamba_num_heads \(128\) must be divisible by "
            r"tensor_parallel_size \(3\)",
        ):
            kv_cache.mamba_page_per_rank(arch, tensor_parallel=3)

    def test_groups_must_divide_or_be_one(self):
        arch = ModelArch.from_hf_config(NEMOTRON_H)

        with pytest.raises(
            ValueError,
            match=r"mamba n_groups \(8\) must divide tensor_parallel_size \(16\)",
        ):
            kv_cache.mamba_page_per_rank(arch, tensor_parallel=16)

    def test_a_single_group_replicates_to_one_per_rank(self):
        # Falcon-H1 publishes one Mamba group. vLLM copies it onto every rank.
        arch = ModelArch.from_hf_config(FALCON_H1)
        tp = 4
        groups = tp
        conv_dim = (
            arch.mamba_n_heads * arch.mamba_d_head + 2 * groups * arch.mamba_d_state
        ) // tp
        recurrent = (arch.mamba_n_heads // tp) * arch.mamba_d_head * arch.mamba_d_state
        expected = conv_dim * max(arch.mamba_d_conv - 1, 0) * 2 + recurrent * 2

        page = kv_cache.mamba_page_per_rank(arch, tensor_parallel=tp)

        assert arch.mamba_n_groups == 1
        assert page == expected
        assert 0 < page < kv_cache.mamba_page_bytes(arch)


class TestAlignedKvBlockSize:
    def test_matches_vllm(self):
        """SSM state dtype changes the Mamba page and the aligned KV block size."""
        falcon = ModelArch.from_hf_config(FALCON_H1).model_copy(
            update={"mamba_ssm_state_dtype": "bf16"}
        )
        assert kv_cache.mamba_page_bytes(falcon) == 1_594_368
        assert kv_cache.aligned_kv_block_size(falcon) == 1568

        nemotron = ModelArch.from_hf_config(NEMOTRON_H)  # fp32 SSM state, the default
        assert kv_cache.mamba_page_bytes(nemotron) == 5_316_608
        assert kv_cache.aligned_kv_block_size(nemotron) == 1312

        # A dense model keeps vLLM's default.
        assert (
            kv_cache.aligned_kv_block_size(QWEN_05B) == kv_cache.KV_BLOCK_SIZE_DEFAULT
        )

    def test_falls_back_when_attention_layers_store_no_kv(self):
        hybrid = ModelArch.from_hf_config(FALCON_H1).model_copy(
            update={"n_kv_shared_layers": 24}
        )

        assert hybrid.is_hybrid_ssm
        assert kv_cache.kv_storing_layers(hybrid) == 0
        assert kv_cache.aligned_kv_block_size(hybrid) == kv_cache.KV_BLOCK_SIZE_DEFAULT


MOE_TINY = ModelArch(
    n_layers=2,
    hidden_size=16,
    intermediate_size=32,
    n_heads=2,
    n_kv_heads=2,
    head_dim=8,
    vocab_size=32,
    n_experts=4,
    n_experts_per_tok=2,
    expert_intermediate_size=32,
)


class TestTensorParallelWidths:
    def test_query_splits_and_indivisible_kv_stays_whole(self):
        assert formulas.qkv_width_per_rank(QWEN_05B, 1) == QWEN_05B.peak_qkv_dim
        assert formulas.qkv_width_per_rank(QWEN_05B, 2) == QWEN_05B.peak_qkv_dim // 2

        replicated_kv = QWEN_05B.model_copy(update={"n_kv_heads": 1, "n_heads": 14})
        full = formulas.qkv_width_per_rank(replicated_kv, 1)
        local = formulas.qkv_width_per_rank(replicated_kv, 2)
        # Query halves. The single KV head is copied onto both ranks.
        assert local == full - replicated_kv.n_heads * replicated_kv.head_dim // 2

    def test_ulysses_replicates_kv_heads_up_to_the_cp_degree(self):
        # Two KV heads split one per TP rank; Ulysses cp=2 copies it to two.
        q = QWEN_05B.n_heads // 2 * QWEN_05B.head_dim
        kv_head = 2 * QWEN_05B.head_dim

        assert formulas.qkv_width_per_rank(QWEN_05B, 2, 2) == q + 2 * kv_head
        assert formulas.qkv_width_per_rank(QWEN_05B, 1, 2) == QWEN_05B.peak_qkv_dim

    def test_qkv_width_uses_the_wider_global_head_dim(self):
        arch = QWEN_05B.model_copy(update={"global_head_dim": 128})

        assert formulas.qkv_width_per_rank(arch, 1) == arch.peak_qkv_dim
        assert formulas.qkv_width_per_rank(arch, 2) == arch.peak_qkv_dim // 2

    def test_tensor_parallel_linears_leave_the_fsdp_gather(self):
        arch = QWEN_05B.model_copy(update={"tensor_parallel_plan": True})
        counts = formulas.param_counts(arch)
        kwargs = {
            "n_gpus": 4,
            "reshard_after_forward": True,
        }

        full = fsdp_gather.fsdp_gathered_params(counts, arch, **kwargs)
        narrowed = fsdp_gather.fsdp_gathered_params(
            counts, arch, tensor_parallel=2, **kwargs
        )

        assert formulas.tensor_parallel_params(counts, arch, 2) == (
            counts.attention + counts.mlp
        )
        assert narrowed < full

    def test_without_a_plan_an_untied_head_gathers_one_over_tp(self):
        arch = QWEN_05B.model_copy(update={"tied_embeddings": False})
        counts = formulas.param_counts(arch)
        kwargs = {
            "n_gpus": 4,
            "reshard_after_forward": True,
        }

        full = fsdp_gather.fsdp_gathered_params(counts, arch, **kwargs)
        narrowed = fsdp_gather.fsdp_gathered_params(
            counts, arch, tensor_parallel=2, **kwargs
        )

        assert formulas.tensor_parallel_params(counts, arch, 2) == counts.lm_head
        assert narrowed == full - counts.lm_head // 2

    def test_tp_planned_hybrid_gathers_its_tp_layers_at_one_over_tp(self):
        # Arrange
        arch = ModelArch.from_hf_config(NEMOTRON_H_MOE)
        counts = formulas.param_counts(arch)
        moe_layer = fsdp_gather.layer_params(arch, "moe")
        mamba_layer = fsdp_gather.layer_params(arch, "mamba")

        # Act
        allocated = fsdp_gather.prefetch_window_params(counts, arch, 2, 2)
        pool = fsdp_gather.fsdp_gather_pool_params(
            counts, arch, 1, 4, tensor_parallel=2
        )

        # Assert: the widest window is a MoE layer beside half a Mamba layer;
        # the untied head's half beside the last MoE unit sets the pool.
        assert arch.tensor_parallel_plan
        assert allocated == moe_layer + mamba_layer // 2
        assert pool == counts.lm_head // 2 + moe_layer


class TestLoraTensorPlacement:
    def test_flat_dp_replicates_every_tensor(self):
        replicated, sharded = lora_tensor_placement(QWEN_05B, 16, "all-linear", 0, None)

        assert sharded == 0
        assert replicated == lora_formulas.lora_param_count(QWEN_05B, 16)

    def test_wide_mlp_matrices_cross_the_persistence_threshold(self):
        arch = ModelArch(
            n_layers=1,
            hidden_size=4096,
            intermediate_size=11008,
            n_heads=32,
            n_kv_heads=8,
            head_dim=128,
            vocab_size=128,
        )
        replicated, sharded = lora_tensor_placement(arch, 16, "all-linear", 0, 100_000)

        assert sharded == 3 * 16 * 11008
        assert replicated + sharded == lora_formulas.lora_param_count(arch, 16)
        attention_only, attention_sharded = lora_tensor_placement(
            arch, 16, "attention-only", 0, 100_000
        )
        assert attention_sharded == 0
        assert attention_only == lora_formulas.lora_param_count(
            arch, 16, "attention-only"
        )

    def test_threshold_zero_shards_every_tensor(self):
        replicated, sharded = lora_tensor_placement(QWEN_05B, 16, "all-linear", 0, 0)

        assert replicated == 0
        assert sharded == lora_formulas.lora_param_count(QWEN_05B, 16)

    def test_zero_rank_is_empty(self):
        replicated, sharded = lora_tensor_placement(QWEN_05B, 0, "all-linear", 0, None)

        assert (replicated, sharded) == (0, 0)

    def test_ungated_mlp_wraps_two_matrices(self):
        gated = ModelArch(
            n_layers=1,
            hidden_size=64,
            intermediate_size=128,
            n_heads=4,
            n_kv_heads=2,
            head_dim=16,
            vocab_size=32,
        )
        ungated = gated.model_copy(update={"gated_mlp": False})
        rank = 4
        gated_repl, gated_shard = lora_tensor_placement(
            gated, rank, "all-linear", 0, None
        )
        ungated_repl, ungated_shard = lora_tensor_placement(
            ungated, rank, "all-linear", 0, None
        )

        assert gated_shard == ungated_shard == 0
        assert gated_repl - ungated_repl == rank * (64 + 128)

    def test_tensor_parallel_splits_column_lora_b_and_row_lora_a(self):
        # Arrange
        arch = ModelArch(
            n_layers=1,
            hidden_size=64,
            intermediate_size=128,
            n_heads=4,
            n_kv_heads=2,
            head_dim=16,
            vocab_size=32,
        )
        rank = 4
        # Halved: q/k/v and gate/up B, o and down A.
        split = rank * (64 + 2 * 32 + 64 + 2 * 128 + 128)

        # Act
        replicated, sharded = lora_tensor_placement(
            arch, rank, "all-linear", 0, None, 2
        )

        # Assert
        assert sharded == 0
        assert replicated == lora_formulas.lora_param_count(arch, rank) - split // 2

    def test_tensor_parallel_keeps_indivisible_kv_lora_b_whole(self):
        # Arrange
        arch = ModelArch(
            n_layers=1,
            hidden_size=64,
            intermediate_size=128,
            n_heads=4,
            n_kv_heads=1,
            head_dim=16,
            vocab_size=32,
        )
        rank = 4

        # Act
        replicated, _ = lora_tensor_placement(arch, rank, "attention-only", 0, None, 2)

        # Assert: only q B and o A halve.
        expected = lora_formulas.lora_param_count(arch, rank, "attention-only")
        assert replicated == expected - rank * (64 + 64) // 2

    def test_tensor_parallel_splits_recurrent_projections(self):
        # Arrange
        arch = ModelArch(
            n_layers=2,
            hidden_size=64,
            intermediate_size=128,
            n_heads=4,
            n_kv_heads=2,
            head_dim=16,
            vocab_size=32,
            n_attention_layers=0,
            n_mlp_layers=0,
            n_mamba_layers=2,
            mamba_n_heads=4,
            mamba_d_head=16,
            mamba_conv_dim=96,
        )
        rank = 4
        # in_proj B rows: d_inner + conv_dim + n_heads; out_proj A: d_inner.
        split = 2 * rank * ((64 + 96 + 4) + 64)

        # Act
        replicated, _ = lora_tensor_placement(arch, rank, "all-linear", 0, None, 2)

        # Assert
        assert replicated == lora_formulas.lora_param_count(arch, rank) - split // 2

    def test_tensor_parallel_bins_by_the_global_matrix(self):
        # Arrange
        arch = ModelArch(
            n_layers=1,
            hidden_size=4096,
            intermediate_size=11008,
            n_heads=32,
            n_kv_heads=8,
            head_dim=128,
            vocab_size=128,
        )

        # Act
        _, sharded = lora_tensor_placement(arch, 16, "all-linear", 0, 100_000, 2)

        # Assert: each MLP factor on ``inter`` is 176k globally, 88k per rank.
        assert sharded == 3 * 16 * 11008 // 2

    def test_tensor_parallel_splits_gemma4_wide_heads(self):
        # Arrange
        arch = ModelArch.from_hf_config(GEMMA4_E4B.config)

        # Act
        whole, _ = lora_tensor_placement(arch, 16, "attention-only", 0, None, 1)
        split, _ = lora_tensor_placement(arch, 16, "attention-only", 0, None, 2)

        # Assert: half of q/k/v B and o A on both head widths.
        assert whole == 16 * 567_296
        assert split == 16 * (567_296 - 229_376 // 2)

    @pytest.mark.parametrize(
        ("tp", "expected"), [(1, 16 * 40 * 7680), (2, 16 * 40 * (7680 - 1536 // 2))]
    )
    def test_qwen36_shared_expert_splits_like_its_mlp(self, tp, expected):
        # Arrange
        arch = ModelArch.from_hf_config(QWEN36.config)
        no_shared = arch.model_copy(update={"n_shared_experts": 0})

        # Act
        with_shared, _ = lora_tensor_placement(arch, 16, "all-linear", 0, None, tp)
        without, _ = lora_tensor_placement(no_shared, 16, "all-linear", 0, None, tp)

        # Assert: gate/up B and down A halve under tp=2.
        assert with_shared - without == expected

    @pytest.mark.parametrize(
        ("tp", "expected"), [(1, 16 * 40 * 6144), (2, 16 * 40 * (6144 - 3072 // 2))]
    )
    def test_granite_shared_mlp_splits_input_b_and_output_a(self, tp, expected):
        # Arrange
        no_shared = GRANITE_H_TINY.model_copy(update={"n_shared_experts": 0})

        # Act
        with_shared, _ = lora_tensor_placement(
            GRANITE_H_TINY, 16, "all-linear", 0, None, tp
        )
        without, _ = lora_tensor_placement(no_shared, 16, "all-linear", 0, None, tp)

        # Assert
        assert with_shared - without == expected

    @pytest.mark.parametrize(
        ("tp", "expected"), [(1, 16 * 434_912), (2, 16 * (434_912 - 334_944 // 2))]
    )
    def test_qwen36_vision_tower_splits_qkv_b_and_fc_factors(self, tp, expected):
        # Arrange
        arch = ModelArch.from_hf_config(QWEN36.config)

        # Act
        tower, _ = lora_tensor_placement(
            arch, 16, "all-linear", 0, None, tp, QWEN_TOWER_TARGETS
        )
        text, _ = lora_tensor_placement(arch, 16, "all-linear", 0, None, tp)

        # Assert: qkv and fc1 B, fc2 A on 27 blocks and the merger halve.
        assert tower - text == expected

    def test_gemma4_towers_stay_whole_under_tensor_parallel(self):
        # Arrange
        arch = ModelArch.from_hf_config(GEMMA4_E4B.config)

        # Act
        tower, _ = lora_tensor_placement(
            arch, 16, "all-linear", 0, None, 2, GEMMA4_TOWER_TARGETS
        )
        text, _ = lora_tensor_placement(arch, 16, "all-linear", 0, None, 2)

        # Assert
        assert tower - text == 16 * 726_272

    @pytest.mark.parametrize(("tp", "per_rank"), [(1, 1_322_880), (2, 978_880)])
    def test_qwen36_splits_gated_q_and_gdn_projections(self, tp, per_rank):
        # Arrange
        arch = ModelArch.from_hf_config(QWEN36.config)

        # Act
        replicated, _ = lora_tensor_placement(arch, 1, "all-linear", 0, None, tp)

        # Assert: at tp=2 each rank keeps half of q/k/v B, o A, every GDN
        # in_proj B and out_proj A, and the shared expert's gate/up B and down A.
        assert replicated == per_rank

    def test_granite_micro_splits_fused_input_b_and_output_a(self):
        # Act
        replicated, _ = lora_tensor_placement(
            GRANITE_MICRO, 1, "all-linear", 0, None, 2
        )

        # Assert: per layer q 3840, k/v 2 x 2816, o 3840, input_linear
        # 2560 + 8192 and output_linear 4096 + 2560.
        assert replicated == 40 * (3840 + 5632 + 3840 + 10752 + 6656)
        assert replicated == 1_228_800

    def test_low_rank_attention_and_kda_place_their_own_linears(self):
        # Arrange
        arch = ModelArch.from_hf_config(GLM_TINY)

        # Act
        replicated, sharded = lora_tensor_placement(arch, 4, "all-linear", 0, 100)

        # Assert: factors under 25 features stay replicated. Per MLA layer
        # the kv_a output and kv_b input (16 each); per KDA layer the 16-wide
        # q/k/v/o and gate-up factors, the f_b and g_b inputs and beta's 2;
        # per MoE layer the shared expert's 16-wide gate, up and down factors.
        assert replicated == 4 * (2 * 32 + 2 * (16 * 6 + 8 * 2 + 2) + 3 * 3 * 16)
        assert replicated + sharded == lora_formulas.lora_param_count(arch, 4)

    @pytest.mark.parametrize(("tp", "per_rank"), [(1, 2900), (2, 2458)])
    def test_glm_splits_mla_up_projections_and_kda_heads(self, tp, per_rank):
        # Arrange
        arch = ModelArch.from_hf_config(GLM_TINY)

        # Act
        replicated, _ = lora_tensor_placement(arch, 1, "all-linear", 0, None, tp)

        # Assert: two MLA layers of q_a 96, q_b 32 + 32/tp, kv_a 80,
        # kv_b 16 + 64/tp and o 32/tp + 64; two KDA layers of q/k/v
        # 3 x (64 + 16/tp), f_b and g_b 2 x (8 + 16/tp), beta 64 + 2/tp and
        # o 16/tp + 64; the dense MLP and three shared experts split gate/up
        # B and down A.
        assert replicated == per_rank

    @pytest.mark.parametrize(("tp", "per_rank"), [(1, 1776), (2, 1584)])
    def test_deepseek_v4_splits_q_b_and_o_b_but_keeps_the_single_kv_head(
        self, tp, per_rank
    ):
        # Arrange
        arch = ModelArch.from_hf_config(DSV4_TINY)

        # Act
        replicated, _ = lora_tensor_placement(arch, 1, "all-linear", 0, None, tp)

        # Assert: three layers of q_a 96, q_b 32 + 64/tp, kv 80, o_b
        # 16/tp + 64 and a shared expert splitting gate/up B and down A
        # (16 each); the one KV head stays whole.
        assert replicated == per_rank


class TestSplitMoeLoraRecomputeBytes:
    def test_keeps_only_the_rank_wide_down_projection_per_matrix(self):
        # Arrange
        arch = ModelArch.from_hf_config(GRANITE_MOE)
        routed = 16384 * 8

        # Act
        split = split_moe_lora_recompute_bytes(arch, 1, 16384, 2, "contracted", 2, 16)

        # Assert: the delta runs at the activation dtype and adds into the
        # expert output in place, so each targeted matrix keeps x @ A only.
        assert split == routed * 16 * 2 * 2
        assert split == 8 * MiB

    def test_scales_with_microbatch_and_seq_on_contracted_path(self):
        one = split_moe_lora_recompute_bytes(MOE_TINY, 1, 1024, 2, "contracted", 2, 16)
        four = split_moe_lora_recompute_bytes(MOE_TINY, 4, 1024, 2, "contracted", 2, 16)
        longer = split_moe_lora_recompute_bytes(
            MOE_TINY, 1, 2048, 2, "contracted", 2, 16
        )

        assert one > 0
        assert four == 4 * one
        assert longer == 2 * one

    def test_zero_when_materialized_or_not_moe(self):
        assert (
            split_moe_lora_recompute_bytes(MOE_TINY, 1, 1024, 2, "materialized", 2, 16)
            == 0
        )
        assert (
            split_moe_lora_recompute_bytes(QWEN_05B, 1, 1024, 2, "contracted", 2, 16)
            == 0
        )
        assert (
            split_moe_lora_recompute_bytes(MOE_TINY, 1, 1024, 0, "contracted", 2, 16)
            == 0
        )


SMOL = ModelArch(
    n_layers=24,
    hidden_size=2048,
    intermediate_size=8192,
    n_heads=32,
    n_kv_heads=32,
    head_dim=64,
    vocab_size=49152,
    tied_embeddings=True,
)
# One attention and three Mamba-2 layers, each mixer followed by a dense MLP;
# Mamba-2 in_proj writes 32 inner + 48 conv + 4 head channels.
MAMBA_HYBRID_TINY = ModelArch(
    n_layers=4,
    hidden_size=16,
    intermediate_size=32,
    n_heads=2,
    n_kv_heads=1,
    head_dim=8,
    vocab_size=32,
    n_attention_layers=1,
    n_mamba_layers=3,
    mamba_n_heads=4,
    mamba_d_head=8,
    mamba_conv_dim=48,
)


class TestLoraLinears:
    def test_mamba2_layers_wrap_in_proj_only(self):
        linears = formulas.lora_linears(MAMBA_HYBRID_TINY)

        assert linears == {
            "attention": [(16, 16), (16, 8), (16, 8), (16, 16)],
            "ffn": [(16, 32), (16, 32), (32, 16)],
            "mamba": [(16, 32 + 48 + 4)],
        }

    def test_gated_delta_layers_wrap_four_in_projs_and_out_proj(self):
        arch = MAMBA_HYBRID_TINY.model_copy(update={"mamba_gated_delta": True})

        linears = formulas.lora_linears(arch)

        assert linears["mamba"] == [(16, 32)] * 4 + [(32, 16)]

    def test_attention_only_scope_wraps_no_mamba_linears(self):
        linears = formulas.lora_linears(MAMBA_HYBRID_TINY, "attention-only")

        assert linears["mamba"] == []
        assert linears["ffn"] == []

    def test_mla_and_kda_split_their_heads_over_tensor_parallel(self):
        arch = ModelArch.from_hf_config(GLM_TINY)

        linears = formulas.lora_linears(arch, tensor_parallel=2)

        # Two of four query heads; the latents and low-rank gates stay whole.
        assert linears["attention"] == [
            (64, 32),
            (32, 16),
            (64, 16),
            (16, 32),
            (16, 64),
        ]
        # f_a and g_a carry no adapter.
        assert linears["mamba"] == [(64, 8)] * 3 + [(8, 8), (64, 1), (8, 8), (8, 64)]


class TestLoraInputCastBytes:
    def test_checkpointed_block_peaks_at_the_up_proj_outputs(self):
        # Arrange: q, k, v, o, gate and up casts kept, plus up_proj's two
        # fp32 lora_B outputs: 4 * 2048 + 2 * 2048 + 2 * 8192 elements.
        tokens_bytes = 8 * 4096 * 4

        # Act
        cast = formulas.lora_input_cast_bytes(SMOL, 8, 4096)

        # Assert
        assert cast == tokens_bytes * 28672

    def test_uncheckpointed_layers_keep_every_wrapped_input(self):
        # Arrange: per layer the four attention inputs, gate, up and the
        # 8192-wide down_proj input.
        tokens_bytes = 8 * 4096 * 4

        # Act
        cast = formulas.lora_input_cast_bytes(
            SMOL, 8, 4096, gradient_checkpointing=False
        )

        # Assert
        assert cast == tokens_bytes * 20480 * 24

    def test_attention_only_peaks_at_the_o_proj_outputs(self):
        cast = formulas.lora_input_cast_bytes(SMOL, 8, 4096, "attention-only")

        assert cast == 8 * 4096 * 4 * (4 * 2048 + 2 * 2048)

    def test_scale_with_gradient_tokens(self):
        single = formulas.lora_input_cast_bytes(SMOL, 8, 4096)

        doubled = formulas.lora_input_cast_bytes(SMOL, 16, 4096)

        assert doubled == 2 * single

    def test_cover_dense_mlp_blocks_on_moe_hybrids(self):
        # Arrange: block-exclusive MoE with dense MLP blocks beside the experts.
        hybrid = MOE_TINY.model_copy(update={"n_mlp_layers": 2})
        dense = MOE_TINY.model_copy(
            update={
                "n_experts": None,
                "n_experts_per_tok": None,
                "expert_intermediate_size": None,
            }
        )
        tokens_bytes = 8 * 512 * 4

        # Act
        checkpointed = formulas.lora_input_cast_bytes(hybrid, 8, 512)
        stored = formulas.lora_input_cast_bytes(
            hybrid, 8, 512, gradient_checkpointing=False
        )

        # Assert: a checkpointed MLP block peaks at up_proj (gate and up
        # casts plus two 32-wide outputs); stored layers match the dense twin.
        assert hybrid.block_exclusive_layers
        assert hybrid.mlp_layers == dense.mlp_layers == 2
        assert checkpointed == tokens_bytes * (16 + 16 + 2 * 32)
        assert stored == formulas.lora_input_cast_bytes(
            dense, 8, 512, gradient_checkpointing=False
        )

    def test_one_mixer_per_layer_stack_peaks_in_the_mamba_and_mlp_sublayers(self):
        # Arrange: in_proj's cast and its two 84-wide fp32 outputs beat the
        # attention + MLP sublayers' peak of 160 elements.
        tokens_bytes = 2 * 64 * 4

        # Act
        cast = formulas.lora_input_cast_bytes(MAMBA_HYBRID_TINY, 2, 64)

        # Assert
        assert MAMBA_HYBRID_TINY.sublayer_transients
        assert not MAMBA_HYBRID_TINY.block_exclusive_layers
        assert cast == tokens_bytes * (16 + 2 * 84)

    def test_named_mixer_block_peaks_with_the_mlp_beside_it(self):
        # Arrange: the four attention casts, gate and up, plus up_proj's two
        # 32-wide outputs: 64 + 32 + 2 * 32 elements.
        tokens_bytes = 2 * 64 * 4

        # Act
        cast = formulas.lora_input_cast_bytes(
            MAMBA_HYBRID_TINY, 2, 64, layer="attention"
        )

        # Assert
        assert cast == tokens_bytes * 160

    def test_named_ffn_block_holds_only_the_mlp(self):
        # Arrange: gate and up casts plus up's two 32-wide outputs.
        tokens_bytes = 2 * 64 * 4

        # Act
        cast = formulas.lora_input_cast_bytes(MAMBA_HYBRID_TINY, 2, 64, layer="ffn")

        # Assert
        assert cast == tokens_bytes * (16 + 16 + 2 * 32)


class TestLoraFp32OutputBytes:
    def test_hold_two_outputs_of_the_widest_linear(self):
        output = formulas.lora_fp32_output_bytes(SMOL, 8, 4096)

        assert output == 8 * 4096 * 2 * 8192 * 4

    def test_attention_only_stops_at_the_attention_width(self):
        output = formulas.lora_fp32_output_bytes(SMOL, 8, 4096, "attention-only")

        assert output == 8 * 4096 * 2 * 2048 * 4


class TestLoraOnlyStoredInputBytes:
    def test_drop_the_down_proj_input(self):
        dropped = formulas.lora_only_stored_input_bytes(SMOL, 8, 4096, 2)

        assert dropped == 8 * 4096 * 8192 * 2

    def test_average_over_the_layers_that_hold_an_mlp(self):
        hybrid = MOE_TINY.model_copy(update={"n_mlp_layers": 1})

        dropped = formulas.lora_only_stored_input_bytes(hybrid, 8, 512, 2)

        assert hybrid.n_layers == 2
        assert dropped == 8 * 512 * 32 * 2 // 2

    def test_moe_without_a_dense_mlp_drops_nothing(self):
        dropped = formulas.lora_only_stored_input_bytes(MOE_TINY, 8, 512, 2)

        assert dropped == 0


class TestMeanBlockRecomputeBytes:
    def test_dense_stack_takes_its_one_block(self):
        seq_len = 1024
        arch = QWEN_05B

        mean = block_activations.mean_block_recompute_bytes(arch, 1, seq_len, 2.0)

        assert (
            mean
            == seq_len
            * (2 * arch.hidden_size + arch.peak_qkv_dim + 4 * arch.intermediate_size)
            * 2
        )

    def test_one_mixer_stack_weights_each_mixer_by_its_layers(self):
        # Arrange
        arch = MAMBA_HYBRID_TINY

        def peak(layer):
            return block_activations.block_recompute_bytes(
                arch, 2, 512, 2.0, backward=True, layer=layer
            )

        # Act
        mean = block_activations.mean_block_recompute_bytes(arch, 2, 512, 2.0)

        # Assert
        assert arch.sublayer_transients
        assert peak("attention") < peak("mamba")
        assert mean == (1 * peak("attention") + 3 * peak("mamba")) // 4
        assert mean < block_activations.block_recompute_bytes(
            arch, 2, 512, 2.0, backward=True
        )

    def test_block_exclusive_stack_takes_the_widest_block(self):
        arch = ModelArch.from_hf_config(NEMOTRON_H)

        mean = block_activations.mean_block_recompute_bytes(arch, 1, 4096, 2.0)

        assert arch.block_exclusive_layers
        assert mean == block_activations.block_recompute_bytes(
            arch, 1, 4096, 2.0, backward=True
        )


SUPER_VL = SUPPORTED_MODEL_INFO["nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"]
QWEN36 = SUPPORTED_MODEL_INFO["Qwen/Qwen3.6-35B-A3B"]
GEMMA4_E4B = SUPPORTED_MODEL_INFO["google/gemma-4-E4B-it"]
QWEN35_08B = SUPPORTED_MODEL_INFO["Qwen/Qwen3.5-0.8B"]
GEMMA4_E4B = SUPPORTED_MODEL_INFO["google/gemma-4-E4B-it"]
# Each family's full ``tower_connector_lora_targets``.
QWEN_TOWER_TARGETS = frozenset({"qkv", "linear_fc1", "linear_fc2"})
GEMMA4_TOWER_TARGETS = frozenset(
    {
        "linear",
        "input_proj",
        "embedding_projection",
        "input_proj_linear",
        "output_proj",
        "relative_k_proj",
    }
)
RADIO_TOWER_TARGETS = frozenset(
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
)


class TestLatentMoe:
    def test_routed_experts_run_at_the_latent_width(self):
        arch = ModelArch.from_hf_config(SUPER_VL.config)

        assert arch.moe_latent_size == 1024
        assert arch.expert_width == 1024

    def test_expert_params_add_the_latent_projections(self):
        # Arrange
        dense = ModelArch.from_hf_config(NEMOTRON_H_MOE)
        latent = ModelArch.from_hf_config({**NEMOTRON_H_MOE, "moe_latent_size": 1024})

        # Act
        experts, _ = formulas.moe_params_per_layer(latent)

        # Assert: 128 ungated experts at 1024 wide, the shared expert at
        # hidden width, and the hidden <-> latent projections.
        assert dense.expert_width == 2688
        assert experts == 128 * 2 * 1024 * 1856 + 2 * 2688 * 3712 + 2 * 2688 * 1024


class TestVisionFields:
    def test_super_vl_reads_its_dynamic_tile_bounds(self):
        arch = ModelArch.from_hf_config(SUPER_VL.config)

        assert arch.vision_hidden_size == 1280
        assert arch.vision_resize == "dynamic_tiles"
        assert (arch.vision_min_patches, arch.vision_max_patches) == (1024, 13312)
        assert arch.vision_spatial_merge_size == 2

    def test_gemma4_budget_is_soft_tokens_times_the_pool(self):
        arch = ModelArch.from_hf_config(GEMMA4_E4B.config)

        assert arch.vision_resize == "pooled_budget"
        assert arch.vision_spatial_merge_size == 3
        assert arch.vision_max_patches == 280 * 9

    def test_qwen_and_gemma_towers_keep_a_feature_cache(self):
        towers = [SUPER_VL, QWEN36, QWEN35_08B, GEMMA4_E4B]

        cached = [
            ModelArch.from_hf_config(t.config).vision_feature_cache for t in towers
        ]

        assert cached == [False, True, True, True]

    def test_vision_encoder_depth(self):
        depths = [
            ModelArch.from_hf_config(t.config).vision_n_layers
            for t in (QWEN35_08B, GEMMA4_E4B)
        ]

        assert depths == [12, 16]


class TestMultimodalTowerBlockParams:
    def test_qwen36_tower_blocks_leave_the_fsdp_root(self):
        # Arrange
        arch = ModelArch.from_hf_config(QWEN36.config)
        whole_tower = arch.model_copy(update={"multimodal_tower_block_params": 0})
        counts = formulas.param_counts(arch)
        kwargs = {
            "n_gpus": 4,
            "reshard_after_forward": True,
        }

        # Act
        gathered = fsdp_gather.fsdp_gathered_params(counts, arch, **kwargs)
        with_whole_tower = fsdp_gather.fsdp_gathered_params(
            counts, whole_tower, **kwargs
        )

        # Assert: 27 blocks of the 1152-wide, 4304-inner vision encoder.
        assert arch.multimodal_tower_block_params == 411_132_672
        assert arch.multimodal_tower_block_params < arch.multimodal_tower_params
        assert gathered == with_whole_tower - arch.multimodal_tower_block_params

    def test_gemma4_towers_count_gated_vision_and_conformer_audio(self):
        arch = ModelArch.from_hf_config(GEMMA4_E4B.config)

        # 16 gated-MLP vision blocks, a (2, 10240, 768) position table and 12
        # conformer audio blocks.
        assert arch.multimodal_tower_params == 151_019_520 + 15_728_640 + 302_014_464

    def test_tower_without_block_units_stays_at_the_root(self):
        arch = ModelArch.from_hf_config(SUPER_VL.config)

        assert arch.multimodal_tower_params > 0
        assert arch.multimodal_tower_block_params == 0


class TestRoutedChunkBytes:
    def test_chunked_expert_lora_grows_with_the_chunk_until_the_rows_cap_it(self):
        # Arrange: 16384 x 6 routed rows of the widest (2688) bf16 row.
        arch = ModelArch.from_hf_config(NEMOTRON_H_MOE)
        rows_bytes = 16384 * 6 * 2688 * 2

        def chunk_term(chunk_mib: int) -> int:
            return formulas.routed_chunk_bytes(
                arch,
                1,
                16384,
                2.0,
                chunk_mib * MiB,
                expert_lora=True,
                contracted=True,
                ep=1,
            )

        # Act
        terms = [chunk_term(mib) for mib in (64, 256, 1024)]

        # Assert
        fraction = formulas.ROUTED_CHUNK_LIVE_FRACTION
        assert terms == [
            int(64 * MiB * fraction),
            int(256 * MiB * fraction),
            int(rows_bytes * fraction),
        ]

    def test_expert_parallel_adds_an_fp32_combine_chunk(self):
        arch = ModelArch.from_hf_config(NEMOTRON_H_MOE)

        term = formulas.routed_chunk_bytes(
            arch, 1, 16384, 2.0, 64 * MiB, expert_lora=True, contracted=True, ep=8
        )

        assert term == int(2 * 64 * MiB * formulas.ROUTED_CHUNK_LIVE_FRACTION)

    def test_sorted_contracted_lora_holds_four_whole_chunks(self):
        arch = ModelArch.from_hf_config(GRANITE_MOE)

        term = formulas.routed_chunk_bytes(
            arch, 1, 16384, 2.0, 16 * MiB, expert_lora=True, contracted=True, ep=1
        )

        assert not arch.chunked_routed_experts
        assert term == 4 * 16 * MiB

    def test_recomputed_contracted_lora_holds_seven_whole_chunks(self):
        arch = ModelArch.from_hf_config(GRANITE_MOE)

        term = formulas.routed_chunk_bytes(
            arch,
            1,
            16384,
            2.0,
            16 * MiB,
            expert_lora=True,
            contracted=True,
            ep=1,
            recomputed=True,
        )

        assert term == 7 * 16 * MiB

    def test_chunk_budget_stops_the_bytes_growing_with_rows(self):
        arch = ModelArch.from_hf_config(GRANITE_MOE)
        chunk = 1024

        one = formulas.routed_chunk_bytes(
            arch, 1, 128, 2.0, chunk, expert_lora=True, contracted=True, ep=1
        )
        many = formulas.routed_chunk_bytes(
            arch, 16, 128, 2.0, chunk, expert_lora=True, contracted=True, ep=1
        )

        assert one == many == 4 * chunk

    def test_no_expert_lora_and_dense_models_are_zero(self):
        moe = ModelArch.from_hf_config(NEMOTRON_H_MOE)

        frozen = formulas.routed_chunk_bytes(
            moe, 1, 16384, 2.0, 64 * MiB, expert_lora=False, contracted=True, ep=1
        )
        dense = formulas.routed_chunk_bytes(
            QWEN_05B, 1, 16384, 2.0, 64 * MiB, expert_lora=True, contracted=True, ep=8
        )

        assert (frozen, dense) == (0, 0)


class TestPackedExpertLoraPlacement:
    def test_each_layer_matrix_shards_as_one_stacked_tensor(self):
        # Arrange: one expert's factor (4 x 16 = 64) sits under the threshold,
        # the stacked [E * r, 16] tensor (256) does not.
        threshold = 100

        # Act
        base_repl, base_shard = lora_tensor_placement(
            MOE_TINY, 4, "attention-only", 0, threshold
        )
        repl, shard = lora_tensor_placement(MOE_TINY, 4, "attention-only", 2, threshold)

        # Assert
        assert repl == base_repl
        assert shard - base_shard == lora_formulas.packed_lora_param_count(
            MOE_TINY, 4, 2
        )

    def test_gated_fused_gate_up_factor_is_twice_the_intermediate_wide(self):
        # Arrange: the stacked [E * r, 2 * inter] gate_up factor (1024) crosses
        # the threshold; the [E * r, inter] down factor (512) does not.
        threshold = 600

        # Act
        base_repl, base_shard = lora_tensor_placement(
            MOE_TINY, 4, "attention-only", 0, threshold
        )
        repl, shard = lora_tensor_placement(MOE_TINY, 4, "attention-only", 2, threshold)

        # Assert
        assert MOE_TINY.gated_mlp
        assert shard - base_shard == 2 * 16 * 2 * 32
        assert repl - base_repl == 2 * 16 * (16 + 16 + 32)


class TestPackedLoraParamCount:
    def test_gated_stack_counts_the_fused_gate_up_width(self):
        # 2 layers * 4 experts * rank 4 * (2 * (16 hidden + 32 inter) + 32)
        assert MOE_TINY.moe_layers == 2
        assert lora_formulas.packed_lora_param_count(MOE_TINY, 4, 2) == 4096

    def test_ungated_stack_counts_one_intermediate_width_per_matrix(self):
        ungated = MOE_TINY.model_copy(update={"gated_mlp": False})

        # 2 layers * 4 experts * rank 4 * 2 * (16 hidden + 32 inter)
        assert lora_formulas.packed_lora_param_count(ungated, 4, 2) == 3072

    def test_no_packed_targets_add_nothing(self):
        assert lora_formulas.packed_lora_param_count(MOE_TINY, 4, 0) == 0


class TestEngineExpertLoraParamCount:
    def test_3d_layout_packs_gate_and_up_into_one_slice(self):
        # 2 layers * 4 experts * rank 8 * (2 * 16 hidden + 3 * 32 inter)
        assert (
            lora_formulas.engine_expert_lora_param_count(MOE_TINY, 8, 1, True) == 8192
        )

    def test_2d_gated_layout_keeps_a_slice_per_projection(self):
        # 2 * 4 * 8 * (2 * 16 + 16 + 3 * 32)
        assert (
            lora_formulas.engine_expert_lora_param_count(MOE_TINY, 8, 1, False) == 9216
        )

    def test_tensor_parallel_shards_only_the_intermediate_dimension(self):
        # 2 * 4 * 8 * (2 * 16 + 3 * 16)
        assert (
            lora_formulas.engine_expert_lora_param_count(MOE_TINY, 8, 2, True) == 5120
        )

    def test_dense_model_has_no_expert_stacks(self):
        assert lora_formulas.engine_expert_lora_param_count(QWEN_05B, 8, 1, False) == 0


class TestReceivedExpertLoraBParamCount:
    def test_gated_experts_copy_gate_up_and_down_lora_b(self):
        # 2 layers * 4 experts * rank 8 * (2 * 32 inter + 16 hidden)
        assert lora_formulas.received_expert_lora_b_param_count(MOE_TINY, 8) == 5120

    def test_ungated_experts_copy_one_up_projection(self):
        ungated = MOE_TINY.model_copy(update={"gated_mlp": False})

        # 2 * 4 * 8 * (32 + 16)
        assert lora_formulas.received_expert_lora_b_param_count(ungated, 8) == 3072

    def test_dense_model_copies_nothing(self):
        assert lora_formulas.received_expert_lora_b_param_count(QWEN_05B, 8) == 0


class TestRoutedExpertParamsPerLayer:
    def test_dense_arch_has_no_routed_experts(self):
        assert formulas.routed_expert_params_per_layer(QWEN_05B) == 0


class TestTrainerNcclBytes:
    GROUP = 100 * MiB

    def test_plain_fsdp_on_one_node_holds_the_base_and_world_group(self):
        assert trainer_nccl_bytes(self.GROUP, 4, 4) == pytest.approx(
            1.14 * self.GROUP, abs=1
        )

    @pytest.mark.parametrize(
        ("n_gpus", "shard", "ep", "tp", "gpus_per_node", "fraction"),
        [
            (4, 2, 1, 1, None, 1.14 + 0.65 + 0.65),  # HSDP shard and replicate
            (4, 4, 1, 2, None, 1.14 + 0.65 + 0.65),  # TP and its replica ranks
            (4, 4, 4, 1, None, 1.14 + 1.0 + 1.1),  # EP spans the shard group
            (4, 4, 2, 1, None, 1.14 + 0.65 + 1.1 + 0.65),  # EP and its leftover ranks
            (4, 4, 2, 2, None, 1.14 + 0.65 + 1.1 + 0.65 + 0.65 + 0.65),  # EP x TP
            (4, 4, 4, 1, 2, 0.14 + 0.06 + 0.06 + 0.54),  # EP across two nodes
            (8, 8, 1, 1, 4, 0.14 + 0.06),  # world across two nodes
            (8, 4, 1, 1, 4, 0.14 + 0.06 + 1.0 + 0.06),  # a shard per node
            (16, 8, 8, 1, 8, 0.14 + 0.06 + 1.13 + 0.06 + 1.13 + 1.1),  # HSDP x EP
            # HSDP x EP with on-node leftover ranks and cross-node replicas.
            (16, 8, 4, 1, 8, 0.14 + 0.06 + 1.13 + 0.06 + 1.0 + 1.1 + 0.65 + 0.06),
        ],
    )
    def test_charges_each_group_by_size_and_node_span(
        self, n_gpus, shard, ep, tp, gpus_per_node, fraction
    ):
        nccl = trainer_nccl_bytes(self.GROUP, n_gpus, shard, ep, tp, gpus_per_node)

        assert nccl == pytest.approx(fraction * self.GROUP, abs=1)

    @pytest.mark.parametrize(
        ("cp", "tp", "fraction"),
        [
            (2, 1, 1.14 + 0.65 + 1.0),  # CP and the dp_shard_cp group
            (4, 1, 1.14 + 1.0 + 1.0),
            # TP and its replicas, CP, dp_shard_cp, and the CP mesh's TP and replicas.
            (2, 2, 1.14 + 0.65 + 0.65 + 0.65 + 0.65 + 0.65 + 0.65),
        ],
    )
    def test_context_parallel_adds_its_mesh_groups(self, cp, tp, fraction):
        nccl = trainer_nccl_bytes(self.GROUP, 4, 4, 1, tp, None, cp)

        assert nccl == pytest.approx(fraction * self.GROUP, abs=1)

    @pytest.mark.parametrize(
        ("group_size", "fraction"), [(2, 0.65), (3, 1.0), (4, 1.0), (8, 1.13)]
    )
    def test_on_node_fraction_grows_with_group_size(self, group_size, fraction):
        assert nccl_on_node_fraction(group_size) == fraction


class TestEpExchangeBytes:
    def test_holds_four_routed_row_buffers(self):
        rows, seq_len, act_bytes = 1, 1024, 2
        routed = rows * seq_len * MOE_TINY.n_experts_per_tok

        exchange = formulas.ep_exchange_bytes(MOE_TINY, rows, seq_len, act_bytes, 2)

        assert exchange == 4 * routed * MOE_TINY.expert_width * act_bytes

    def test_tensor_parallel_ranks_route_their_share_of_tokens(self):
        full = formulas.ep_exchange_bytes(MOE_TINY, 1, 1024, 2, 2)

        split = formulas.ep_exchange_bytes(MOE_TINY, 1, 1024, 2, 2, tensor_parallel=2)

        assert split * 2 == full

    def test_zero_without_expert_parallel_or_experts(self):
        assert formulas.ep_exchange_bytes(MOE_TINY, 1, 1024, 2, 1) == 0
        assert formulas.ep_exchange_bytes(QWEN_05B, 1, 1024, 2, 4) == 0


class TestCheckpointedBlocks:
    @pytest.mark.parametrize(
        ("every_n", "expected"), [(1, 24), (2, 12), (4, 6), (5, 5), (48, 1)]
    )
    def test_counts_the_first_block_and_every_nth_after_it(
        self, every_n: int, expected: int
    ) -> None:
        assert formulas.checkpointed_blocks(24, True, every_n) == expected

    def test_off_checkpoints_no_block(self) -> None:
        assert formulas.checkpointed_blocks(24, False, 1) == 0


class TestTrainingCheckpointedBlocks:
    def test_stride_runs_over_every_block(self) -> None:
        # Arrange
        arch = ModelArch.from_hf_config(NEMOTRON_H_MOE)
        settings = TrainingSettings(fsdp=FSDPConfig(checkpoint_every_n_blocks=2))

        # Act
        checkpointed = formulas.training_checkpointed_blocks(arch, settings)

        # Assert
        assert checkpointed == -(-arch.n_layers // 2)

    def test_unset_stride_with_checkpointing_raises(self) -> None:
        settings = TrainingSettings(fsdp=FSDPConfig())

        with pytest.raises(ValueError, match="checkpoint_every_n_blocks unset"):
            formulas.training_checkpointed_blocks(QWEN_05B, settings)


class TestCheckpointedMoeBlocks:
    @pytest.mark.parametrize(("every_n", "expected"), [(1, 3), (2, 1), (3, 2)])
    def test_block_exclusive_stack_counts_checkpointed_moe_layers(
        self, every_n: int, expected: int
    ) -> None:
        arch = ModelArch.from_hf_config(NEMOTRON_H_MOE)
        settings = TrainingSettings(fsdp=FSDPConfig(checkpoint_every_n_blocks=every_n))

        assert formulas.checkpointed_moe_blocks(arch, settings) == expected

    def test_every_block_moe_counts_every_checkpointed_block(self) -> None:
        arch = ModelArch.from_hf_config(GRANITE_MOE)
        settings = TrainingSettings(fsdp=FSDPConfig(checkpoint_every_n_blocks=2))

        assert formulas.checkpointed_moe_blocks(arch, settings) == 16

    def test_dense_model_and_checkpointing_off_count_none(self) -> None:
        arch = ModelArch.from_hf_config(NEMOTRON_H_MOE)
        settings = TrainingSettings(fsdp=FSDPConfig(checkpoint_every_n_blocks=1))
        off = settings.model_copy(update={"gradient_checkpointing": False})

        assert formulas.checkpointed_moe_blocks(QWEN_05B, settings) == 0
        assert formulas.checkpointed_moe_blocks(arch, off) == 0


class TestMergedLoraSyncBytes:
    def test_attention_scope_pushes_the_attention_projections(self):
        # Act
        total, largest = lora_formulas.merged_lora_sync_bytes(
            QWEN_05B, "attention-only", 0, 2
        )

        # Assert: q/o 896x896 and k/v 896x128 on 24 layers, without biases.
        assert total == 2 * 24 * (2 * 896 * 896 + 2 * 896 * 128)
        assert largest == 2 * 896 * 896

    def test_all_linear_scope_adds_the_mlp(self):
        # Arrange
        attention, _ = lora_formulas.merged_lora_sync_bytes(
            QWEN_05B, "attention-only", 0, 2
        )

        # Act
        total, largest = lora_formulas.merged_lora_sync_bytes(
            QWEN_05B, "all-linear", 0, 2
        )

        # Assert: gate, up and down 896x4864 on 24 layers.
        assert total == attention + 2 * 24 * 3 * 896 * 4864
        assert largest == 2 * 896 * 4864

    def test_packed_experts_push_whole_stacks(self):
        # Arrange
        unpacked, _ = lora_formulas.merged_lora_sync_bytes(MOE_TINY, "all-linear", 0, 2)

        # Act
        total, largest = lora_formulas.merged_lora_sync_bytes(
            MOE_TINY, "all-linear", 2, 2
        )

        # Assert: 4 experts of gate, up and down 16x32 on 2 layers; the
        # gate_up stack is the largest tensor.
        assert total == unpacked + 2 * 2 * 4 * 3 * 16 * 32
        assert largest == 2 * 4 * 2 * 32 * 16

    def test_one_packed_target_on_a_gated_mlp_pushes_gate_up(self):
        # Arrange
        unpacked, _ = lora_formulas.merged_lora_sync_bytes(MOE_TINY, "all-linear", 0, 2)

        # Act
        total, largest = lora_formulas.merged_lora_sync_bytes(
            MOE_TINY, "all-linear", 1, 2
        )

        # Assert: gate_up is 2 of the 3 expert matrices: 4 experts of 16x64
        # on 2 layers.
        assert total == unpacked + 2 * 2 * 4 * 2 * 16 * 32
        assert largest == 2 * 4 * 2 * 32 * 16

    def test_qwen35_output_gate_doubles_the_query_rows(self):
        # Arrange
        arch = ModelArch.from_hf_config(QWEN35_08B.config)

        # Act
        total, largest = lora_formulas.merged_lora_sync_bytes(
            arch, "attention-only", 0, 2
        )

        # Assert: 6 attention layers; gated q_proj 1024x4096, k/v 1024x512,
        # o_proj 2048x1024.
        assert total == 2 * 6 * (1024 * 4096 + 2 * 1024 * 512 + 2048 * 1024)
        assert largest == 2 * 1024 * 4096

    def test_gemma4_skips_shared_kv_and_pushes_per_layer_inputs(self):
        # Arrange
        arch = ModelArch.from_hf_config(GEMMA4_E4B.config)

        # Act
        total, largest = lora_formulas.merged_lora_sync_bytes(arch, "all-linear", 0, 2)

        # Assert: 35 sliding layers of head dim 256 and 7 full of 512; 18 of
        # the 42 layers share k/v, leaving 20 sliding and 4 full with k/v.
        # Gate, up and down 2560x10240 on 42 layers. Per-layer inputs are a
        # 2560x256 gate and 256x2560 projection per layer, plus the model's
        # 2560x10752 projection, the largest tensor.
        attention = (
            35 * (2560 * 2048 + 2048 * 2560)
            + 2 * 20 * 2560 * 512
            + 7 * (2560 * 4096 + 4096 * 2560)
            + 2 * 4 * 2560 * 1024
        )
        mlp = 42 * 3 * 2560 * 10240
        per_layer_inputs = 42 * 2 * 2560 * 256 + 2560 * 42 * 256
        assert total == 2 * (attention + mlp + per_layer_inputs)
        assert largest == 2 * 2560 * 42 * 256

    def test_nemotron_h_pushes_the_mamba_projections(self):
        # Arrange
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"].config
        )

        # Act
        total, largest = lora_formulas.merged_lora_sync_bytes(arch, "all-linear", 0, 2)

        # Assert: 4 attention layers (q/o 3136x5120, k/v 3136x1024), 17
        # ungated MLPs (up and down 3136x12544) and 21 Mamba-2 mixers whose
        # in_proj 3136x17504 (x 7680, conv 9728, dt 96) is the largest tensor.
        attention = 4 * 2 * 3136 * 5120 + 8 * 3136 * 1024
        mlp = 17 * 2 * 3136 * 12544
        mamba = 21 * (3136 * (7680 + 9728 + 96) + 7680 * 3136)
        assert total == 2 * (attention + mlp + mamba)
        assert largest == 2 * 3136 * 17504

    def test_qwen36_pushes_its_gated_delta_projections(self):
        # Arrange
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["Qwen/Qwen3.6-35B-A3B"].config
        )
        attention, _ = lora_formulas.merged_lora_sync_bytes(
            arch, "attention-only", 0, 2
        )

        # Act
        total, largest = lora_formulas.merged_lora_sync_bytes(arch, "all-linear", 2, 2)

        # Assert: 30 GDN layers project 2048 to qkv (8192), z (4096) and b, a
        # (32 each), and 4096 back; shared experts are 3 of 2048x512 on 40
        # layers. The 256-expert 2048x1024 gate_up stack is the largest.
        gdn = 30 * (2048 * (8192 + 4096 + 2 * 32) + 4096 * 2048)
        shared = 40 * 3 * 2048 * 512
        routed = 40 * 256 * 3 * 2048 * 512
        assert total == attention + 2 * (gdn + shared + routed)
        assert largest == 2 * 256 * 1024 * 2048
