# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Formula-level tests: golden numbers on known geometries."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from agilerl.arena.memory import formulas
from agilerl.arena.memory.specs import GiB, MiB, ModelArch, WeightVariant

ASSETS = Path(__file__).parent / "assets"

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


class TestParamCounts:
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


class TestModelArchFromHfConfig:
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
        windowed = formulas.kv_cache_demand_bytes(arch, "bf16", 8, 8192)
        globl = formulas.kv_cache_demand_bytes(
            arch.model_copy(update={"sliding_window": None}), "bf16", 8, 8192
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
        assert formulas.lora_param_count(arch, rank) == expected
        # Attention-only scope drops the MLP terms.
        attn_only = rank * ((8 + 8) + (8 + 4) + (8 + 4) + (8 + 8))
        assert formulas.lora_param_count(arch, rank, "attention-only") == attn_only

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
        attn_only = formulas.lora_param_count(nemotron, rank, "attention-only")
        all_linear = formulas.lora_param_count(nemotron, rank)

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
        # Arrange: sliding-width attention term and the layer-averaged width.
        h, dh = arch.hidden_size, arch.head_dim
        q_dim, kv_dim = arch.n_heads * dh, arch.n_kv_heads * dh
        attn = (h + q_dim) + 2 * (h + kv_dim) + (q_dim + h)
        scaled = int(attn * arch.mean_qkv_dim / (q_dim + 2 * kv_dim))

        # Act
        rank = 4
        mixed = formulas.lora_param_count(arch, rank, "attention-only")
        sliding = formulas.lora_param_count(
            arch.model_copy(update={"global_head_dim": None}), rank, "attention-only"
        )

        # Assert: full-attention layers adapt the wider head.
        assert mixed == rank * scaled * arch.attention_layers
        assert mixed > sliding


class TestResolveChunkRows:
    def test_targets_256mib_logit_tile(self):
        # 256 MiB / (151936 vocab * 4 bytes) = 441 rows.
        assert formulas.resolve_chunk_rows(151936) == 441
        assert formulas.resolve_chunk_rows(1000) == 4096  # clamped high
        assert formulas.resolve_chunk_rows(10_000_000) == 128  # clamped low
        assert formulas.resolve_chunk_rows(151936, explicit=64) == 64


class TestResolveMaxNumBatchedTokens:
    def test_caps_at_seqs_times_8192(self):
        assert formulas.resolve_max_num_batched_tokens(8, 1024) == 8 * 1024
        # Long context: capped at max(len, seqs * 8192).
        assert formulas.resolve_max_num_batched_tokens(8, 32768) == 8 * 8192
        assert formulas.resolve_max_num_batched_tokens(2, 32768) == 32768
        assert formulas.resolve_max_num_batched_tokens(8, 1024, explicit=4096) == 4096


class TestKvCacheBytesPerToken:
    def test_gqa(self):
        # 2 (K+V) * 24 layers * 2 kv-heads * 64 head-dim * 2 bytes.
        assert formulas.kv_cache_bytes_per_token(QWEN_05B, "bf16") == 12288

    def test_counts_only_attention_layers(self):
        nemotron = ModelArch.from_hf_config(NEMOTRON_H)
        per_token = formulas.kv_cache_bytes_per_token(nemotron, "bf16")
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
        per_token = formulas.kv_cache_bytes_per_token(arch, "bf16")
        demand = formulas.kv_cache_demand_bytes(arch, "bf16", 8, 8192)

        # Assert: full-attention KV at the 512-wide head.
        assert per_token == pytest.approx(2 * storing * 1 * mean_dh * kv_bytes)
        assert demand == int(
            8
            * 2
            * storing
            * 1
            * kv_bytes
            * (windowed * min(8192, 512) * 256 + (1.0 - windowed) * 8192 * 512)
        )
        sliding_only = formulas.kv_cache_bytes_per_token(
            arch.model_copy(update={"global_head_dim": None}), "bf16"
        )
        assert per_token == pytest.approx(sliding_only * mean_dh / 256)

        # No sliding window: every storing layer uses the full-attention head width.
        full = arch.model_copy(update={"sliding_window": None})
        assert formulas.kv_cache_bytes_per_token(full, "bf16") == (
            2 * storing * 1 * 512 * 2
        )
        assert formulas.kv_cache_demand_bytes(full, "bf16", 8, 8192) == int(
            8 * 8192 * 2 * storing * 1 * 2 * 512
        )


class TestKvCacheDemandBytes:
    def test_sliding_window_caps_growth(self):
        windowed = QWEN_05B.model_copy(update={"sliding_window": 1024})
        full = formulas.kv_cache_demand_bytes(QWEN_05B, "bf16", 8, 8192)
        capped = formulas.kv_cache_demand_bytes(windowed, "bf16", 8, 8192)
        assert capped == full // 8  # window is 1/8 of the sequence


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


class TestRoutedExpertBytes:
    def test_matches_the_grouped_gemm_backward_workspace(self):
        arch = ModelArch.from_hf_config(GRANITE_MOE)

        backward = formulas.routed_expert_bytes(arch, 1, 16384, 2.0, backward=True)

        # gather + expert output (hidden each), gate/up, activation, its grad.
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
        engine = formulas.block_recompute_bytes(arch, 1, 1024, 2.0, engine_forward=True)
        trainer = formulas.block_recompute_bytes(arch, 1, 1024, 2.0)

        assert engine != trainer
        assert engine > 0


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

        backward = formulas.mamba_block_bytes(arch, 1, 16384, 2.0, backward=True)

        assert arch.mamba_chunk_size == 256
        assert backward == pytest.approx(3.63 * GiB, rel=0.03)

    def test_forward_is_smaller_and_attention_only_is_zero(self):
        arch = ModelArch.from_hf_config(NEMOTRON_NANO_MAMBA)

        forward = formulas.mamba_block_bytes(arch, 1, 16384, 2.0)
        backward = formulas.mamba_block_bytes(arch, 1, 16384, 2.0, backward=True)

        assert 0 < forward < backward
        assert formulas.mamba_block_bytes(QWEN_05B, 1, 16384, 2.0) == 0

    def test_eager_scan_adds_the_fp32_broadcast_product(self):
        arch = ModelArch.from_hf_config(NEMOTRON_NANO_MAMBA)

        fused = formulas.mamba_block_bytes(arch, 1, 16384, 2.0, backward=True)
        eager = formulas.mamba_block_bytes(
            arch, 1, 16384, 2.0, backward=True, eager_scan=True
        )

        # An L4 trainer asked for one 48 GiB tensor at 16k tokens.
        assert eager - fused == 16384 * 64 * 96 * 128 * 4
        assert eager - fused == 48 * GiB

    def test_block_exclusive_recompute_is_the_widest_layer(self):
        arch = ModelArch.from_hf_config(NEMOTRON_NANO_MAMBA)

        per_layer = {
            kind: formulas.block_recompute_bytes(
                arch, 1, 16384, 2.0, backward=True, layer=kind
            )
            for kind in formulas.block_kinds(arch)
        }
        widest = formulas.block_recompute_bytes(arch, 1, 16384, 2.0, backward=True)

        assert set(per_layer) == {"attention", "ffn", "mamba"}
        assert per_layer["mamba"] == formulas.mamba_block_bytes(
            arch, 1, 16384, 2.0, backward=True
        )
        assert widest == max(per_layer.values())

    def test_parallel_hybrid_block_adds_the_mamba_scan(self):
        falcon = ModelArch.from_hf_config(FALCON_H1)
        dense = falcon.model_copy(update={"n_mamba_layers": 0})

        with_scan = formulas.block_recompute_bytes(falcon, 1, 1024, 2.0)
        without = formulas.block_recompute_bytes(dense, 1, 1024, 2.0)

        assert with_scan - without == formulas.mamba_block_bytes(falcon, 1, 1024, 2.0)


class TestBlockRecomputeBackwardPeak:
    def test_gated_backward_keeps_two_residuals_and_four_mlp_tensors(self):
        # Arrange
        seq_len = 16384
        arch = QWEN_05B

        # Act
        backward = formulas.block_recompute_bytes(arch, 1, seq_len, 2.0, backward=True)

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

        forward = formulas.block_recompute_bytes(arch, 1, seq_len, 2.0)
        backward = formulas.block_recompute_bytes(arch, 1, seq_len, 2.0, backward=True)

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

        forward = formulas.block_recompute_bytes(arch, 1, seq_len, 2.0)

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

        backward = formulas.block_recompute_bytes(arch, 1, seq_len, 2.0, backward=True)
        base = formulas.block_recompute_bytes(without, 1, seq_len, 2.0, backward=True)

        assert arch.qk_norm
        assert not without.qk_norm
        assert backward - base == seq_len * arch.n_heads * arch.head_dim * 4

    def test_qwen2_has_no_query_key_norm(self):
        qwen2 = dict(QWEN3_4B)
        qwen2["model_type"] = "qwen2"

        assert not ModelArch.from_hf_config(qwen2).qk_norm


class TestBlockRecomputeNormInputs:
    def test_stock_rms_norm_backward_keeps_two_fp32_norm_inputs(self):
        # Arrange
        granite = ModelArch.from_hf_config(GRANITE_MOE)
        fused = granite.model_copy(update={"fp32_norm_inputs": False})

        # Act
        stock = formulas.block_recompute_bytes(granite, 1, 65536, 2.0, backward=True)
        liger = formulas.block_recompute_bytes(fused, 1, 65536, 2.0, backward=True)

        # Assert: a 64k-token Granite MoE snapshot held two 0.375 GiB fp32 copies.
        assert granite.fp32_norm_inputs
        assert stock - liger == 2 * 65536 * 1536 * 4
        assert stock - liger == 0.75 * GiB

    def test_liger_families_and_forward_pay_nothing(self):
        granite = ModelArch.from_hf_config(GRANITE_MOE)
        fused = granite.model_copy(update={"fp32_norm_inputs": False})

        assert not ModelArch.from_hf_config(QWEN3_4B).fp32_norm_inputs
        assert not ModelArch.from_hf_config(NEMOTRON_NANO_MAMBA).fp32_norm_inputs
        assert formulas.block_recompute_bytes(
            granite, 1, 1024, 2.0
        ) == formulas.block_recompute_bytes(fused, 1, 1024, 2.0)

    def test_block_exclusive_layer_keeps_one_norm_input(self):
        nano = ModelArch.from_hf_config(NEMOTRON_NANO_MAMBA)
        stock = nano.model_copy(update={"fp32_norm_inputs": True})

        added = {
            kind: formulas.block_recompute_bytes(
                stock, 1, 4096, 2.0, backward=True, layer=kind
            )
            - formulas.block_recompute_bytes(
                nano, 1, 4096, 2.0, backward=True, layer=kind
            )
            for kind in formulas.block_kinds(nano)
        }

        assert added == dict.fromkeys(("attention", "ffn", "mamba"), 4096 * 3136 * 4)


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

    def test_block_exclusive_stack_charges_its_widest_layer(self):
        # Nemotron-H: one mixer or MLP per layer; the MLP layer is widest.
        arch = ModelArch.from_hf_config(NEMOTRON_H)
        h, inter = NEMOTRON_H["hidden_size"], NEMOTRON_H["intermediate_size"]
        mlp_width = (2 * h if arch.gated_mlp else h) + inter

        dropout = formulas.lora_dropout_bytes(arch, 1, 1024, 4.0)

        assert arch.block_exclusive_layers
        assert dropout == 1024 * mlp_width * 5

    def test_without_checkpointing_every_layer_keeps_its_copies(self):
        arch = ModelArch.from_hf_config(QWEN3_4B)

        per_block = formulas.lora_dropout_bytes(arch, 1, 1024, 2.0)
        every_layer = formulas.lora_dropout_bytes(
            arch, 1, 1024, 2.0, gradient_checkpointing=False
        )

        assert every_layer == 36 * per_block

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


class TestAllocatorReserveBytes:
    def test_is_a_markup_on_allocated_bytes(self):
        assert formulas.allocator_reserve_bytes(0) == 0
        assert formulas.allocator_reserve_bytes(100 * MiB) == pytest.approx(7 * MiB)
        # Floor at zero.
        assert formulas.allocator_reserve_bytes(-1 * MiB) == 0

    def test_multi_rank_trainers_reserve_more_and_fsdp_most(self):
        one_rank = formulas.allocator_reserve_bytes(100 * MiB)
        data_parallel = formulas.allocator_reserve_bytes(100 * MiB, n_ranks=4)
        fsdp = formulas.allocator_reserve_bytes(100 * MiB, n_ranks=4, sharded=True)

        assert one_rank < data_parallel < fsdp
        assert data_parallel == pytest.approx(9 * MiB)
        assert fsdp == pytest.approx(14 * MiB)


class TestMambaStateBytes:
    def test_charges_two_blocks_per_sequence(self):
        arch = ModelArch.from_hf_config(NEMOTRON_H)
        per_layer = formulas.mamba_page_bytes(arch)
        expected = arch.n_mamba_layers * per_layer * 16 * 2

        assert formulas.mamba_state_bytes(arch, 16) == expected > 0
        assert formulas.mamba_page_bytes(QWEN_05B) == 0
        # Align mode caps residency at two blocks however long the context runs.
        assert formulas.mamba_state_bytes(QWEN_05B, 16) == 0


class TestAlignedKvBlockSize:
    def test_matches_vllm(self):
        """SSM state dtype changes the Mamba page and the aligned KV block size."""
        falcon = ModelArch.from_hf_config(FALCON_H1).model_copy(
            update={"mamba_ssm_state_dtype": "bf16"}
        )
        assert formulas.mamba_page_bytes(falcon) == 1_594_368
        assert formulas.aligned_kv_block_size(falcon) == 1568

        nemotron = ModelArch.from_hf_config(NEMOTRON_H)  # fp32 SSM state, the default
        assert formulas.mamba_page_bytes(nemotron) == 5_316_608
        assert formulas.aligned_kv_block_size(nemotron) == 1312

        # A dense model keeps vLLM's default.
        assert (
            formulas.aligned_kv_block_size(QWEN_05B) == formulas.KV_BLOCK_SIZE_DEFAULT
        )

    def test_falls_back_when_attention_layers_store_no_kv(self):
        hybrid = ModelArch.from_hf_config(FALCON_H1).model_copy(
            update={"n_kv_shared_layers": 24}
        )

        assert hybrid.is_hybrid_ssm
        assert formulas.kv_storing_layers(hybrid) == 0
        assert formulas.aligned_kv_block_size(hybrid) == formulas.KV_BLOCK_SIZE_DEFAULT


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


class TestFSDPGatheredParams:
    def test_reshard_keeps_a_prefetch_window_and_not_the_embedding(self):
        counts = formulas.param_counts(QWEN_05B)
        block = formulas.largest_block_params(counts, QWEN_05B)

        gathered = formulas.fsdp_gathered_params(
            counts,
            QWEN_05B,
            n_gpus=4,
            reshard_after_forward=True,
            prefetch_units=1,
            wrap_every_n_blocks=1,
            cpu_offload=False,
        )

        assert gathered == 2 * block
        assert gathered < counts.total - formulas.token_embedding_params(QWEN_05B)

    def test_no_reshard_gathers_every_sharded_parameter(self):
        counts = formulas.param_counts(QWEN_05B)
        sharded = counts.total - formulas.token_embedding_params(QWEN_05B)

        gathered = formulas.fsdp_gathered_params(
            counts,
            QWEN_05B,
            n_gpus=4,
            reshard_after_forward=False,
            prefetch_units=1,
            wrap_every_n_blocks=1,
            cpu_offload=False,
        )

        assert gathered == sharded

    def test_untied_head_stays_gathered(self):
        untied = QWEN_05B.model_copy(update={"tied_embeddings": False})
        counts = formulas.param_counts(untied)
        block = formulas.largest_block_params(counts, untied)

        gathered = formulas.fsdp_gathered_params(
            counts,
            untied,
            n_gpus=4,
            reshard_after_forward=True,
            prefetch_units=1,
            wrap_every_n_blocks=1,
            cpu_offload=False,
        )

        assert gathered == 2 * block + counts.lm_head

    def test_wrap_and_prefetch_set_the_window(self):
        counts = formulas.param_counts(QWEN_05B)
        block = formulas.largest_block_params(counts, QWEN_05B)

        gathered = formulas.fsdp_gathered_params(
            counts,
            QWEN_05B,
            n_gpus=4,
            reshard_after_forward=True,
            prefetch_units=1,
            wrap_every_n_blocks=2,
            cpu_offload=False,
        )

        assert gathered == 4 * block

    def test_one_gpu_has_no_extra_copy_unless_parameters_are_offloaded(self):
        counts = formulas.param_counts(QWEN_05B)
        block = formulas.largest_block_params(counts, QWEN_05B)
        kwargs = {
            "n_gpus": 1,
            "reshard_after_forward": True,
            "prefetch_units": 1,
            "wrap_every_n_blocks": 1,
        }

        assert (
            formulas.fsdp_gathered_params(counts, QWEN_05B, cpu_offload=False, **kwargs)
            == 0
        )
        assert (
            formulas.fsdp_gathered_params(counts, QWEN_05B, cpu_offload=True, **kwargs)
            == 2 * block
        )

    def test_block_exclusive_hybrid_window_uses_the_widest_layer_kind(self):
        arch = ModelArch.from_hf_config(NEMOTRON_H)
        counts = formulas.param_counts(arch)
        block = formulas.largest_block_params(counts, arch)
        mlp = formulas.mlp_params_per_layer(arch)

        gathered = formulas.fsdp_gathered_params(
            counts,
            arch,
            n_gpus=4,
            reshard_after_forward=True,
            prefetch_units=1,
            wrap_every_n_blocks=1,
            cpu_offload=False,
        )

        assert arch.block_exclusive_layers
        assert arch.mlp_layers
        assert not arch.tied_embeddings
        assert block == mlp
        assert block > formulas.mamba_params_per_layer(arch)
        assert block > formulas.attention_params_per_layer(arch)
        assert gathered == 2 * block + counts.lm_head
        window_gap_bf16 = 2 * (block - formulas.mamba_params_per_layer(arch)) * 2
        assert window_gap_bf16 == pytest.approx(241 * MiB, rel=0.01)


class TestLoraTensorPlacement:
    def test_flat_dp_replicates_every_tensor(self):
        replicated, sharded = formulas.lora_tensor_placement(
            QWEN_05B, 16, "all-linear", 0, None
        )

        assert sharded == 0
        assert replicated == formulas.lora_param_count(QWEN_05B, 16)

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
        replicated, sharded = formulas.lora_tensor_placement(
            arch, 16, "all-linear", 0, 100_000
        )

        assert sharded == 3 * 16 * 11008
        assert replicated + sharded == formulas.lora_param_count(arch, 16)
        attention_only, attention_sharded = formulas.lora_tensor_placement(
            arch, 16, "attention-only", 0, 100_000
        )
        assert attention_sharded == 0
        assert attention_only == formulas.lora_param_count(arch, 16, "attention-only")

    def test_threshold_zero_shards_every_tensor(self):
        replicated, sharded = formulas.lora_tensor_placement(
            QWEN_05B, 16, "all-linear", 0, 0
        )

        assert replicated == 0
        assert sharded == formulas.lora_param_count(QWEN_05B, 16)

    def test_zero_rank_is_empty(self):
        replicated, sharded = formulas.lora_tensor_placement(
            QWEN_05B, 0, "all-linear", 0, None
        )

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
        gated_repl, gated_shard = formulas.lora_tensor_placement(
            gated, rank, "all-linear", 0, None
        )
        ungated_repl, ungated_shard = formulas.lora_tensor_placement(
            ungated, rank, "all-linear", 0, None
        )

        assert gated_shard == ungated_shard == 0
        assert gated_repl - ungated_repl == rank * (64 + 128)


class TestSplitMoeLoraRecomputeBytes:
    def test_matches_the_fp32_and_bf16_adapter_snapshots(self):
        arch = ModelArch.from_hf_config(GRANITE_MOE)

        fp32 = formulas.split_moe_lora_recompute_bytes(
            arch, 1, 16384, 2, "contracted", 2, 4
        )
        bf16 = formulas.split_moe_lora_recompute_bytes(
            arch, 1, 16384, 2, "contracted", 2, 2
        )

        # fp32 adapters add a cast copy of the expert input and output.
        assert fp32 == 3.0 * GiB
        assert bf16 == 1.0 * GiB

    def test_scales_with_microbatch_and_seq_on_contracted_path(self):
        one = formulas.split_moe_lora_recompute_bytes(
            MOE_TINY, 1, 1024, 2, "contracted", 2, 2
        )
        four = formulas.split_moe_lora_recompute_bytes(
            MOE_TINY, 4, 1024, 2, "contracted", 2, 2
        )
        longer = formulas.split_moe_lora_recompute_bytes(
            MOE_TINY, 1, 2048, 2, "contracted", 2, 2
        )

        assert one > 0
        assert four == 4 * one
        assert longer == 2 * one

    def test_zero_when_materialized_or_not_moe(self):
        assert (
            formulas.split_moe_lora_recompute_bytes(
                MOE_TINY, 1, 1024, 2, "materialized", 2, 4
            )
            == 0
        )
        assert (
            formulas.split_moe_lora_recompute_bytes(
                QWEN_05B, 1, 1024, 2, "contracted", 2, 4
            )
            == 0
        )
        assert (
            formulas.split_moe_lora_recompute_bytes(
                MOE_TINY, 1, 1024, 0, "contracted", 2, 4
            )
            == 0
        )


class TestLoraInputCastBytes:
    def test_hold_only_the_widest_single_cast(self):
        # Checkpointed LoRA: one fp32 cast of the widest input (down_proj).
        # 8 rows x 4096 tokens x 8192 intermediate x fp32 = 1024 MiB.
        smol = ModelArch(
            n_layers=24,
            hidden_size=2048,
            intermediate_size=8192,
            n_heads=32,
            n_kv_heads=32,
            head_dim=64,
            vocab_size=49152,
            tied_embeddings=True,
        )
        assert formulas.lora_input_cast_bytes(smol, 8, 4096) == 1024 * MiB

        # The intermediate is the widest input, so the term tracks it alone.
        narrower = smol.model_copy(update={"intermediate_size": 4096})
        assert formulas.lora_input_cast_bytes(narrower, 8, 4096) == 512 * MiB

        # Attention-only skips down_proj; the residual stream or attention output binds.
        attn_only = formulas.lora_input_cast_bytes(smol, 8, 4096, "attention-only")
        assert attn_only == 8 * 4096 * 2048 * 4

        # Linear in gradient tokens.
        assert formulas.lora_input_cast_bytes(smol, 16, 4096) == 2048 * MiB

        # Without checkpointing, every layer's casts are saved through backward.
        assert formulas.lora_input_cast_bytes(
            smol, 8, 4096, gradient_checkpointing=False
        ) > 24 * formulas.lora_input_cast_bytes(smol, 8, 4096)

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
        assert hybrid.mlp_layers == dense.mlp_layers == 2

        # Act / Assert: the down-projection cast matches the dense twin.
        assert formulas.lora_input_cast_bytes(hybrid, 8, 512) == (
            formulas.lora_input_cast_bytes(dense, 8, 512)
        )
        assert formulas.lora_input_cast_bytes(
            hybrid, 8, 512, gradient_checkpointing=False
        ) == formulas.lora_input_cast_bytes(dense, 8, 512, gradient_checkpointing=False)
