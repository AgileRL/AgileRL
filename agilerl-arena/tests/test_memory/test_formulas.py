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


def test_param_counts_match_published_total():
    counts = formulas.param_counts(QWEN_05B)
    # Qwen2.5-0.5B has ~494M parameters.
    assert counts.total == pytest.approx(494_000_000, rel=0.01)
    assert counts.lm_head == 0  # tied embeddings


def test_param_counts_untied_adds_lm_head():
    untied = QWEN_05B.model_copy(update={"tied_embeddings": False})
    delta = formulas.param_counts(untied).total - formulas.param_counts(QWEN_05B).total
    assert delta == QWEN_05B.vocab_size * QWEN_05B.hidden_size


def test_from_hf_config_tiny_llm_asset():
    config = json.loads((ASSETS / "tiny_llm" / "config.json").read_text())
    arch = ModelArch.from_hf_config(config)
    assert arch.n_layers == config["num_hidden_layers"]
    assert arch.hidden_size == config["hidden_size"]
    assert arch.vocab_size == config["vocab_size"]
    assert formulas.param_counts(arch).total > 0


def test_lora_param_count_hand_computed():
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


def test_resolve_chunk_rows_targets_256mib_logit_tile():
    # 256 MiB / (151936 vocab * 4 bytes) = 441 rows.
    assert formulas.resolve_chunk_rows(151936) == 441
    assert formulas.resolve_chunk_rows(1000) == 4096  # clamped high
    assert formulas.resolve_chunk_rows(10_000_000) == 128  # clamped low
    assert formulas.resolve_chunk_rows(151936, explicit=64) == 64


def test_resolve_max_num_batched_tokens_caps_at_seqs_times_8192():
    assert formulas.resolve_max_num_batched_tokens(8, 1024) == 8 * 1024
    # Long context: capped at max(len, seqs * 8192).
    assert formulas.resolve_max_num_batched_tokens(8, 32768) == 8 * 8192
    assert formulas.resolve_max_num_batched_tokens(2, 32768) == 32768
    assert formulas.resolve_max_num_batched_tokens(8, 1024, explicit=4096) == 4096


def test_kv_cache_bytes_per_token_gqa():
    # 2 (K+V) * 24 layers * 2 kv-heads * 64 head-dim * 2 bytes.
    assert formulas.kv_cache_bytes_per_token(QWEN_05B, "bf16") == 12288


def test_kv_demand_sliding_window_caps_growth():
    windowed = QWEN_05B.model_copy(update={"sliding_window": 1024})
    full = formulas.kv_cache_demand_bytes(QWEN_05B, "bf16", 8, 8192)
    capped = formulas.kv_cache_demand_bytes(windowed, "bf16", 8, 8192)
    assert capped == full // 8  # window is 1/8 of the sequence


def test_weight_bytes_variants():
    counts = formulas.param_counts(QWEN_05B)
    dense = formulas.weight_bytes(counts, "bf16", WeightVariant())
    assert dense == int(counts.total * 2)

    towered = replace(counts, multimodal_towers=1_000_000)
    full = formulas.weight_bytes(towered, "bf16", WeightVariant())
    stripped = formulas.weight_bytes(
        towered, "bf16", WeightVariant(name="stripped", stripped_multimodal=True)
    )
    assert full - stripped == 1_000_000 * 2


def test_moe_resident_gather_is_training_only_and_scales_with_routing():
    # MoE gathered expert copies outlive the block that made them; dense
    # checkpointing frees them at the boundary.
    moe = QWEN_05B.model_copy(
        update={
            "n_experts": 32,
            "n_experts_per_tok": 8,
            "expert_intermediate_size": 512,
        }
    )
    assert formulas.moe_resident_gather_bytes(QWEN_05B, 4, 512, 2.0) == 0

    gathered = formulas.moe_resident_gather_bytes(moe, 4, 512, 2.0)
    assert gathered == (
        formulas.MOE_GATHER_RESIDENT_BLOCKS * 4 * 512 * 8 * moe.hidden_size * 2
    )
    # Linear in gradient tokens and in the routed expert count.
    assert formulas.moe_resident_gather_bytes(moe, 8, 512, 2.0) == 2 * gathered
    wider = moe.model_copy(update={"n_experts_per_tok": 16})
    assert formulas.moe_resident_gather_bytes(wider, 4, 512, 2.0) == 2 * gathered


def test_param_counts_without_checkpoint_total_attributes_nothing():
    assert formulas.param_counts(QWEN_05B).unattributed == 0


def test_param_counts_reconciles_to_the_checkpoint_total():
    analytic = formulas.param_counts(QWEN_05B)
    exact = formulas.param_counts(QWEN_05B, analytic.total + 1_000_000)
    assert exact.unattributed == 1_000_000
    assert exact.total == analytic.total + 1_000_000


def test_param_counts_reconciles_when_geometry_over_counts():
    analytic = formulas.param_counts(QWEN_05B)
    exact = formulas.param_counts(QWEN_05B, analytic.total - 5_000)
    assert exact.unattributed == -5_000
    assert exact.total == analytic.total - 5_000


def test_weight_bytes_carries_the_reconciliation_at_checkpoint_dtype():
    analytic = formulas.param_counts(QWEN_05B)
    exact = formulas.param_counts(QWEN_05B, analytic.total + 1_000_000)
    delta = formulas.weight_bytes(exact, "bf16", WeightVariant()) - (
        formulas.weight_bytes(analytic, "bf16", WeightVariant())
    )
    assert delta == 1_000_000 * 2


def test_hybrid_attention_layer_fraction_from_layer_types():
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
    # head_dim follows the config's 256; hidden // heads would give 192.
    assert arch.head_dim == 256
    assert arch.n_kv_heads == 1  # MQA

    # Only windowed layers cap KV growth, so long context costs less than a fully-global model of the same geometry.
    windowed = formulas.kv_cache_demand_bytes(arch, "bf16", 8, 8192)
    globl = formulas.kv_cache_demand_bytes(
        arch.model_copy(update={"sliding_window": None}), "bf16", 8, 8192
    )
    assert windowed < globl / 3


def test_sliding_window_pattern_integer_form():
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


def test_allocator_reserve_is_a_markup_on_allocated_bytes():
    assert formulas.allocator_reserve_bytes(0) == 0
    assert formulas.allocator_reserve_bytes(100 * MiB) == pytest.approx(7 * MiB)
    # Floor at zero.
    assert formulas.allocator_reserve_bytes(-1 * MiB) == 0


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


def test_hybrid_layer_mix_is_read_from_every_config_spelling():
    nemotron = ModelArch.from_hf_config(NEMOTRON_H)
    # 4 attention layers of 56: counting all as attention overstates KV 14x.
    assert (nemotron.attention_layers, nemotron.n_mamba_layers) == (4, 27)
    assert nemotron.is_hybrid_ssm

    falcon = ModelArch.from_hf_config(FALCON_H1)
    assert (falcon.attention_layers, falcon.n_mamba_layers) == (24, 24)

    dense = QWEN_05B
    assert not dense.is_hybrid_ssm
    assert dense.attention_layers == dense.n_layers


def test_kv_cache_counts_only_attention_layers():
    nemotron = ModelArch.from_hf_config(NEMOTRON_H)
    per_token = formulas.kv_cache_bytes_per_token(nemotron, "bf16")
    assert per_token == 2 * 4 * nemotron.n_kv_heads * nemotron.head_dim * 2


def test_mamba_state_charges_two_blocks_per_sequence():
    arch = ModelArch.from_hf_config(NEMOTRON_H)
    per_layer = formulas.mamba_page_bytes(arch)
    expected = arch.n_mamba_layers * per_layer * 16 * 2

    assert formulas.mamba_state_bytes(arch, 16) == expected > 0
    # Align mode caps residency at two blocks however long the context runs.
    assert formulas.mamba_state_bytes(QWEN_05B, 16) == 0


def test_aligned_block_size_matches_vllm():
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
    assert formulas.aligned_kv_block_size(QWEN_05B) == formulas.KV_BLOCK_SIZE_DEFAULT


@pytest.mark.parametrize(
    "config", [NEMOTRON_H, FALCON_H1], ids=["nemotron-h", "falcon-h1"]
)
def test_hybrid_mamba_params_are_counted(config):
    counts = formulas.param_counts(ModelArch.from_hf_config(config))
    assert counts.mamba > 0


def test_nemotron_h_mamba_param_share():
    counts = formulas.param_counts(ModelArch.from_hf_config(NEMOTRON_H))

    # ~4.0B Mamba params → ~7.4 GiB bf16.
    assert counts.mamba == pytest.approx(4.0e9, rel=0.05)
    assert counts.mamba * 2 == pytest.approx(7.4 * GiB, rel=0.05)


def test_lora_param_count_follows_hybrid_layer_mix():
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

    # Arrange
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


def test_kv_cache_uses_global_head_dim_on_full_attention_layers():
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
    # Arrange
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


def test_lora_input_casts_hold_only_the_widest_single_cast():
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


class TestSplitMoeLoraRecomputeBytes:
    def test_scales_with_microbatch_and_seq_on_contracted_path(self):
        one = formulas.split_moe_lora_recompute_bytes(
            MOE_TINY, 1, 1024, 2, "contracted", 2
        )
        four = formulas.split_moe_lora_recompute_bytes(
            MOE_TINY, 4, 1024, 2, "contracted", 2
        )
        longer = formulas.split_moe_lora_recompute_bytes(
            MOE_TINY, 1, 2048, 2, "contracted", 2
        )

        assert one > 0
        assert four == 4 * one
        assert longer == 2 * one

    def test_zero_when_materialized_or_not_moe(self):
        assert (
            formulas.split_moe_lora_recompute_bytes(
                MOE_TINY, 1, 1024, 2, "materialized", 2
            )
            == 0
        )
        assert (
            formulas.split_moe_lora_recompute_bytes(
                QWEN_05B, 1, 1024, 2, "contracted", 2
            )
            == 0
        )
        assert (
            formulas.split_moe_lora_recompute_bytes(
                MOE_TINY, 1, 1024, 0, "contracted", 2
            )
            == 0
        )


def test_lora_param_count_scales_with_global_head_dim():
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


def test_moe_dispatch_mask_ignores_shared_experts():
    # Arrange: shared experts join the gather but bypass the router.
    tokens, hidden, act = 4 * 512, MOE_TINY.hidden_size, 2.0
    base = MOE_TINY
    shared = MOE_TINY.model_copy(update={"n_shared_experts": 2})

    # Act
    forward_delta = formulas.moe_dispatch_bytes(
        shared, 4, 512, act
    ) - formulas.moe_dispatch_bytes(base, 4, 512, act)
    backward_delta = formulas.moe_dispatch_bytes(
        shared, 4, 512, act, backward=True
    ) - formulas.moe_dispatch_bytes(base, 4, 512, act, backward=True)

    # Assert: only the gather grows; the one-hot mask is unchanged.
    assert forward_delta == tokens * 2 * hidden * act
    assert backward_delta == 2 * forward_delta


def test_lora_casts_cover_dense_mlp_blocks_on_moe_hybrids():
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
