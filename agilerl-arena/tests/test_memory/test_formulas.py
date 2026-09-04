# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Formula-level tests: golden numbers on known geometries."""

import json
from pathlib import Path

import pytest

from agilerl.arena.memory import formulas
from agilerl.arena.memory.specs import MiB, ModelArch, WeightVariant

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
    assert counts.quantizable == counts.attention + counts.mlp


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


def test_resolve_chunk_rows_mirrors_framework_heuristic():
    # 256 MiB / (151936 vocab * 4 bytes) = 441 rows.
    assert formulas.resolve_chunk_rows(151936) == 441
    assert formulas.resolve_chunk_rows(1000) == 4096  # clamped high
    assert formulas.resolve_chunk_rows(10_000_000) == 128  # clamped low
    assert formulas.resolve_chunk_rows(151936, explicit=64) == 64


def test_resolve_max_num_batched_tokens_mirrors_framework_rule():
    assert formulas.resolve_max_num_batched_tokens(8, 1024) == 8 * 1024
    # Long context: capped at max(len, seqs * 8192).
    assert formulas.resolve_max_num_batched_tokens(8, 32768) == 8 * 8192
    assert formulas.resolve_max_num_batched_tokens(2, 32768) == 32768
    assert formulas.resolve_max_num_batched_tokens(8, 1024, explicit=4096) == 4096


def test_kv_cache_bytes_per_token_gqa():
    # 2 (K+V) * 24 layers * 2 kv-heads * 64 head-dim * 2 bytes.
    assert formulas.kv_cache_bytes_per_token(QWEN_05B, "auto", "bf16") == 12288
    assert formulas.kv_cache_bytes_per_token(QWEN_05B, "fp8", "bf16") == 6144


def test_kv_demand_sliding_window_caps_growth():
    windowed = QWEN_05B.model_copy(update={"sliding_window": 1024})
    full = formulas.kv_cache_demand_bytes(QWEN_05B, "auto", "bf16", 8, 8192)
    capped = formulas.kv_cache_demand_bytes(windowed, "auto", "bf16", 8, 8192)
    assert capped == full // 8  # window is 1/8 of the sequence


def test_weight_bytes_variants():
    counts = formulas.param_counts(QWEN_05B)
    dense = formulas.weight_bytes(counts, "bf16", WeightVariant())
    assert dense == int(counts.total * 2)

    nf4 = formulas.weight_bytes(
        counts,
        "bf16",
        WeightVariant(name="nf4", quantization="nf4"),
        kbit_prepared=True,
    )
    assert nf4 < dense
    # Embeddings stay bf16 and norms go fp32, so the saving is well short of 4x.
    assert nf4 > dense // 4

    realised = formulas.weight_bytes(
        counts,
        "bf16",
        WeightVariant(name="nf4", quantization="nf4", realised_bytes=123),
    )
    assert realised == 123


def test_moe_resident_gather_is_training_only_and_scales_with_routing():
    # Three MoEs (OLMoE 64 experts, granite-3.1 40 and 32) all under-predicted
    # training by 8-14% of peak, always low, in proportion to the expert
    # gather: the gathered copies outlive the block that made them, which the
    # dense checkpointing path assumes they do not.
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


def test_bnb_workspace_scales_with_gradient_tokens_and_geometry():
    # Paired bf16/nf4 runs at four context lengths on four models put the
    # per-token cost at 68, 104, 160 and 194 KiB/token (Qwen2.5-0.5B/1.5B/3B,
    # Gemma 4 E4B) -- the term that makes the nf4 trade invert with context.
    assert formulas.bnb_backward_workspace_bytes(QWEN_05B, 0, 4096) == 0

    one = formulas.bnb_backward_workspace_bytes(QWEN_05B, 1, 1024)
    assert formulas.bnb_backward_workspace_bytes(QWEN_05B, 2, 1024) == 2 * one
    assert formulas.bnb_backward_workspace_bytes(QWEN_05B, 1, 2048) == 2 * one

    # Deeper and wider both cost more, and Qwen2.5-0.5B's own rate lands near
    # the 68 KiB/token measured for it.
    deeper = QWEN_05B.model_copy(update={"n_layers": 48})
    assert formulas.bnb_backward_workspace_bytes(deeper, 1, 1024) > one
    per_token = one / 1024 / 1024
    assert 55 < per_token < 85, f"{per_token:.1f} KiB/token"


def test_kbit_prep_upcasts_a_tied_embedding_to_fp32():
    # prepare_model_for_kbit_training upcasts lm_head to fp32, and on a tied
    # model lm_head *is* the embedding table -- which counts.lm_head reports as
    # 0 precisely because it is tied. Measured (nf4 - bf16) at 512 tokens:
    # Qwen2.5-1.5B -1384 MiB against -1391 predicted with this, -1837 without;
    # gemma-4-E4B +894 against +1126 with, -5530 without, i.e. the wrong sign.
    tied = QWEN_05B  # tied_embeddings=True
    counts = formulas.param_counts(tied)
    nf4 = WeightVariant(name="nf4", quantization="nf4")

    without = formulas.weight_bytes(counts, "bf16", nf4, kbit_prepared=True)
    with_upcast = formulas.weight_bytes(
        counts, "bf16", nf4, kbit_prepared=True, tied_embeddings=True
    )
    # bf16 -> fp32 on the embedding block: two extra bytes per parameter.
    assert with_upcast - without == counts.embedding * 2

    # Untied models already charge lm_head separately, so nothing changes.
    untied_counts = formulas.param_counts(
        QWEN_05B.model_copy(update={"tied_embeddings": False})
    )
    assert formulas.weight_bytes(
        untied_counts, "bf16", nf4, kbit_prepared=True, tied_embeddings=False
    ) == formulas.weight_bytes(untied_counts, "bf16", nf4, kbit_prepared=True)

    # And it is inert without k-bit preparation.
    plain = WeightVariant()
    assert formulas.weight_bytes(
        counts, "bf16", plain, tied_embeddings=True
    ) == formulas.weight_bytes(counts, "bf16", plain)


def test_engine_overhead_tracks_depth_and_slots():
    # Three round constants: 512 MiB + 8 MiB per layer + 8 MiB per engine
    # sequence slot. Checked on leave-one-model-out over 182 stored points --
    # 119 MiB held-out error for a single constant, 71 for these, and 72 for a
    # least-squares fit that also carried a per-device term. Round numbers with
    # no lookup beat the fit, so the lookup is gone.
    shallow = QWEN_05B  # 24 layers
    deep = QWEN_05B.model_copy(update={"n_layers": 36})

    base = formulas.engine_process_overhead_bytes(shallow, 8)
    assert (
        formulas.engine_process_overhead_bytes(deep, 8) - base
        == 12 * formulas.ENGINE_OVERHEAD_PER_LAYER_BYTES
    )
    assert (
        formulas.engine_process_overhead_bytes(shallow, 16) - base
        == 8 * formulas.ENGINE_OVERHEAD_PER_SEQ_BYTES
    )
    # Within the form's held-out accuracy of the measured medians -- 741 MiB on
    # the A100 and 687 on the L4 for Qwen2.5-0.5B, over sweeps centred on 8
    # slots. Device-independent by construction now.
    assert abs(base - 741 * MiB) < 100 * MiB
    assert abs(base - 687 * MiB) < 100 * MiB


def test_param_counts_without_checkpoint_total_attributes_nothing():
    assert formulas.param_counts(QWEN_05B).unattributed == 0


def test_param_counts_reconciles_to_the_checkpoint_total():
    analytic = formulas.param_counts(QWEN_05B)
    exact = formulas.param_counts(QWEN_05B, analytic.total + 1_000_000)
    assert exact.unattributed == 1_000_000
    assert exact.total == analytic.total + 1_000_000
    # A count of nothing in particular: bnb and PEFT cannot reach it.
    assert exact.quantizable == analytic.quantizable


def test_param_counts_reconciles_when_geometry_over_counts():
    analytic = formulas.param_counts(QWEN_05B)
    exact = formulas.param_counts(QWEN_05B, analytic.total - 5_000)
    assert exact.unattributed == -5_000
    assert exact.total == analytic.total - 5_000


@pytest.mark.parametrize("quantization", ["none", "nf4"])
def test_weight_bytes_carries_the_reconciliation_at_checkpoint_dtype(quantization):
    analytic = formulas.param_counts(QWEN_05B)
    exact = formulas.param_counts(QWEN_05B, analytic.total + 1_000_000)
    variant = WeightVariant(name=quantization, quantization=quantization)
    delta = formulas.weight_bytes(exact, "bf16", variant) - formulas.weight_bytes(
        analytic, "bf16", variant
    )
    assert delta == 1_000_000 * 2


def test_block_recompute_sdpa_path_adds_score_matrix():
    with_flash = formulas.block_recompute_bytes(
        QWEN_05B, 2, 512, 2.0, flash_attention=True
    )
    without = formulas.block_recompute_bytes(
        QWEN_05B, 2, 512, 2.0, flash_attention=False
    )
    # Three copies: scores, saved softmax output, and its gradient.
    assert without - with_flash == 3 * 2 * QWEN_05B.n_heads * 512 * 512 * 2


def test_hybrid_attention_layer_fraction_from_layer_types():
    # Gemma-4-style hybrid attention: most layers windowed, a few global.
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
    # head_dim is taken from the config, not hidden // heads (192 here).
    assert arch.head_dim == 256
    assert arch.n_kv_heads == 1  # MQA

    # Only the windowed layers cap KV growth, so long context costs far less
    # than a fully-global model of the same geometry.
    windowed = formulas.kv_cache_demand_bytes(arch, "auto", "bf16", 8, 8192)
    globl = formulas.kv_cache_demand_bytes(
        arch.model_copy(update={"sliding_window": None}), "auto", "bf16", 8, 8192
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


def test_only_eager_materializes_attention_scores():
    assert formulas.resolve_attn_implementation("auto", False) == "sdpa"
    assert formulas.resolve_attn_implementation("auto", True) == "flash_attention_2"
    assert (
        formulas.resolve_attn_implementation("flex_attention", True) == "flex_attention"
    )

    assert formulas.materializes_attention_scores("eager")
    for impl in ("sdpa", "flash_attention_2", "flex_attention"):
        assert not formulas.materializes_attention_scores(impl)


def test_allocator_reserve_is_a_markup_on_allocated_bytes():
    # Every other term is an allocation size; the device is charged the
    # caching allocator's *reservation*, which rounds to segment boundaries.
    # 7% is the median measured reserved/allocated ratio over the corpus's
    # training points, not a value chosen to fit.
    assert formulas.allocator_reserve_bytes(0) == 0
    assert formulas.allocator_reserve_bytes(100 * MiB) == pytest.approx(7 * MiB)
    # Never negative, whatever a caller passes.
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
#: Falcon-H1 runs attention and the SSM in parallel in *every* block, and
#: declares no layout at all -- so both counts are n_layers.
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
    # 4 attention layers out of 56: counting them all as attention overstates
    # the KV cache 14x.
    assert (nemotron.attention_layers, nemotron.n_mamba_layers) == (4, 27)
    assert nemotron.is_hybrid_ssm

    falcon = ModelArch.from_hf_config(FALCON_H1)
    assert (falcon.attention_layers, falcon.n_mamba_layers) == (24, 24)

    dense = QWEN_05B
    assert not dense.is_hybrid_ssm
    assert dense.attention_layers == dense.n_layers


def test_kv_cache_counts_only_attention_layers():
    nemotron = ModelArch.from_hf_config(NEMOTRON_H)
    per_token = formulas.kv_cache_bytes_per_token(nemotron, "auto", "bf16")
    assert per_token == 2 * 4 * nemotron.n_kv_heads * nemotron.head_dim * 2


def test_mamba_state_is_constant_in_context_length():
    arch = ModelArch.from_hf_config(NEMOTRON_H)
    short = formulas.mamba_state_bytes(arch, 16, "none", 512)
    long = formulas.mamba_state_bytes(arch, 16, "none", 131072)
    assert short == long > 0
    # 'align' bounds residency at two blocks; 'none' at one.
    assert formulas.mamba_state_bytes(arch, 16, "align", 4096) == 2 * short
    assert formulas.mamba_state_bytes(QWEN_05B, 16, "none", 4096) == 0


def test_aligned_block_size_matches_vllm():
    """Both halves against vLLM 0.26's own numbers, measured on an A100.

    vLLM logs the block size it picks, and its
    ``get_mamba_state_dtype_from_config`` gives the page it picked it from.
    The SSM half is the trap: it is resolved *per model*, and Nemotron-H
    keeps it in fp32 where Falcon-H1 keeps it in the model dtype -- which
    doubles the page and the block size with it.
    """
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


def test_lora_input_casts_hold_only_the_widest_single_cast():
    # PEFT's fp32 input casts do not accumulate across a checkpointed block's
    # wrapped linears: two allocator timelines put the live bytes at exactly
    # one cast of the widest input, both times the MLP down-projection's.
    #
    #   SmolLM2-1.7B  8 x 4096 rows-tokens, inter 8192  -> 1024 MiB observed
    #   Gemma 4 E2B   4 x 4096,     peak inter 12288    ->  768 MiB observed
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

    # Attention-only targeting never wraps down_proj, so the residual stream
    # and the attention output are all that can bind.
    attn_only = formulas.lora_input_cast_bytes(smol, 8, 4096, "attention-only")
    assert attn_only == 8 * 4096 * 2048 * 4

    # Linear in gradient tokens.
    assert formulas.lora_input_cast_bytes(smol, 16, 4096) == 2048 * MiB

    # Without checkpointing nothing is recomputed, so every layer's casts are
    # saved from forward to backward and they do accumulate.
    assert formulas.lora_input_cast_bytes(
        smol, 8, 4096, gradient_checkpointing=False
    ) > 24 * formulas.lora_input_cast_bytes(smol, 8, 4096)
