"""Formula-level tests: golden numbers on known geometries."""

import json
from pathlib import Path

import pytest

from agilerl.memory import formulas
from agilerl.memory.specs import ModelArch, WeightVariant

ASSETS = Path(__file__).parent.parent / "assets"

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


def test_block_recompute_sdpa_path_adds_score_matrix():
    with_flash = formulas.block_recompute_bytes(
        QWEN_05B, 2, 512, 2.0, flash_attention=True
    )
    without = formulas.block_recompute_bytes(
        QWEN_05B, 2, 512, 2.0, flash_attention=False
    )
    assert without - with_flash == 2 * QWEN_05B.n_heads * 512 * 512 * 2
