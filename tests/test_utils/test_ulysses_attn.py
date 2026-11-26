# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for Ulysses attention helpers."""

from __future__ import annotations

import functools
import importlib.machinery
import sys
import types
from typing import Any

import pytest
import torch
import torch.distributed as dist
import transformers.modeling_flash_attention_utils as flash_utils
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from transformers import PretrainedConfig, Qwen3_5Config, Qwen3Config, Qwen3ForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssRotaryEmbedding,
)

from agilerl.distributed.context_parallel import (
    ULYSSES_PARAMS,
    PublishedUlyssesParams,
    attention_head_counts,
    clear_ulysses_params,
    model_attention_backend,
    replicate_kv_heads,
    substitute_hf_ulysses_attn,
    ulysses,
)
from tests.parallel_helpers import (
    assert_exact,
    cpu_flex_attention,
    publish_varlen_offsets,
)
from tests.rank_launch import run_ranks

HIDDEN = 16
HEADS = 8
HEAD_DIM = 4
BATCH = 2
SEQ = 8
WINDOW = 3


def _flex_attention_layers(kv_heads: int) -> nn.ModuleList:
    config = GptOssConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=HEADS,
        num_key_value_heads=kv_heads,
        head_dim=HEAD_DIM,
        sliding_window=WINDOW,
        layer_types=["sliding_attention", "full_attention"],
        attention_bias=True,
    )
    config._attn_implementation = "flex_attention"
    layers = nn.ModuleList([GptOssAttention(config, 0), GptOssAttention(config, 1)])
    with torch.no_grad():
        for layer in layers:
            layer.sinks.normal_()
    return layers


def _reference_mask(window: int | None) -> BlockMask:
    def mask(
        _b: torch.Tensor, _h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        allowed = kv_idx <= q_idx
        if window is not None:
            allowed &= kv_idx > q_idx - window
        return allowed

    return create_block_mask(mask, B=None, H=None, Q_LEN=SEQ, KV_LEN=SEQ, device="cpu")


def _flex_ulysses_worker(
    rank: int, world_size: int, kv_heads: int, take_first_sinks: bool
) -> None:
    """Ulysses flex over sequence shards matches unsharded flex, forward and grads."""
    # Arrange
    torch.manual_seed(0)
    ALL_ATTENTION_FUNCTIONS["flex_attention"] = cpu_flex_attention
    if take_first_sinks:
        ulysses.cp_local_heads = lambda per_head, _group: per_head.reshape(
            world_size, -1
        )[0]
    layers = _flex_attention_layers(kv_heads)
    rotary = GptOssRotaryEmbedding(layers[0].config)
    hidden = torch.randn(BATCH, SEQ, HIDDEN)
    probe = torch.randn(BATCH, SEQ, HIDDEN)
    cos, sin = rotary(hidden, torch.arange(SEQ).expand(BATCH, SEQ))
    masks = (_reference_mask(WINDOW), _reference_mask(None))
    expected = []
    for layer, mask in zip(layers, masks, strict=True):
        output = layer(hidden, (cos, sin), mask)[0]
        (output * probe).sum().backward()
        expected.append(output.detach())
    expected_grads = [
        {name: param.grad.clone() for name, param in layer.named_parameters()}
        for layer in layers
    ]
    layers.zero_grad()
    group = dist.new_group(list(range(world_size)))
    substitute_hf_ulysses_attn(group, "flex_attention")
    local = slice(rank * SEQ // world_size, (rank + 1) * SEQ // world_size)
    publish_varlen_offsets(torch.arange(BATCH + 1, dtype=torch.int32) * SEQ)

    # Act
    outputs = []
    for layer in layers:
        output = layer(hidden[:, local], (cos[:, local], sin[:, local]), None)[0]
        (output * probe[:, local]).sum().backward()
        outputs.append(output.detach())
    clear_ulysses_params()

    # Assert
    for index, (output, full) in enumerate(zip(outputs, expected, strict=True)):
        assert_exact(output, full[:, local], f"layer {index} output")
    for layer, grads in zip(layers, expected_grads, strict=True):
        for name, param in layer.named_parameters():
            grad = param.grad.clone()
            dist.all_reduce(grad, group=group)
            assert_exact(grad, grads[name], f"{name} grad")


class TestUlyssesFlexAttention:
    @pytest.mark.multi_rank
    @pytest.mark.parametrize("kv_heads", [4, 1], ids=["split_kv", "replicated_kv"])
    def test_sliding_and_full_layers_match_unsharded_flex(self, kv_heads: int) -> None:
        run_ranks(_flex_ulysses_worker, 2, kv_heads, False, timeout=300)

    @pytest.mark.multi_rank
    def test_parity_fails_when_every_rank_takes_the_first_sinks(self) -> None:
        with pytest.raises(AssertionError, match="output"):
            run_ranks(_flex_ulysses_worker, 2, 4, True, timeout=300)


class TestFlashWindowSize:
    def test_none_or_short_sequence_is_full_causal(self) -> None:
        assert ulysses.flash_window_size(None, 16) == (-1, -1)
        assert ulysses.flash_window_size(8, 8) == (-1, -1)
        assert ulysses.flash_window_size(8, 4) == (-1, -1)

    def test_gathered_length_uses_inclusive_flash_bounds(self) -> None:
        assert ulysses.flash_window_size(4, 16) == (3, 3)
        assert ulysses.flash_window_size(WINDOW, SEQ) == (WINDOW - 1, WINDOW - 1)


class TestFullSequenceBlockMask:
    @pytest.mark.parametrize("window", [WINDOW, None], ids=["sliding", "full"])
    def test_mask_is_causal_and_windowed(self, window: int | None) -> None:
        query = torch.arange(SEQ)[:, None]
        key = torch.arange(SEQ)[None, :]
        zero = torch.tensor(0)

        mask = ulysses.full_sequence_block_mask(SEQ, window, torch.device("cpu"))

        allowed = key <= query
        if window is not None:
            allowed &= key > query - window
        assert mask.seq_lengths == (SEQ, SEQ)
        assert torch.equal(mask.mask_mod(zero, zero, query, key), allowed)


class TestReplicateKvHeads:
    def test_repeats_heads_to_cp_size(self) -> None:
        torch.manual_seed(0)
        data = torch.arange(2 * 3 * 2 * 4).reshape(2, 3, 2, 4).float()

        out = replicate_kv_heads(data, 4)

        assert out.shape == (2, 3, 4, 4)
        assert torch.equal(out[..., 0, :], data[..., 0, :])
        assert torch.equal(out[..., 1, :], data[..., 0, :])
        assert torch.equal(out[..., 2, :], data[..., 1, :])
        assert torch.equal(out[..., 3, :], data[..., 1, :])

    def test_rejects_when_heads_do_not_divide_cp(self) -> None:
        with pytest.raises(ValueError, match="must divide cp_size"):
            replicate_kv_heads(torch.zeros(1, 2, 3, 4), 4)


class TestPublishedUlyssesParams:
    def test_restores_previous_params(self) -> None:
        publish_varlen_offsets(torch.tensor([0, 4]))

        with PublishedUlyssesParams(
            {"cu_seqlens": torch.tensor([0, 8]), "max_seqlen": 8}
        ):
            assert ULYSSES_PARAMS["max_seqlen"] == 8

        assert ULYSSES_PARAMS["max_seqlen"] == 4
        clear_ulysses_params()

    def test_clear_drops_published_params(self) -> None:
        publish_varlen_offsets(torch.tensor([0, 2]))

        clear_ulysses_params()

        assert ULYSSES_PARAMS == {}


class TestAttentionHeadCounts:
    def test_reads_config_heads(self) -> None:
        model = nn.Linear(2, 2)
        model.config = PretrainedConfig(num_attention_heads=8, num_key_value_heads=2)

        assert attention_head_counts(model) == (8, 2)

    def test_reads_vision_language_text_config_heads(self) -> None:
        model = nn.Linear(2, 2)
        model.config = Qwen3_5Config(
            text_config={"num_attention_heads": 24, "num_key_value_heads": 4}
        )

        assert attention_head_counts(model) == (24, 4)

    def test_missing_heads_raises(self) -> None:
        model = nn.Linear(2, 2)
        model.config = PretrainedConfig()

        with pytest.raises(AttributeError, match="num_attention_heads"):
            attention_head_counts(model)


class TestModelAttentionBackend:
    def test_reads_attn_implementation(self) -> None:
        model = nn.Linear(2, 2)
        model.config = PretrainedConfig(attn_implementation="flash_attention_2")

        assert model_attention_backend(model) == "flash_attention_2"


class TestModelConfig:
    def test_reads_the_peft_base_model_config(self) -> None:
        # Arrange
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
        )
        peft_model = get_peft_model(
            Qwen3ForCausalLM(config), LoraConfig(r=2, target_modules=["q_proj"])
        )

        # Act
        resolved = ulysses.model_config(peft_model)

        # Assert
        assert resolved.num_attention_heads == 4
        assert resolved.num_key_value_heads == 2

    def test_raises_without_a_config(self) -> None:
        with pytest.raises(AttributeError, match="config"):
            ulysses.model_config(nn.Linear(2, 2))


def _qkv_exchange_worker(rank: int, world_size: int) -> None:
    """One q/k/v all-to-all gives each rank its heads of the full sequence."""
    # Arrange
    torch.manual_seed(0)
    query = torch.randn(BATCH, SEQ, 4, 3)
    key = torch.randn(BATCH, SEQ, 2, 3)
    value = torch.randn(BATCH, SEQ, 2, 5)
    local = slice(rank * SEQ // world_size, (rank + 1) * SEQ // world_size)
    shards = [
        tensor[:, local].clone().requires_grad_(True) for tensor in (query, key, value)
    ]

    # Act
    exchanged = ulysses.all_to_all_qkv_seq_to_head(
        *shards, world_size, dist.group.WORLD
    )
    restored = [
        ulysses.all_to_all_head_to_seq(tensor, world_size, dist.group.WORLD)
        for tensor in exchanged
    ]
    sum(tensor.sum() for tensor in restored).backward()

    # Assert
    for name, full, out in zip(
        ("query", "key", "value"), (query, key, value), exchanged, strict=True
    ):
        heads = full.shape[2] // world_size
        assert_exact(
            out, full[:, :, rank * heads : (rank + 1) * heads], f"{name} heads"
        )
    for name, shard, out in zip(
        ("query", "key", "value"), shards, restored, strict=True
    ):
        assert_exact(out, shard, f"{name} round trip")
        assert_exact(shard.grad, torch.ones_like(shard), f"{name} grad")


class TestAllToAllQkvSeqToHead:
    @pytest.mark.multi_rank
    def test_each_rank_gets_its_heads_of_the_full_sequence(self) -> None:
        run_ranks(_qkv_exchange_worker, 2, timeout=300)

    def test_rejects_heads_not_divisible_by_cp(self) -> None:
        query = torch.zeros(1, 4, 4, 2)

        with pytest.raises(ValueError, match=r"num_heads \(3\) must be divisible"):
            ulysses.all_to_all_qkv_seq_to_head(
                query, torch.zeros(1, 4, 3, 2), torch.zeros(1, 4, 3, 2), 2, None
            )


class TestAllToAllHeadToSeq:
    def test_rejects_sequence_not_divisible_by_cp(self) -> None:
        with pytest.raises(
            ValueError, match=r"global sequence length \(5\) must be divisible"
        ):
            ulysses.all_to_all_head_to_seq(torch.zeros(1, 5, 2, 2), 2, None)


class AttentionModule(nn.Module):
    def __init__(self, is_causal: bool) -> None:
        super().__init__()
        self.is_causal = is_causal


def _qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return (
        torch.randn(1, 2, 4, 3),
        torch.randn(1, 2, 4, 3),
        torch.randn(1, 2, 4, 3),
    )


@pytest.fixture
def published_params():
    publish_varlen_offsets(torch.tensor([0, 4], dtype=torch.int32))
    yield
    clear_ulysses_params()


@pytest.mark.usefixtures("gloo_process_group")
class TestUlyssesFlashAttnVarlenFunc:
    def test_passes_optional_flash_arguments(self) -> None:
        # Arrange
        seen: dict[str, Any] = {}

        def flash_fn(query, _key, _value, cu_q, _cu_k, max_q, _max_k, **kwargs):
            seen.update(kwargs, shape=tuple(query.shape), cu=cu_q.tolist(), max=max_q)
            return query * 2

        query = torch.randn(1, 4, 2, 3)
        cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)

        # Act
        out = ulysses.ulysses_flash_attn_varlen_func(
            flash_fn,
            query,
            query,
            query,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=4,
            max_seqlen_k=4,
            causal=True,
            cp_group=dist.group.WORLD,
            cp_size=1,
            window_size=(1, 1),
            softmax_scale=0.5,
            dropout_p=0.1,
            deterministic=True,
        )

        # Assert
        assert torch.equal(out, query * 2)
        assert seen == {
            "causal": True,
            "window_size": (1, 1),
            "softmax_scale": 0.5,
            "dropout_p": 0.1,
            "deterministic": True,
            "shape": (4, 2, 3),
            "cu": [0, 4],
            "max": 4,
        }


@pytest.fixture
def stock_attention(monkeypatch) -> dict[str, Any]:
    """Stand-in stock flex and flash entrypoints, restored after the test."""

    @functools.wraps(flash_utils._flash_attention_forward)
    def stock_flash(*_args: Any, **_kwargs: Any) -> str:
        return "stock-flash"

    def stock_flash_v2(*_args: Any, **_kwargs: Any) -> tuple[str, None]:
        return "stock-flash-v2", None

    def stock_flex(*_args: Any, **_kwargs: Any) -> tuple[str, None]:
        return "stock-flex", None

    monkeypatch.setattr(flash_utils, "_flash_attention_forward", stock_flash)
    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS, "flash_attention_2", stock_flash_v2)
    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS, "flex_attention", stock_flex)
    return {"flash": stock_flash, "flash_v2": stock_flash_v2, "flex": stock_flex}


@pytest.fixture
def fake_flash_attn(monkeypatch) -> list[dict[str, Any]]:
    """Install a ``flash_attn`` module whose varlen kernel records its kwargs."""
    calls: list[dict[str, Any]] = []

    def flash_attn_varlen_func(query, *_args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(kwargs)
        return query

    module = types.ModuleType("flash_attn")
    module.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    module.flash_attn_varlen_func = flash_attn_varlen_func
    monkeypatch.setitem(sys.modules, "flash_attn", module)
    return calls


@pytest.mark.usefixtures("gloo_process_group", "stock_attention")
class TestSubstituteHfUlyssesAttn:
    def test_flex_patch_is_idempotent(self) -> None:
        substitute_hf_ulysses_attn(dist.group.WORLD, "flex_attention")
        patched = ALL_ATTENTION_FUNCTIONS["flex_attention"]

        substitute_hf_ulysses_attn(dist.group.WORLD, "flex_attention")

        assert ALL_ATTENTION_FUNCTIONS["flex_attention"] is patched

    def test_flex_without_published_params_runs_stock(self) -> None:
        substitute_hf_ulysses_attn(dist.group.WORLD, "flex_attention")
        query, key, value = _qkv()

        out = ALL_ATTENTION_FUNCTIONS["flex_attention"](
            AttentionModule(is_causal=True), query, key, value, None
        )

        assert out == ("stock-flex", None)

    @pytest.mark.usefixtures("published_params")
    def test_flex_rejects_non_causal_attention(self) -> None:
        substitute_hf_ulysses_attn(dist.group.WORLD, "flex_attention")
        query, key, value = _qkv()

        with pytest.raises(ValueError, match="only supports causal"):
            ALL_ATTENTION_FUNCTIONS["flex_attention"](
                AttentionModule(is_causal=False), query, key, value, None
            )

    def test_flash_requires_flash_attn(
        self, monkeypatch, stock_attention: dict[str, Any]
    ) -> None:
        monkeypatch.setitem(sys.modules, "flash_attn", None)

        with pytest.raises(ImportError, match="needs the flash_attn package"):
            substitute_hf_ulysses_attn(dist.group.WORLD, "flash_attention_2")

        assert flash_utils._flash_attention_forward is stock_attention["flash"]
        assert (
            ALL_ATTENTION_FUNCTIONS["flash_attention_2"] is stock_attention["flash_v2"]
        )

    @pytest.mark.usefixtures("published_params")
    def test_flash_runs_varlen_flash_on_published_params(
        self, fake_flash_attn: list[dict[str, Any]]
    ) -> None:
        # Arrange
        substitute_hf_ulysses_attn(dist.group.WORLD, "flash_attention_2")
        query, key, value = _qkv()

        # Act
        out, weights = ALL_ATTENTION_FUNCTIONS["flash_attention_2"](
            AttentionModule(is_causal=True),
            query,
            key,
            value,
            None,
            dropout=0.1,
            scaling=0.5,
            sliding_window=2,
        )

        # Assert
        assert torch.equal(out, query.transpose(1, 2))
        assert weights is None
        assert fake_flash_attn == [
            {
                "causal": True,
                "window_size": (1, 1),
                "softmax_scale": 0.5,
                "dropout_p": 0.1,
            }
        ]

    @pytest.mark.usefixtures("fake_flash_attn")
    def test_flash_patch_is_idempotent(self) -> None:
        substitute_hf_ulysses_attn(dist.group.WORLD, "flash_attention_2")
        patched = flash_utils._flash_attention_forward
        patched_v2 = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

        substitute_hf_ulysses_attn(dist.group.WORLD, "flash_attention_2")

        assert flash_utils._flash_attention_forward is patched
        assert ALL_ATTENTION_FUNCTIONS["flash_attention_2"] is patched_v2

    @pytest.mark.usefixtures("fake_flash_attn")
    def test_flash_without_published_params_runs_stock(self) -> None:
        substitute_hf_ulysses_attn(dist.group.WORLD, "flash_attention_2")
        query, key, value = _qkv()

        out = flash_utils._flash_attention_forward(
            query, key, value, None, query_length=4, is_causal=True
        )
        out_v2 = ALL_ATTENTION_FUNCTIONS["flash_attention_2"](
            AttentionModule(is_causal=True), query, key, value, None
        )

        assert out == "stock-flash"
        assert out_v2 == ("stock-flash-v2", None)

    @pytest.mark.usefixtures("fake_flash_attn", "published_params")
    @pytest.mark.parametrize(
        ("is_causal", "softcap", "match"),
        [(False, None, "only supports causal"), (True, 1.0, "softcap")],
        ids=["non_causal", "softcap"],
    )
    def test_flash_rejects_unsupported_attention(
        self, is_causal: bool, softcap: float | None, match: str
    ) -> None:
        substitute_hf_ulysses_attn(dist.group.WORLD, "flash_attention_2")
        query, key, value = _qkv()

        with pytest.raises(ValueError, match=match):
            flash_utils._flash_attention_forward(
                query,
                key,
                value,
                None,
                query_length=4,
                is_causal=is_causal,
                softcap=softcap,
            )
