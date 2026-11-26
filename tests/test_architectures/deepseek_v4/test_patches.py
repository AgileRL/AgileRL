# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import pytest
import torch
from torch.nn.attention.flex_attention import (
    AuxOutput,
    BlockMask,
    create_block_mask,
    create_mask,
    flex_attention,
)
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4HyperConnection,
    DeepseekV4Indexer,
    DeepseekV4IndexerScorer,
    DeepseekV4Model,
    DeepseekV4PreTrainedModel,
)

from agilerl.architectures.catalog import FAMILY_RUNTIME_CONFIGS
from agilerl.architectures.deepseek_v4 import patches as deepseek_v4_module
from agilerl.architectures.deepseek_v4.patches import (
    CHUNKS_INDEXER_SCORES,
    FLEX_COMPRESSED_KV,
    PACKED_DOCUMENTS,
    SELECTS_ALL_WHEN_TOPK_COVERS,
    EntryMemo,
    append_entries,
    causal_entry_blocks,
    causal_entry_indices,
    chunked_scorer_forward,
    compress_documents,
    document_entries,
    flex_attention_with_sinks,
    flex_compressed_attention,
    install_deepseek_v4_patches,
    packed_document_lengths,
    pass_packed_document_lengths,
    select_all_when_topk_covers,
)
from agilerl.architectures.hyper_connections import (
    COMPILED_HYPER_CONNECTION_MIX,
    hyper_connection_forward,
)
from tests.helper_functions import (
    load_without_llm_dependencies,
    skip_torch_compile_on_windows_cpu,
)
from tests.parallel_helpers import dense_flex_attention

HIDDEN = 64
Q_LORA_RANK = 32
COMPRESS_RATE = 4
INDEX_TOPK = 4
BATCH = 2
SEED = 0
# Default init (std 0.02) leaves gradients near the comparison tolerance.
INIT_STD = 0.1
# Same ops in the same order on CPU fp32; the only change is which entries are scored.
ATOL = 1e-6
RTOL = 1e-5
COVERED_SEQ = COMPRESS_RATE * INDEX_TOPK
UNCOVERED_SEQ = COMPRESS_RATE * (INDEX_TOPK + 1)
CSA_ATTENTION = "layers.1.self_attn."

STOCK_INDEXER_FORWARD = DeepseekV4Indexer.forward
PATCHED_INDEXER_FORWARD = select_all_when_topk_covers(STOCK_INDEXER_FORWARD)
STOCK_SCORER_FORWARD = DeepseekV4IndexerScorer.__dict__["forward"]
STOCK_ATTENTION_FORWARD = DeepseekV4Attention.__dict__["forward"]
STOCK_MODEL_FORWARD = DeepseekV4Model.__dict__["forward"]
STOCK_HYPER_CONNECTION_FORWARD = DeepseekV4HyperConnection.__dict__["forward"]

HEADS = 4
CSA_RATE = 4
HCA_RATE = 8
# 32 tokens give 8 CSA entries (the indexer keeps 4 of them) and 4 HCA entries.
FLEX_SEQ = 32
# The 20-token document has 5 CSA entries, so the indexer scores inside a document too.
DOCUMENT_LENGTHS = (20, 12)
# fp32 CPU: flex and eager sum the same scores in a different order.
FLEX_ATOL = 1e-5
FLEX_RTOL = 1e-5
# Small mask blocks so a few queries span several blocks, some full, some partial.
MASK_BLOCK = 4
# 11 tokens: the sliding keys end inside a mask block. Rate 2 gives 3 + 2 entries.
SPLICE_DOCUMENTS = (6, 5)
SPLICE_RATE = 2
SPLICE_WINDOW = 8


def tiny_config(
    num_hidden_layers: int = 2,
    layer_types: list[str] | None = None,
    compress_rates: dict[str, int] | None = None,
) -> DeepseekV4Config:
    config = DeepseekV4Config(
        vocab_size=32,
        hidden_size=HIDDEN,
        moe_intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=HEADS,
        num_key_value_heads=1,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=Q_LORA_RANK,
        o_groups=2,
        o_lora_rank=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=0,
        # By default one sliding layer, one compressed sparse attention layer.
        compress_ratios=[0, COMPRESS_RATE],
        layer_types=layer_types,
        compress_rates=compress_rates,
        sliding_window=8,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=INDEX_TOPK,
        hc_mult=4,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return config


def reinit(module: torch.nn.Module) -> None:
    torch.manual_seed(SEED)
    for parameter in module.parameters():
        torch.nn.init.normal_(parameter, std=INIT_STD)


def indexer_inputs(
    seq_len: int,
) -> tuple[DeepseekV4Indexer, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(SEED)
    indexer = DeepseekV4Indexer(tiny_config())
    reinit(indexer)
    hidden_states = torch.randn(BATCH, seq_len, HIDDEN)
    q_residual = torch.randn(BATCH, seq_len, Q_LORA_RANK)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    return indexer, hidden_states, q_residual, position_ids


def run_model(
    model: DeepseekV4Model, inputs: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]:
    model.zero_grad(set_to_none=True)
    inputs = inputs.clone().requires_grad_(True)
    output = model(inputs_embeds=inputs, use_cache=False).last_hidden_state
    (output * weights).sum().backward()
    assert inputs.grad is not None
    grads = {
        name: None if parameter.grad is None else parameter.grad.clone()
        for name, parameter in model.named_parameters()
    }
    return output.detach(), inputs.grad, grads


def flex_model() -> DeepseekV4Model:
    """Sliding, CSA and HCA layers with distinct per-head sinks."""
    config = tiny_config(
        num_hidden_layers=3,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        compress_rates={
            "compressed_sparse_attention": CSA_RATE,
            "heavily_compressed_attention": HCA_RATE,
        },
    )
    model = DeepseekV4Model(config)
    reinit(model)
    with torch.no_grad():
        for layer in model.layers:
            layer.self_attn.sinks.copy_(torch.linspace(-1.0, 1.0, HEADS))
    return model


def run_with_backend(
    model: DeepseekV4Model,
    attn_implementation: str,
    inputs: torch.Tensor,
    weights: torch.Tensor,
    **kwargs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]:
    model.config._attn_implementation = attn_implementation
    model.zero_grad(set_to_none=True)
    inputs = inputs.clone().requires_grad_(True)
    output = model(inputs_embeds=inputs, use_cache=False, **kwargs).last_hidden_state
    (output * weights).sum().backward()
    assert inputs.grad is not None
    grads = {
        name: None if parameter.grad is None else parameter.grad.clone()
        for name, parameter in model.named_parameters()
    }
    return output.detach(), inputs.grad, grads


def forward_with_backend(
    model: DeepseekV4Model,
    attn_implementation: str,
    inputs: torch.Tensor,
    **kwargs: torch.Tensor,
) -> torch.Tensor:
    model.config._attn_implementation = attn_implementation
    with torch.no_grad():
        return model(inputs_embeds=inputs, use_cache=False, **kwargs).last_hidden_state


def decode_with_cache(
    model: DeepseekV4Model,
    attn_implementation: str,
    inputs: torch.Tensor,
    prefill: int,
) -> torch.Tensor:
    """Prefill ``prefill`` tokens on the config's default cache, then decode the rest one at a time."""
    model.config._attn_implementation = attn_implementation
    with torch.no_grad():
        output = model(inputs_embeds=inputs[:, :prefill])
        outputs = [output.last_hidden_state]
        for step in range(prefill, inputs.shape[1]):
            output = model(
                inputs_embeds=inputs[:, step : step + 1],
                past_key_values=output.past_key_values,
            )
            outputs.append(output.last_hidden_state)
    return torch.cat(outputs, dim=1)


def packed_position_ids(lengths: tuple[int, ...], rows: int) -> torch.Tensor:
    positions = torch.cat([torch.arange(length) for length in lengths])
    return positions.unsqueeze(0).expand(rows, -1).contiguous()


def assert_grads_close(
    grads: dict[str, torch.Tensor | None], expected: dict[str, torch.Tensor | None]
) -> None:
    for name, expected_grad in expected.items():
        grad = grads[name]
        assert (grad is None) == (expected_grad is None), name
        if grad is not None and expected_grad is not None:
            assert torch.allclose(
                grad, expected_grad, atol=FLEX_ATOL, rtol=FLEX_RTOL
            ), name


def uncompiled_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    training: bool,
    **kwargs: object,
) -> tuple[torch.Tensor, AuxOutput]:
    return flex_attention(query, key, value, **kwargs)


@pytest.fixture
def flex_patched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DeepseekV4Indexer, "forward", STOCK_INDEXER_FORWARD)
    monkeypatch.setattr(
        DeepseekV4Attention,
        "forward",
        flex_compressed_attention(STOCK_ATTENTION_FORWARD),
    )
    monkeypatch.setattr(
        DeepseekV4Model, "forward", pass_packed_document_lengths(STOCK_MODEL_FORWARD)
    )


def use_flex_kernel(
    monkeypatch: pytest.MonkeyPatch, kernel: Callable[..., object]
) -> None:
    # CPU flex has no backward, and compiled CPU flex returns no lse.
    monkeypatch.setattr(deepseek_v4_module, "compile_friendly_flex_attention", kernel)


class TestPackedDocumentLengths:
    def test_returns_lengths_shared_by_every_row(self) -> None:
        lengths = packed_document_lengths(packed_position_ids((3, 2), rows=2))

        assert lengths == (3, 2)

    def test_single_document_rows_give_none(self) -> None:
        assert packed_document_lengths(packed_position_ids((5,), rows=2)) is None

    def test_rows_with_different_layouts_raise(self) -> None:
        position_ids = torch.stack(
            [packed_position_ids((3, 2), 1)[0], packed_position_ids((2, 3), 1)[0]]
        )

        with pytest.raises(ValueError, match="share one document layout"):
            packed_document_lengths(position_ids)


def listed_blocks(
    num_blocks: torch.Tensor, indices: torch.Tensor, blocks: int
) -> torch.Tensor:
    """``[batch, rows, blocks]`` bool of the blocks a BlockMask list names per row."""
    valid = torch.arange(indices.shape[-1]) < num_blocks.unsqueeze(-1)
    named = torch.where(valid, indices.long(), blocks)
    dense = torch.zeros((*num_blocks.shape, blocks + 1), dtype=torch.bool)
    return dense.scatter_(-1, named, True)[:, 0, :, :blocks]


def block_visibility(
    block_mask: BlockMask, full_kv: tuple[torch.Tensor, torch.Tensor] | None
) -> torch.Tensor:
    """``[batch, queries, keys]`` keys flex reads: ``mask_mod`` in partial blocks, all of a full block."""
    q_len, kv_len = block_mask.seq_lengths
    q_block, kv_block = block_mask.BLOCK_SIZE
    batch = block_mask.kv_num_blocks.shape[0]
    kv_blocks = -(-kv_len // kv_block)

    def tokens(blocks: torch.Tensor) -> torch.Tensor:
        rows = blocks.repeat_interleave(q_block, dim=1)[:, :q_len]
        return rows.repeat_interleave(kv_block, dim=2)[:, :, :kv_len]

    partial = tokens(
        listed_blocks(block_mask.kv_num_blocks, block_mask.kv_indices, kv_blocks)
    )
    full = torch.zeros(batch, q_len, kv_len, dtype=torch.bool)
    if full_kv is not None:
        full = tokens(listed_blocks(*full_kv, kv_blocks))
    dense = create_mask(block_mask.mask_mod, batch, 1, q_len, kv_len, "cpu")[:, 0]
    return (partial & dense) | full


def assert_query_lists_transpose_key_lists(block_mask: BlockMask) -> None:
    q_len, kv_len = block_mask.seq_lengths
    q_blocks = -(-q_len // block_mask.BLOCK_SIZE[0])
    kv_blocks = -(-kv_len // block_mask.BLOCK_SIZE[1])
    pairs = [
        (
            block_mask.kv_num_blocks,
            block_mask.kv_indices,
            block_mask.q_num_blocks,
            block_mask.q_indices,
        )
    ]
    if block_mask.full_kv_num_blocks is not None:
        pairs.append(
            (
                block_mask.full_kv_num_blocks,
                block_mask.full_kv_indices,
                block_mask.full_q_num_blocks,
                block_mask.full_q_indices,
            )
        )
    for kv_num, kv_indices, q_num, q_indices in pairs:
        by_query = listed_blocks(kv_num, kv_indices, kv_blocks)
        by_key = listed_blocks(q_num, q_indices, q_blocks)
        assert torch.equal(by_key, by_query.transpose(1, 2))


def splice_inputs(
    separate_full_blocks: bool,
) -> tuple[torch.Tensor, torch.Tensor, BlockMask, BlockMask, torch.Tensor]:
    torch.manual_seed(SEED)
    seq_len = sum(SPLICE_DOCUMENTS)
    sliding_mask = create_block_mask(
        lambda b, h, q_idx, kv_idx: (
            (kv_idx <= q_idx) & (kv_idx > q_idx - SPLICE_WINDOW)
        ),
        B=BATCH,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device="cpu",
        BLOCK_SIZE=MASK_BLOCK,
    )
    position_ids = packed_position_ids(SPLICE_DOCUMENTS, 1)
    _, layout = document_entries(SPLICE_DOCUMENTS, SPLICE_RATE, torch.device("cpu"))
    entry_blocks = causal_entry_blocks(
        layout,
        position_ids,
        SPLICE_RATE,
        BATCH,
        (MASK_BLOCK, MASK_BLOCK),
        separate_full_blocks,
    )
    kv = torch.randn(BATCH, 1, seq_len, 3)
    entries = torch.randn(BATCH, 1, layout.entry_window.numel(), 3)
    return kv, entries, sliding_mask, entry_blocks, position_ids


def expected_splice(entry_visible: torch.Tensor) -> torch.Tensor:
    """Sliding window over the tokens, one masked padding key, then the entries."""
    seq_len = sum(SPLICE_DOCUMENTS)
    query = torch.arange(seq_len).unsqueeze(-1)
    key = torch.arange(seq_len)
    sliding = (key <= query) & (key > query - SPLICE_WINDOW)
    padding = torch.zeros(seq_len, 1, dtype=torch.bool)
    return torch.cat(
        [sliding.expand(BATCH, -1, -1), padding.expand(BATCH, -1, -1), entry_visible],
        dim=-1,
    )


def causal_document_entries() -> torch.Tensor:
    """``[batch, queries, entries]`` own-document entries below each query's threshold."""
    visible = torch.zeros(sum(SPLICE_DOCUMENTS), 5, dtype=torch.bool)
    # Document 0: tokens 0-5, entries 0-2. Document 1: tokens 6-10, entries 3-4.
    for query, (document, position) in enumerate(
        [(0, p) for p in range(6)] + [(1, p) for p in range(5)]
    ):
        first, count = (0, 3) if document == 0 else (3, 2)
        for window in range(count):
            visible[query, first + window] = window < (position + 1) // SPLICE_RATE
    return visible.expand(BATCH, -1, -1)


class TestDocumentEntries:
    def test_pads_entries_to_seq_len_over_rate_for_any_packing(self) -> None:
        # Act
        layouts = [
            document_entries(lengths, HCA_RATE, torch.device("cpu"))[1]
            for lengths in ((20, 12), (16, 16))
        ]

        # Assert
        # 2 + 1 entries pad to 32 // 8 = 4 with one entry in no document; 2 + 2 fill them.
        assert [layout.entry_document.tolist() for layout in layouts] == [
            [0, 0, 1, -1],
            [0, 0, 1, 1],
        ]
        assert layouts[0].spans == ((0, 20, 0, 2), (20, 32, 2, 3))

    def test_windows_cover_only_document_entries(self) -> None:
        # Act
        windows, _ = document_entries((20, 12), HCA_RATE, torch.device("cpu"))

        # Assert
        assert windows.starts.tolist() == [0, 8, 20]
        assert windows.positions.tolist() == [0, 8, 0]


class TestAppendEntries:
    def test_block_lists_give_each_query_its_documents_causal_entries(self) -> None:
        # Arrange
        kv, entries, sliding_mask, entry_blocks, _ = splice_inputs(True)

        # Act
        keys, block_mask = append_entries(
            kv, entries, sliding_mask, entry_blocks, entry_blocks.mask_mod
        )

        # Assert
        assert sliding_mask.full_kv_num_blocks is not None
        assert block_mask.seq_lengths == (11, 12 + 5)
        assert torch.equal(keys[:, :, :11], kv)
        assert torch.equal(keys[:, :, 11], torch.zeros(BATCH, 1, 3))
        assert torch.equal(keys[:, :, 12:], entries)
        full_kv = (block_mask.full_kv_num_blocks, block_mask.full_kv_indices)
        assert torch.equal(
            block_visibility(block_mask, full_kv),
            expected_splice(causal_document_entries()),
        )
        assert_query_lists_transpose_key_lists(block_mask)

    def test_superset_blocks_read_a_narrower_selection(self) -> None:
        # Arrange
        kv, entries, sliding_mask, entry_blocks, _ = splice_inputs(False)
        generator = torch.Generator().manual_seed(SEED)
        selected = causal_document_entries() & (
            torch.rand(BATCH, 11, 5, generator=generator) > 0.5
        )

        def selected_mod(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, entry: torch.Tensor
        ) -> torch.Tensor:
            return selected[b, q_idx, entry]

        # Act
        _, block_mask = append_entries(
            kv, entries, sliding_mask, entry_blocks, selected_mod
        )

        # Assert
        assert entry_blocks.full_kv_num_blocks is None
        full_kv = (block_mask.full_kv_num_blocks, block_mask.full_kv_indices)
        assert torch.equal(
            block_visibility(block_mask, full_kv), expected_splice(selected)
        )
        assert_query_lists_transpose_key_lists(block_mask)

    def test_masks_without_full_blocks_list_none_apart(self) -> None:
        # Arrange
        kv, entries, sliding_full, entry_blocks, _ = splice_inputs(False)
        kv_blocks = sliding_full.kv_indices.shape[-1]
        # Every key block is partial, so mask_mod alone decides the sliding keys.
        sliding_mask = BlockMask.from_kv_blocks(
            torch.full_like(sliding_full.kv_num_blocks, kv_blocks),
            torch.arange(kv_blocks, dtype=torch.int32).expand_as(
                sliding_full.kv_indices
            ),
            BLOCK_SIZE=sliding_full.BLOCK_SIZE,
            mask_mod=sliding_full.mask_mod,
            seq_lengths=sliding_full.seq_lengths,
        )

        # Act
        _, block_mask = append_entries(
            kv, entries, sliding_mask, entry_blocks, entry_blocks.mask_mod
        )

        # Assert
        assert block_mask.full_kv_num_blocks is None
        assert block_mask.full_kv_indices is None
        assert block_mask.full_q_num_blocks is None
        assert block_mask.full_q_indices is None
        assert torch.equal(
            block_visibility(block_mask, None),
            expected_splice(causal_document_entries()),
        )
        assert_query_lists_transpose_key_lists(block_mask)


def compressor_inputs(
    layer: int,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor]:
    model = flex_model()
    compressor = model.layers[layer].self_attn.compressor
    torch.manual_seed(SEED)
    hidden_states = torch.randn(BATCH, FLEX_SEQ, HIDDEN)
    q_residual = torch.randn(BATCH, FLEX_SEQ, Q_LORA_RANK)
    return (
        compressor,
        hidden_states,
        q_residual,
        packed_position_ids(DOCUMENT_LENGTHS, 1),
    )


def stock_documents(
    compressor: torch.nn.Module,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stock compressor run on each document alone: entries and block-diagonal visibility."""
    entries, blocks = [], []
    for start, end in ((0, 20), (20, 32)):
        compressed, block_bias = compressor(
            hidden_states[:, start:end],
            q_residual[:, start:end],
            position_ids[:, start:end],
            None,
            1,
        )
        entries.append(compressed)
        blocks.append(block_bias[:, 0] == 0)
    first = blocks[0].shape[2]
    visible = torch.zeros(BATCH, FLEX_SEQ, first + blocks[1].shape[2], dtype=torch.bool)
    visible[:, :20, :first] = blocks[0]
    visible[:, 20:, first:] = blocks[1]
    return torch.cat(entries, dim=2), visible


@pytest.mark.usefixtures("flex_patched")
class TestCompressDocuments:
    def test_csa_entries_and_selection_match_stock_per_document(self) -> None:
        # Arrange
        compressor, hidden_states, q_residual, position_ids = compressor_inputs(1)

        # Act
        entries, layout, visible = compress_documents(
            compressor, hidden_states, q_residual, position_ids, DOCUMENT_LENGTHS
        )

        # Assert
        stock_entries, stock_visible = stock_documents(
            compressor, hidden_states, q_residual, position_ids
        )
        assert layout.spans == ((0, 20, 0, 5), (20, 32, 5, 8))
        assert torch.allclose(entries, stock_entries, atol=ATOL, rtol=RTOL)
        assert visible is not None
        assert torch.equal(visible, stock_visible)
        assert int(visible[:, 19, :5].sum(-1).max()) == INDEX_TOPK

    def test_hca_entries_and_causal_mask_match_stock_per_document(self) -> None:
        # Arrange
        compressor, hidden_states, q_residual, position_ids = compressor_inputs(2)

        # Act
        entries, layout, visible = compress_documents(
            compressor, hidden_states, q_residual, position_ids, DOCUMENT_LENGTHS
        )
        entry_blocks = causal_entry_blocks(
            layout, position_ids, HCA_RATE, BATCH, (MASK_BLOCK, MASK_BLOCK), True
        )

        # Assert
        stock_entries, stock_visible = stock_documents(
            compressor, hidden_states, q_residual, position_ids
        )
        # 2 + 1 document entries, then one padding entry up to 32 // 8.
        assert visible is None
        assert entries.shape[2] == FLEX_SEQ // HCA_RATE
        assert torch.allclose(entries[:, :, :3], stock_entries, atol=ATOL, rtol=RTOL)
        assert torch.equal(entries[:, :, 3:], torch.zeros_like(entries[:, :, 3:]))
        dense = create_mask(
            entry_blocks.mask_mod, 1, 1, FLEX_SEQ, entries.shape[2], "cpu"
        )[:, 0]
        assert torch.equal(dense[..., :3].expand(BATCH, -1, -1), stock_visible)
        assert not dense[..., 3:].any()

    @pytest.mark.parametrize("layer", [1, 2])
    def test_gradients_match_stock_per_document(self, layer: int) -> None:
        # Arrange
        compressor, hidden_states, q_residual, position_ids = compressor_inputs(layer)
        hidden_states.requires_grad_(True)
        weights = torch.randn(BATCH, 1, 8 if layer == 1 else 3, 32)

        def grads(entries: torch.Tensor) -> list[torch.Tensor | None]:
            compressor.zero_grad(set_to_none=True)
            hidden_states.grad = None
            (entries * weights).sum().backward()
            assert hidden_states.grad is not None
            return [hidden_states.grad.clone()] + [
                None if parameter.grad is None else parameter.grad.clone()
                for parameter in compressor.parameters()
            ]

        # Act
        entries, _, _ = compress_documents(
            compressor, hidden_states, q_residual, position_ids, DOCUMENT_LENGTHS
        )
        patched = grads(entries[:, :, : weights.shape[2]])
        stock = grads(
            stock_documents(compressor, hidden_states, q_residual, position_ids)[0]
        )

        # Assert
        assert patched[0].abs().max() > 1e-2
        for grad, stock_grad in zip(patched, stock, strict=True):
            assert (grad is None) == (stock_grad is None)
            if grad is not None and stock_grad is not None:
                assert torch.allclose(grad, stock_grad, atol=ATOL, rtol=RTOL)

    def test_query_chunks_select_the_same_entries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        compressor, hidden_states, q_residual, position_ids = compressor_inputs(1)
        _, _, whole = compress_documents(
            compressor, hidden_states, q_residual, position_ids, DOCUMENT_LENGTHS
        )
        # Three queries per chunk: batch 2 x 4 heads x 5 entries x 3 queries.
        monkeypatch.setattr(deepseek_v4_module, "INDEXER_SCORE_ELEMENTS", 120)

        # Act
        _, _, chunked = compress_documents(
            compressor,
            hidden_states,
            q_residual,
            packed_position_ids(DOCUMENT_LENGTHS, 1),
            DOCUMENT_LENGTHS,
        )

        # Assert
        assert whole is not None
        assert chunked is not None
        assert torch.equal(chunked, whole)


def scorer_inputs() -> tuple[
    DeepseekV4IndexerScorer, torch.Tensor, torch.Tensor, torch.Tensor
]:
    torch.manual_seed(SEED)
    scorer = DeepseekV4IndexerScorer(tiny_config())
    reinit(scorer)
    query = torch.randn(BATCH, 10, 4, 16)
    keys = torch.randn(BATCH, 6, 16)
    hidden_states = torch.randn(BATCH, 10, HIDDEN)
    return scorer, query, keys, hidden_states


class TestChunkedScorerForward:
    def test_chunked_scores_match_stock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        scorer, query, keys, hidden_states = scorer_inputs()
        # Three queries per chunk: batch 2 x 4 heads x 6 entries x 3 queries.
        monkeypatch.setattr(deepseek_v4_module, "INDEXER_SCORE_ELEMENTS", 144)

        # Act
        with torch.no_grad():
            chunked = chunked_scorer_forward(scorer, query, keys, hidden_states)
            stock = STOCK_SCORER_FORWARD(scorer, query, keys, hidden_states)

        # Assert
        assert chunked.shape == (BATCH, 10, 6)
        assert torch.allclose(chunked, stock, atol=ATOL, rtol=RTOL)


class TestEntryMemoGet:
    def test_builds_once_per_position_ids_and_key(self) -> None:
        # Arrange
        memo = EntryMemo()
        position_ids = torch.arange(4).unsqueeze(0)
        builds: list[str] = []

        def build(name: str) -> object:
            builds.append(name)
            return object()

        # Act
        first = memo.get(position_ids, ("a",), lambda: build("a"))
        again = memo.get(position_ids, ("a",), lambda: build("a"))
        other = memo.get(position_ids, ("b",), lambda: build("b"))

        # Assert
        assert again is first
        assert other is not first
        assert builds == ["a", "b"]

    def test_new_position_ids_rebuild(self) -> None:
        # Arrange
        memo = EntryMemo()
        first = memo.get(torch.arange(4).unsqueeze(0), ("a",), object)

        # Act
        second = memo.get(torch.arange(4).unsqueeze(0), ("a",), object)

        # Assert
        assert second is not first


class TestFlexAttentionWithSinks:
    def test_matches_softmax_over_keys_and_a_sink_column(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        use_flex_kernel(monkeypatch, uncompiled_flex_attention)
        torch.manual_seed(SEED)
        query = torch.randn(BATCH, HEADS, 6, 8)
        key_value = torch.randn(BATCH, 1, 6, 8)
        sinks = torch.linspace(-2.0, 2.0, HEADS)
        block_mask = create_block_mask(
            lambda b, h, q_idx, kv_idx: kv_idx <= q_idx,
            B=BATCH,
            H=None,
            Q_LEN=6,
            KV_LEN=6,
            device="cpu",
        )

        # Act
        with torch.no_grad():
            output = flex_attention_with_sinks(
                query, key_value, block_mask, sinks, 8**-0.5, training=False
            )

        # Assert
        scores = (query @ key_value.transpose(-1, -2)) * 8**-0.5
        scores = scores.masked_fill(
            ~torch.ones(6, 6, dtype=torch.bool).tril(), float("-inf")
        )
        sink_column = sinks.view(1, -1, 1, 1).expand(BATCH, -1, 6, 1)
        probs = torch.cat([scores, sink_column], dim=-1).softmax(-1)[..., :-1]
        assert torch.allclose(output, probs @ key_value, atol=FLEX_ATOL, rtol=FLEX_RTOL)


@skip_torch_compile_on_windows_cpu
@pytest.mark.usefixtures("flex_patched")
class TestDeepseekV4ModelFlexAttention:
    def test_forward_matches_eager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        use_flex_kernel(monkeypatch, uncompiled_flex_attention)
        model = flex_model().eval()
        inputs = torch.randn(BATCH, FLEX_SEQ, HIDDEN)

        # Act
        eager = forward_with_backend(model, "eager", inputs)
        flex = forward_with_backend(model, "flex_attention", inputs)

        # Assert
        assert torch.allclose(flex, eager, atol=FLEX_ATOL, rtol=FLEX_RTOL)

    def test_gradients_match_eager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        use_flex_kernel(monkeypatch, dense_flex_attention)
        model = flex_model()
        inputs = torch.randn(BATCH, FLEX_SEQ, HIDDEN)
        weights = torch.randn(BATCH, FLEX_SEQ, HIDDEN)

        # Act
        eager, eager_input_grad, eager_grads = run_with_backend(
            model, "eager", inputs, weights
        )
        flex, input_grad, grads = run_with_backend(
            model, "flex_attention", inputs, weights
        )

        # Assert
        assert torch.allclose(flex, eager, atol=FLEX_ATOL, rtol=FLEX_RTOL)
        assert torch.allclose(
            input_grad, eager_input_grad, atol=FLEX_ATOL, rtol=FLEX_RTOL
        )
        assert_grads_close(grads, eager_grads)
        for layer in range(3):
            sink_grad = grads[f"layers.{layer}.self_attn.sinks"]
            assert sink_grad is not None
            assert sink_grad.abs().min() > 1e-4

    def test_packed_rows_match_documents_run_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        use_flex_kernel(monkeypatch, uncompiled_flex_attention)
        model = flex_model().eval()
        inputs = torch.randn(BATCH, FLEX_SEQ, HIDDEN)
        position_ids = packed_position_ids(DOCUMENT_LENGTHS, BATCH)

        # Act
        packed = forward_with_backend(
            model, "flex_attention", inputs, position_ids=position_ids
        )
        alone = torch.cat(
            [
                forward_with_backend(model, "eager", document)
                for document in inputs.split(DOCUMENT_LENGTHS, dim=1)
            ],
            dim=1,
        )

        # Assert
        assert torch.allclose(packed, alone, atol=FLEX_ATOL, rtol=FLEX_RTOL)

    def test_packed_gradients_match_documents_run_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        use_flex_kernel(monkeypatch, dense_flex_attention)
        model = flex_model()
        inputs = torch.randn(BATCH, FLEX_SEQ, HIDDEN)
        weights = torch.randn(BATCH, FLEX_SEQ, HIDDEN)
        position_ids = packed_position_ids(DOCUMENT_LENGTHS, BATCH)

        # Act
        packed, input_grad, grads = run_with_backend(
            model, "flex_attention", inputs, weights, position_ids=position_ids
        )
        alone = [
            run_with_backend(model, "eager", document, document_weights)
            for document, document_weights in zip(
                inputs.split(DOCUMENT_LENGTHS, dim=1),
                weights.split(DOCUMENT_LENGTHS, dim=1),
                strict=True,
            )
        ]

        # Assert
        assert torch.allclose(
            packed,
            torch.cat([output for output, _, _ in alone], dim=1),
            atol=FLEX_ATOL,
            rtol=FLEX_RTOL,
        )
        assert torch.allclose(
            input_grad,
            torch.cat([grad for _, grad, _ in alone], dim=1),
            atol=FLEX_ATOL,
            rtol=FLEX_RTOL,
        )
        summed: dict[str, torch.Tensor | None] = {}
        for name in grads:
            document_grads = [document[name] for _, _, document in alone]
            summed[name] = (
                None
                if any(grad is None for grad in document_grads)
                else torch.stack(document_grads).sum(0)
            )
        assert_grads_close(grads, summed)

    @pytest.mark.parametrize("pad_side", ["left", "right"])
    def test_padded_rows_match_eager(
        self, monkeypatch: pytest.MonkeyPatch, pad_side: str
    ) -> None:
        # Arrange
        use_flex_kernel(monkeypatch, uncompiled_flex_attention)
        model = flex_model().eval()
        inputs = torch.randn(BATCH, FLEX_SEQ, HIDDEN)
        attention_mask = torch.ones(BATCH, FLEX_SEQ, dtype=torch.long)
        pad = slice(0, 5) if pad_side == "left" else slice(-5, None)
        attention_mask[1, pad] = 0
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)

        # Act
        eager = forward_with_backend(
            model,
            "eager",
            inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        flex = forward_with_backend(
            model,
            "flex_attention",
            inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # Assert
        assert torch.allclose(flex, eager, atol=FLEX_ATOL, rtol=FLEX_RTOL)

    def test_attention_dropout_in_training_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        use_flex_kernel(monkeypatch, dense_flex_attention)
        model = flex_model()
        for layer in model.layers:
            layer.self_attn.attention_dropout = 0.1
        model.config._attn_implementation = "flex_attention"

        # Act / Assert
        with pytest.raises(ValueError, match="attention dropout"):
            model(inputs_embeds=torch.randn(1, FLEX_SEQ, HIDDEN), use_cache=False)

    # A 3-token prefill holds no compressed entry at either rate; 20 holds some.
    @pytest.mark.parametrize("prefill", [3, 20])
    def test_cached_decode_matches_eager(
        self, monkeypatch: pytest.MonkeyPatch, prefill: int
    ) -> None:
        # Arrange
        use_flex_kernel(monkeypatch, uncompiled_flex_attention)
        model = flex_model().eval()
        inputs = torch.randn(BATCH, FLEX_SEQ, HIDDEN)

        # Act
        eager = decode_with_cache(model, "eager", inputs, prefill)
        flex = decode_with_cache(model, "flex_attention", inputs, prefill)

        # Assert
        assert torch.allclose(flex, eager, atol=FLEX_ATOL, rtol=FLEX_RTOL)

    def test_sequence_shorter_than_a_compress_window_matches_eager(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        use_flex_kernel(monkeypatch, uncompiled_flex_attention)
        model = flex_model().eval()
        inputs = torch.randn(BATCH, CSA_RATE - 1, HIDDEN)

        # Act
        eager = forward_with_backend(model, "eager", inputs)
        flex = forward_with_backend(model, "flex_attention", inputs)

        # Assert
        assert torch.allclose(flex, eager, atol=FLEX_ATOL, rtol=FLEX_RTOL)


class TestCausalEntryIndices:
    def test_marks_entries_past_each_query_invalid(self) -> None:
        # Arrange
        position_ids = torch.arange(8).unsqueeze(0)

        # Act
        indices = causal_entry_indices(
            position_ids, batch=2, compressed_len=2, compress_rate=4
        )

        # Assert
        row = torch.tensor(
            [[-1, -1], [-1, -1], [-1, -1], [0, -1], [0, -1], [0, -1], [0, -1], [0, 1]]
        )
        assert indices.dtype == torch.int64
        assert torch.equal(indices, row.expand(2, -1, -1))

    def test_no_compressed_entries_gives_empty_selection(self) -> None:
        indices = causal_entry_indices(
            torch.arange(3).unsqueeze(0), batch=2, compressed_len=0, compress_rate=4
        )

        assert indices.shape == (2, 3, 0)
        assert indices.dtype == torch.int64


class TestSelectAllWhenTopkCovers:
    def test_topk_equal_to_compressed_len_selects_every_causal_entry(self) -> None:
        # Arrange
        indexer, hidden_states, q_residual, position_ids = indexer_inputs(COVERED_SEQ)

        # Act
        stock = STOCK_INDEXER_FORWARD(
            indexer, hidden_states, q_residual, position_ids, None, 1
        )
        patched = PATCHED_INDEXER_FORWARD(
            indexer, hidden_states, q_residual, position_ids, None, 1
        )

        # Assert
        assert torch.equal(
            patched,
            causal_entry_indices(position_ids, BATCH, INDEX_TOPK, COMPRESS_RATE),
        )
        assert torch.equal(patched.sort(-1).values, stock.sort(-1).values)

    def test_compressed_len_one_past_topk_matches_stock(self) -> None:
        # Arrange
        indexer, hidden_states, q_residual, position_ids = indexer_inputs(UNCOVERED_SEQ)

        # Act
        stock = STOCK_INDEXER_FORWARD(
            indexer, hidden_states, q_residual, position_ids, None, 1
        )
        patched = PATCHED_INDEXER_FORWARD(
            indexer, hidden_states, q_residual, position_ids, None, 1
        )

        # Assert
        assert patched.shape == (BATCH, UNCOVERED_SEQ, INDEX_TOPK)
        assert torch.equal(patched, stock)

    @pytest.mark.parametrize("seq_len", [COVERED_SEQ, UNCOVERED_SEQ])
    def test_scoring_saves_no_activations(self, seq_len: int) -> None:
        # Arrange
        indexer, hidden_states, q_residual, position_ids = indexer_inputs(seq_len)
        saved: dict[str, int] = {"stock": 0, "patched": 0}

        def count_into(key: str) -> tuple:
            def pack(tensor: torch.Tensor) -> torch.Tensor:
                saved[key] += 1
                return tensor

            return pack, lambda tensor: tensor

        # Act
        for key, forward in (
            ("stock", STOCK_INDEXER_FORWARD),
            ("patched", PATCHED_INDEXER_FORWARD),
        ):
            with torch.autograd.graph.saved_tensors_hooks(*count_into(key)):
                forward(indexer, hidden_states, q_residual, position_ids, None, 1)

        # Assert
        assert saved["stock"] > 0
        assert saved["patched"] == 0


class TestDeepseekV4ModelWithSelectAll:
    @pytest.mark.parametrize("seq_len", [COVERED_SEQ, UNCOVERED_SEQ])
    def test_outputs_and_gradients_match_stock(
        self, monkeypatch: pytest.MonkeyPatch, seq_len: int
    ) -> None:
        # Arrange
        model = DeepseekV4Model(tiny_config())
        reinit(model)
        inputs = torch.randn(BATCH, seq_len, HIDDEN)
        weights = torch.randn(BATCH, seq_len, HIDDEN)
        monkeypatch.setattr(DeepseekV4Indexer, "forward", STOCK_INDEXER_FORWARD)
        stock_output, stock_input_grad, stock_grads = run_model(model, inputs, weights)

        # Act
        monkeypatch.setattr(DeepseekV4Indexer, "forward", PATCHED_INDEXER_FORWARD)
        output, input_grad, grads = run_model(model, inputs, weights)

        # Assert
        assert torch.allclose(output, stock_output, atol=ATOL, rtol=RTOL)
        assert torch.allclose(input_grad, stock_input_grad, atol=ATOL, rtol=RTOL)
        assert stock_input_grad.abs().max() > 1e-2
        csa_attention = [name for name in grads if name.startswith(CSA_ATTENTION)]
        assert all(
            grads[name] is not None for name in csa_attention if ".indexer." not in name
        )
        assert all(grads[name] is None for name in csa_attention if ".indexer." in name)
        for name, stock_grad in stock_grads.items():
            grad = grads[name]
            assert (grad is None) == (stock_grad is None), name
            if grad is not None and stock_grad is not None:
                assert torch.allclose(grad, stock_grad, atol=ATOL, rtol=RTOL), name


@pytest.fixture
def stock_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    for cls, forward, flag in (
        (DeepseekV4Indexer, STOCK_INDEXER_FORWARD, SELECTS_ALL_WHEN_TOPK_COVERS),
        (DeepseekV4IndexerScorer, STOCK_SCORER_FORWARD, CHUNKS_INDEXER_SCORES),
        (DeepseekV4Attention, STOCK_ATTENTION_FORWARD, FLEX_COMPRESSED_KV),
        (DeepseekV4Model, STOCK_MODEL_FORWARD, PACKED_DOCUMENTS),
        (
            DeepseekV4HyperConnection,
            STOCK_HYPER_CONNECTION_FORWARD,
            COMPILED_HYPER_CONNECTION_MIX,
        ),
    ):
        monkeypatch.setattr(cls, "forward", forward)
        monkeypatch.setattr(cls, flag, False, raising=False)
    monkeypatch.setattr(DeepseekV4PreTrainedModel, "_supports_flex_attn", False)


@pytest.mark.usefixtures("stock_classes")
class TestInstallDeepseekV4Patches:
    def test_patches_indexer_forward_once(self) -> None:
        # Arrange
        indexer, hidden_states, q_residual, position_ids = indexer_inputs(COVERED_SEQ)

        # Act
        install_deepseek_v4_patches(FAMILY_RUNTIME_CONFIGS["deepseek_v4"].patch)
        patched_forward = DeepseekV4Indexer.forward
        install_deepseek_v4_patches(FAMILY_RUNTIME_CONFIGS["deepseek_v4"].patch)

        # Assert
        assert DeepseekV4Indexer.forward is patched_forward
        assert patched_forward.__wrapped__ is STOCK_INDEXER_FORWARD
        assert torch.equal(
            indexer(hidden_states, q_residual, position_ids, None, 1),
            causal_entry_indices(position_ids, BATCH, INDEX_TOPK, COMPRESS_RATE),
        )

    def test_patches_scorer_forward(self) -> None:
        # Arrange
        scorer, query, keys, hidden_states = scorer_inputs()

        # Act
        install_deepseek_v4_patches(FAMILY_RUNTIME_CONFIGS["deepseek_v4"].patch)

        # Assert
        assert DeepseekV4IndexerScorer.forward is chunked_scorer_forward
        with torch.no_grad():
            assert torch.allclose(
                scorer(query, keys, hidden_states),
                STOCK_SCORER_FORWARD(scorer, query, keys, hidden_states),
                atol=ATOL,
                rtol=RTOL,
            )

    def test_enables_flex_attention_once(self) -> None:
        # Act
        install_deepseek_v4_patches(FAMILY_RUNTIME_CONFIGS["deepseek_v4"].patch)
        patched_forward = DeepseekV4Attention.forward
        install_deepseek_v4_patches(FAMILY_RUNTIME_CONFIGS["deepseek_v4"].patch)

        # Assert
        assert DeepseekV4Attention.forward is patched_forward
        assert patched_forward.__wrapped__ is STOCK_ATTENTION_FORWARD
        assert DeepseekV4PreTrainedModel._supports_flex_attn is True

    def test_compiles_the_hyper_connection_mix(self) -> None:
        install_deepseek_v4_patches(FAMILY_RUNTIME_CONFIGS["deepseek_v4"].patch)

        assert DeepseekV4HyperConnection.forward is hyper_connection_forward

    @pytest.mark.parametrize("use_sequence_packing", [False, True])
    def test_passes_document_lengths_only_with_packing(
        self, use_sequence_packing: bool
    ) -> None:
        # Act
        install_deepseek_v4_patches(
            FAMILY_RUNTIME_CONFIGS["deepseek_v4"].patch, None, use_sequence_packing
        )

        # Assert
        wraps_forward = DeepseekV4Model.forward is not STOCK_MODEL_FORWARD
        assert wraps_forward is use_sequence_packing
        if use_sequence_packing:
            assert DeepseekV4Model.forward.__wrapped__ is STOCK_MODEL_FORWARD

    def test_catalog_installs_the_indexer_patch(self) -> None:
        patch = FAMILY_RUNTIME_CONFIGS["deepseek_v4"].patch

        assert patch.install is install_deepseek_v4_patches


class TestImportFallbacks:
    def test_without_llm_dependencies_leaves_modeling_names_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = load_without_llm_dependencies(deepseek_v4_module, monkeypatch)

        for name in (
            "compile_friendly_flex_attention",
            "DeepseekV4Attention",
            "DeepseekV4CSACompressor",
            "DeepseekV4HCACompressor",
            "DeepseekV4HyperConnection",
            "DeepseekV4Indexer",
            "DeepseekV4IndexerScorer",
            "DeepseekV4Model",
            "DeepseekV4PreTrainedModel",
            "apply_rotary_pos_emb",
        ):
            assert getattr(module, name) is None, name
