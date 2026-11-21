# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel GLM-5.3 blocks, alone and with tensor and expert parallel, on a gloo mesh."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from torch import nn
from transformers.models.glm5_next import modeling_glm5_next
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextAttention,
    Glm5NextTextIndexer,
    Glm5NextTextLinearAttention,
)

from agilerl.algorithms.core.sequence_layout import ContextParallelSequence
from agilerl.architectures.glm5_next.context_parallel import (
    install_glm5_next_attention_context_parallel,
)
from agilerl.architectures.glm5_next.patches import (
    CAUSAL_SDPA_WHEN_TOPK_COVERS,
    SEQUENCE_SHARD,
    SequenceShard,
    causal_sdpa_attention_forward,
)
from agilerl.architectures.glm5_next.tensor_parallel import (
    GLM5_NEXT_TENSOR_PARALLEL_PLAN,
)
from agilerl.distributed import fsdp_blocks
from agilerl.distributed.context_parallel import gdn as context_parallel_gdn
from agilerl.distributed.context_parallel.sequence import (
    CP_SCAN_CHUNK,
    gather_sequence_for_cp,
    pad_right_for_cp,
    shard_for_cp,
)
from agilerl.distributed.context_parallel.ulysses import (
    clear_ulysses_params,
)
from agilerl.distributed.fsdp import materialize_fsdp2_from_cpu_state
from agilerl.distributed.runtime import FSDPRuntime
from tests.parallel_helpers import (
    assert_exact,
    assert_lora,
    assert_resharded,
    broadcast_module,
    fsdp_config,
    parallel_mesh,
    publish_varlen_offsets,
    sgd,
)
from tests.rank_launch import run_ranks
from tests.test_architectures.glm5_next.test_parallel import (
    HC_MULT,
    HIDDEN,
    TP_PROBE_WEIGHTS,
    Glm5NextBlockStack,
    _apply_liger_swaps,
    _assert_experts_on_ep,
    _assert_stepped,
    _assert_tp_blocks_sharded,
    _stepped_reference,
    _with_moe_lora,
)

BATCH = 2
# Eight k-pools of two tokens: each rank's queries pick pools from the other shard.
SEQ = 16
INDEXED_TOPK = 4
# Top-k keeps every pool, so the mask is the causal mask over the whole sequence.
COVERING_TOPK = SEQ
VOCAB = 32
# Two rows of real tokens; the layout right-pads them to cp * CP_SCAN_CHUNK + 1.
TOKEN_LENGTHS = (100, 90)
KDA_CHUNK_SIZES = (CP_SCAN_CHUNK, CP_SCAN_CHUNK // 2)


def _fake_fla_kernels() -> None:
    """Swap FLA's GPU-only CP kernels for transformers' torch kernels on the gathered sequence.

    Each shard's inputs are all-gathered over the context-parallel group, the
    torch kernel runs on the whole sequence and the rank keeps its slice, so a
    shard starts from the conv and recurrent state the earlier shards leave.
    Without a context the scan runs on the shard alone.
    """
    conv = modeling_glm5_next.causal_conv1d_fn
    scan = modeling_glm5_next.chunk_kimi_delta_attention

    def fla_conv(
        x: torch.Tensor,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        activation: str | None,
        cp_context: SimpleNamespace,
    ) -> tuple[torch.Tensor, None]:
        full = gather_sequence_for_cp(x, cp_context.group)
        out = conv(full.transpose(1, 2), weight, bias, activation=activation)
        group = cp_context.group
        own = shard_for_cp(
            out.transpose(1, 2), dist.get_rank(group), dist.get_world_size(group)
        )
        return own, None

    def fla_kda(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int,
        cp_context: SimpleNamespace | None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert chunk_size in KDA_CHUNK_SIZES
        if cp_context is None:
            return scan(q, k, v, g=g, beta=beta, chunk_size=chunk_size, **kwargs)
        group = cp_context.group
        q, k, v, g, beta = (
            gather_sequence_for_cp(t, group) for t in (q, k, v, g, beta)
        )
        out, state = scan(q, k, v, g=g, beta=beta, chunk_size=chunk_size, **kwargs)
        own = shard_for_cp(out, dist.get_rank(group), dist.get_world_size(group))
        return own, state

    context_parallel_gdn.gdn_kernel_import_errors = lambda _kernels: []
    context_parallel_gdn._fla_build_cp_context = (
        lambda _cu_seqlens, group, conv1d_kernel_size: SimpleNamespace(group=group)
    )
    context_parallel_gdn._fla_causal_conv1d = fla_conv
    context_parallel_gdn._fla_chunk_kda = fla_kda


def _install_attention_patch() -> None:
    """The class-level DSA forward ``install_glm5_next_patches`` sets."""
    attention_cls: Any = Glm5NextTextAttention
    attention_cls.forward = causal_sdpa_attention_forward
    setattr(attention_cls, CAUSAL_SDPA_WHEN_TOPK_COVERS, True)


def _with_index_topk(stack: nn.Module, index_topk: int) -> nn.Module:
    for module in stack.modules():
        if isinstance(module, Glm5NextTextAttention):
            module.config.index_topk = index_topk
        elif isinstance(module, Glm5NextTextIndexer):
            module.index_topk = index_topk
    return stack


def _cp_data(
    dp_rank: int, left_pad: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mHC hidden streams, an output probe and a padding mask for one data-parallel rank."""
    generator = torch.Generator().manual_seed(400 + dp_rank)
    shape = (BATCH, SEQ, HC_MULT, HIDDEN)
    mask = torch.ones(BATCH, SEQ, dtype=torch.bool)
    # Left padding moves the first k-pool off position 0 of the whole sequence.
    mask[1, :left_pad] = False
    return (
        torch.randn(shape, generator=generator),
        torch.randn(shape, generator=generator),
        mask,
    )


def _full_loss(stack: nn.Module, dp_rank: int, left_pad: int) -> torch.Tensor:
    hidden, probe, mask = _cp_data(dp_rank, left_pad)
    return (stack(hidden, mask) * probe).sum()


def _assert_wrapped_for_context_parallel(stack: nn.Module, linear_heads: int) -> None:
    """Every KDA and DSA layer is wrapped, and CP leaves their heads whole."""
    for module in stack.modules():
        if isinstance(module, Glm5NextTextLinearAttention):
            assert module._agilerl_gdn_cp is True
            assert module.num_heads == linear_heads
        elif isinstance(module, Glm5NextTextAttention):
            assert module._agilerl_glm5_next_cp is True
            assert module.num_heads == 4


def _worker(
    rank: int,
    world_size: int,
    cp: int,
    tp: int,
    ep: int,
    index_topk: int,
    left_pad: int,
) -> None:
    """Sharded forward, reshard, LoRA gather and one SGD step match the unsharded stack."""
    # Arrange
    torch.manual_seed(0)
    _fake_fla_kernels()
    _install_attention_patch()
    # FSDP x TP splits A_log and b_proj rows over every rank.
    linear_heads = 4 * tp
    stack = _with_index_topk(
        _with_moe_lora(Glm5NextBlockStack(linear_heads)), index_topk
    )
    broadcast_module(stack)
    reference = copy.deepcopy(stack)
    stack.train()
    reference.train()
    _apply_liger_swaps(stack)
    dp = world_size // (cp * tp)
    mesh = parallel_mesh(world_size, cp, tp, ep)
    assert mesh.cp is not None
    # The FSDP mesh folds cp into dp_shard.
    assert mesh.dp_shard.size() == dp * cp
    config = fsdp_config(cp, tp, ep)
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(
            stack, "cpu", config, parallel_mesh=mesh, gradient_checkpointing=True
        )
    cp_rank = int(mesh.cp.get_local_rank())
    hidden, probe, mask = _cp_data(rank // (cp * tp), left_pad)
    # Reference first: the KDA kernels run the stock path until cu_seqlens is published.
    with torch.no_grad():
        full_output = reference(hidden, mask)
    reference_loss = sum(
        _full_loss(reference, dp_rank, left_pad) for dp_rank in range(dp)
    )
    (reference_loss / dp).backward()
    expected = _stepped_reference(reference)
    runtime = FSDPRuntime(config)
    runtime.parallel_mesh = mesh
    optimizer = sgd(stack, lr=1.0)

    # Act
    publish_varlen_offsets(torch.arange(BATCH + 1, dtype=torch.int32) * SEQ)
    try:
        output = stack(
            shard_for_cp(hidden, cp_rank, cp), shard_for_cp(mask, cp_rank, cp)
        )
        assert_resharded(stack, reference, tp, world_size // tp)
        _assert_wrapped_for_context_parallel(stack, linear_heads)
        if tp > 1:
            _assert_tp_blocks_sharded(stack, tp, TP_PROBE_WEIGHTS)
        if ep > 1:
            _assert_experts_on_ep(stack, ep)
        assert_lora(stack, reference)
        step = runtime.backward(
            (output * shard_for_cp(probe, cp_rank, cp)).sum(),
            optimizer,
            gradient_accumulation_steps=1,
            actor=stack,
            max_grad_norm=1e6,
        )
    finally:
        clear_ulysses_params()

    # Assert
    assert_exact(output, shard_for_cp(full_output, cp_rank, cp), "output")
    assert step is not None
    _assert_stepped(stack, expected, GLM5_NEXT_TENSOR_PARALLEL_PLAN.row_orders(stack))


def _token_batch(dp_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded token ids and their mask for one data-parallel rank."""
    generator = torch.Generator().manual_seed(500 + dp_rank)
    width = max(TOKEN_LENGTHS)
    ids = torch.randint(1, VOCAB, (BATCH, width), generator=generator)
    mask = torch.zeros(BATCH, width, dtype=torch.long)
    for row, length in enumerate(TOKEN_LENGTHS):
        mask[row, :length] = 1
    return ids.masked_fill(mask == 0, 0), mask


def _sequence_worker(rank: int, world_size: int) -> None:
    """ContextParallelSequence shards token ids into the stack; the gathered output matches the unsharded hidden states."""
    # Arrange
    cp = 2
    torch.manual_seed(0)
    _fake_fla_kernels()
    _install_attention_patch()
    stack = _with_moe_lora(Glm5NextBlockStack())
    broadcast_module(stack)
    reference = copy.deepcopy(stack)
    _apply_liger_swaps(stack)
    mesh = parallel_mesh(world_size, cp, 1, 1)
    assert mesh.cp is not None
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(
            stack,
            "cpu",
            fsdp_config(cp, 1, 1),
            parallel_mesh=mesh,
            gradient_checkpointing=True,
        )
    embedding = torch.randn(VOCAB, HIDDEN, generator=torch.Generator().manual_seed(7))

    def streams(ids: torch.Tensor) -> torch.Tensor:
        return embedding[ids].unsqueeze(2).expand(-1, -1, HC_MULT, -1).contiguous()

    ids, mask = _token_batch(rank // cp)
    layout = ContextParallelSequence(
        0, cp, mesh.cp.get_group(), int(mesh.cp.get_local_rank()), cp * CP_SCAN_CHUNK
    )
    aligned_ids, aligned_mask = pad_right_for_cp(ids, cp * CP_SCAN_CHUNK, 0, mask)
    action_len = aligned_ids.shape[1] - 1
    with torch.no_grad():
        expected = reference(
            streams(aligned_ids[:, :action_len]), aligned_mask[:, :action_len].bool()
        )

    # Act
    forward = layout.prepare_forward(ids, mask)
    with torch.no_grad(), forward:
        output = stack(
            streams(forward.kwargs["input_ids"]),
            forward.kwargs["attention_mask"].bool(),
        )
        gathered = gather_sequence_for_cp(output, mesh.cp.get_group())[
            :, : ids.shape[1]
        ]

    # Assert
    assert action_len == cp * CP_SCAN_CHUNK
    assert_resharded(stack, reference, 1, world_size)
    assert_exact(gathered, expected[:, : ids.shape[1]], "gathered hidden")


def _new_padding_worker(rank: int, world_size: int) -> None:
    """A second sharded forward with new padding matches the unsharded stack."""
    # Arrange
    cp = 2
    torch.manual_seed(0)
    _fake_fla_kernels()
    _install_attention_patch()
    stack = _with_index_topk(_with_moe_lora(Glm5NextBlockStack()), COVERING_TOPK)
    broadcast_module(stack)
    reference = copy.deepcopy(stack)
    _apply_liger_swaps(stack)
    mesh = parallel_mesh(world_size, cp, 1, 1)
    assert mesh.cp is not None
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        materialize_fsdp2_from_cpu_state(
            stack,
            "cpu",
            fsdp_config(cp, 1, 1),
            parallel_mesh=mesh,
            gradient_checkpointing=True,
        )
    cp_rank = int(mesh.cp.get_local_rank())
    outputs = []
    expected = []

    # Act
    # No padding takes the causal shortcut; left padding then needs the indexer.
    for left_pad in (0, 3):
        hidden, _, mask = _cp_data(rank // cp, left_pad)
        with torch.no_grad():
            expected.append(shard_for_cp(reference(hidden, mask), cp_rank, cp))
            publish_varlen_offsets(torch.arange(BATCH + 1, dtype=torch.int32) * SEQ)
            try:
                outputs.append(
                    stack(
                        shard_for_cp(hidden, cp_rank, cp),
                        shard_for_cp(mask, cp_rank, cp),
                    )
                )
            finally:
                clear_ulysses_params()

    # Assert
    assert_exact(outputs[0], expected[0], "unpadded output")
    assert_exact(outputs[1], expected[1], "padded output")


class RecordingAttention(nn.Module):
    """Attention stand-in whose forward returns the kwargs it was called with."""

    def forward(self, *_args: object, **kwargs: object) -> dict[str, object]:
        return kwargs


setattr(RecordingAttention, CAUSAL_SDPA_WHEN_TOPK_COVERS, True)


def _shard() -> SequenceShard:
    return SequenceShard(MagicMock())


class TestInstallGlm5NextAttentionContextParallel:
    def test_passes_the_shard_only_while_cu_seqlens_is_published(self) -> None:
        # Arrange
        attention = RecordingAttention()
        shard = _shard()

        # Act
        wrapped = install_glm5_next_attention_context_parallel(attention, shard)
        rollout_kwargs = attention(torch.zeros(1))
        publish_varlen_offsets(torch.tensor([0, 8], dtype=torch.int32))
        try:
            training_kwargs = attention(torch.zeros(1))
        finally:
            clear_ulysses_params()

        # Assert
        assert wrapped is True
        assert rollout_kwargs == {}
        assert training_kwargs == {SEQUENCE_SHARD: shard}

    def test_second_install_leaves_the_wrap(self) -> None:
        attention = RecordingAttention()
        install_glm5_next_attention_context_parallel(attention, _shard())

        assert (
            install_glm5_next_attention_context_parallel(attention, _shard()) is False
        )

    def test_unpatched_attention_class_raises(self) -> None:
        unpatched = type("Glm5NextTextAttention", (nn.Module,), {})

        with pytest.raises(RuntimeError, match="needs install_glm5_next_patches"):
            install_glm5_next_attention_context_parallel(unpatched(), _shard())


class TestGlm5NextContextParallel:
    @pytest.mark.multi_rank
    @pytest.mark.parametrize(
        ("cp", "tp", "ep", "world"),
        [(2, 1, 1, 4), (2, 1, 2, 4), (2, 2, 2, 8)],
        ids=["cp", "cp_ep", "cp_tp_ep"],
    )
    def test_forward_and_step_match_unsharded(
        self, cp: int, tp: int, ep: int, world: int
    ) -> None:
        run_ranks(_worker, world, cp, tp, ep, INDEXED_TOPK, 3, timeout=300)

    @pytest.mark.multi_rank
    def test_covering_topk_matches_unsharded(self) -> None:
        run_ranks(_worker, 4, 2, 1, 1, COVERING_TOPK, 0, timeout=300)

    @pytest.mark.multi_rank
    def test_sequence_layout_gathered_output_matches_unsharded(self) -> None:
        run_ranks(_sequence_worker, 4, timeout=300)

    @pytest.mark.multi_rank
    def test_new_padding_in_a_later_forward_matches_unsharded(self) -> None:
        run_ranks(_new_padding_worker, 4, timeout=300)
