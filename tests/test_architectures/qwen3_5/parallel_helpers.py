# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Inputs, forwards and kernel stand-ins shared by the Qwen3.5 parallel tests."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.models.qwen3_5 import modeling_qwen3_5

from agilerl.algorithms.core.sequence_layout import ContextParallelSequence
from agilerl.distributed.context_parallel import gdn as context_parallel_gdn
from agilerl.distributed.context_parallel.sequence import (
    CP_SCAN_CHUNK,
    pad_right_for_cp,
    shard_for_cp,
)
from tests.parallel_helpers import assert_exact, broadcast_tensor, unwrap

HIDDEN = 16
HEAD_DIM = 4
BATCH = 2
SEQ = 4
# The fp32 GDN decay gradient is ill-conditioned: under tp x cp the unsharded
# fp32 reference is itself ~1e-4 relative off float64.
DECAY_PARAMS = (
    "linear_attn.dt_bias",
    "linear_attn.A_log",
    "linear_attn.in_proj_a.base_layer.weight",
    "linear_attn.in_proj_a.lora_B.default.weight",
)


def rope(batch: int, seq: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ones(batch, seq, HEAD_DIM), torch.zeros(batch, seq, HEAD_DIM)


def forward_blocks(
    stack: nn.Module,
    gdn_hidden: torch.Tensor,
    attn_hidden: torch.Tensor,
    vision_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gdn_cos, gdn_sin = rope(gdn_hidden.shape[0], gdn_hidden.shape[1])
    attn_cos, attn_sin = rope(attn_hidden.shape[0], attn_hidden.shape[1])
    vision_cos = torch.ones(vision_hidden.shape[0], HEAD_DIM)
    vision_sin = torch.zeros(vision_hidden.shape[0], HEAD_DIM)
    cu_seqlens = torch.tensor([0, vision_hidden.shape[0]], dtype=torch.int32)
    gdn_out = stack.layers[0](gdn_hidden, (gdn_cos, gdn_sin))
    attn_out = stack.layers[1](attn_hidden, (attn_cos, attn_sin))
    vision_out = stack.vision(
        vision_hidden,
        cu_seqlens=cu_seqlens,
        position_embeddings=(vision_cos, vision_sin),
    )
    merger_out = stack.merger(vision_hidden)
    return gdn_out, attn_out, vision_out, merger_out


def assert_sequence_roundtrip(
    model: nn.Module, reference: nn.Module, cp: int, mesh: Any
) -> None:
    group = mesh.cp.get_group()
    rank = int(mesh.cp.get_local_rank())
    layout = ContextParallelSequence(
        pad_token_id=0,
        cp=cp,
        group=group,
        rank=rank,
        round_to=cp * CP_SCAN_CHUNK,
    )
    ids = torch.randint(1, 32, (BATCH, 5))
    mask = torch.ones_like(ids)
    broadcast_tensor(ids)
    table = broadcast_tensor(torch.randn(32, HIDDEN))
    aligned, _aligned_mask = pad_right_for_cp(
        ids, cp * CP_SCAN_CHUNK, layout.pad_token_id, mask
    )
    assert aligned.shape[1] - 1 == cp * CP_SCAN_CHUNK
    block = model.layers[1]
    mlp = unwrap(block).mlp
    reference_mlp = reference.layers[1].mlp
    block.unshard()
    with layout.prepare_forward(ids, mask) as forward:
        local = table[forward.kwargs["input_ids"]]
        restored = forward.scoring_hidden(mlp(local))
    block.reshard()
    full = table[aligned[:, :-1]]
    expected = reference_mlp(full)
    expected = shard_for_cp(expected, layout.rank, cp)
    assert_exact(restored, expected, "cp sequence")


def fake_fla_kernels() -> None:
    """Swap FLA's GPU-only CP kernels for transformers' torch conv and scan.

    The torch kernels scan each shard from zero state, as the reference does on
    the same shard, so the comparison checks TP x CP sharding, not GDN state passing.
    """
    conv = modeling_qwen3_5.causal_conv1d_fn
    scan = modeling_qwen3_5.torch_chunk_gated_delta_rule

    def fla_conv(
        x: torch.Tensor,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        activation: str | None,
        _cp_context: object,
    ) -> tuple[torch.Tensor, None]:
        out = conv(x.transpose(1, 2), weight, bias, activation=activation)
        return out.transpose(1, 2), None

    def fla_scan(*args: object, **kwargs: object) -> object:
        kwargs.pop("cp_context")
        return scan(*args, **kwargs)

    context_parallel_gdn.gdn_kernel_import_errors = lambda _kernels: []
    context_parallel_gdn._fla_build_cp_context = lambda *_args, **_kwargs: None
    context_parallel_gdn._fla_causal_conv1d = fla_conv
    context_parallel_gdn._fla_chunk_gated_delta_rule = fla_scan


def shard_data(
    dp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """GDN, attention and vision inputs plus output probes for one data shard."""
    generator = torch.Generator().manual_seed(100 + dp_rank)

    def draw(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    gdn = draw(BATCH, SEQ, HIDDEN)
    attn = draw(BATCH, SEQ, HIDDEN)
    vision = draw(SEQ, HIDDEN)
    probes = (
        draw(BATCH, SEQ, HIDDEN),
        draw(BATCH, SEQ, HIDDEN),
        draw(SEQ, HIDDEN),
        draw(SEQ, HIDDEN),
    )
    return gdn, attn, vision, probes


def cp_rank_loss(stack: nn.Module, dp_rank: int, cp_rank: int, cp: int) -> torch.Tensor:
    """One CP rank's GDN slice loss plus 1/cp of the blocks CP does not split."""
    gdn, attn, vision, probes = shard_data(dp_rank)
    gdn_probe, *other_probes = probes
    gdn_out, *others = forward_blocks(
        stack, shard_for_cp(gdn, cp_rank, cp), attn, vision
    )
    loss = (gdn_out * shard_for_cp(gdn_probe, cp_rank, cp)).sum()
    for output, probe in zip(others, other_probes, strict=True):
        loss = loss + (output * probe).sum() / cp
    return loss
