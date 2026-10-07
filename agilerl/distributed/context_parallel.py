# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel sequence sharding and training-tensor bundle.

``cp`` splits the sequence across ranks: every rank in a CP group holds the
same batch rows and one even ``1/cp`` slice of each sequence. Ulysses
attention (see ``agilerl.algorithms.core.llm_ops.ulysses_attn``) restores
exact full-sequence attention inside the forward, and per-rank log-probs are
gathered back to full sequences before the loss, so denominators and metrics
stay global. Everything here is inert at ``cp == 1``. The live mesh is
``build_parallel_mesh``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
import torch.distributed as dist

from agilerl.distributed.expert_parallel import validate_cp_degree

if TYPE_CHECKING:
    from agilerl.arena.models.fsdp import FSDPConfig


class GatherCpCtx(Protocol):
    """Autograd context for a context-parallel all-gather."""

    cp_rank: int
    shard_len: int


class GatherCpSumCtx(Protocol):
    """Autograd context for an all-gather whose backward sums partial grads."""

    cp_rank: int
    shard_len: int
    group: dist.ProcessGroup


CP_STYLE = "ulysses"
CP_STYLES = frozenset({CP_STYLE})
CP_SUPPORTED_ALGOS = ("GRPO", "SFT", "PPO", "REINFORCE", "DPO")


def reject_unsupported_cp(algo_name: str, cp: int) -> None:
    """Raise when ``cp > 1`` on a trainer that has no context-parallel forward.

    :param algo_name: Algorithm class name used in the error.
    :param cp: Requested context-parallel degree.
    """
    if cp <= 1:
        return
    names = ", ".join(CP_SUPPORTED_ALGOS[:-1]) + f", and {CP_SUPPORTED_ALGOS[-1]}"
    msg = (
        f"{algo_name} does not implement the context-parallel "
        f"forward; cp > 1 needs {names}."
    )
    raise ValueError(msg)


def validate_cp_style(cp_style: str) -> str:
    """Return the normalized CP style, rejecting anything but Ulysses."""
    if cp_style != CP_STYLE:
        msg = (
            f"cp_style={cp_style!r} is not supported: only 'ulysses' is "
            "supported. Ulysses redistributes sequence shards to head shards "
            "with two all-to-alls around local flash attention."
        )
        raise ValueError(msg)
    return cp_style


def cp_data_parallel_size(world_size: int, cp: int) -> int:
    """Data-parallel size after folding ``world_size`` by ``cp``."""
    validate_cp_degree(cp)
    if world_size < 1:
        msg = f"world_size must be >= 1, got {world_size}"
        raise ValueError(msg)
    if world_size % cp != 0:
        msg = (
            f"world_size ({world_size}) must be divisible by cp ({cp}) "
            "for context parallel."
        )
        raise ValueError(msg)
    return world_size // cp


def cp_data_parallel_rank(rank: int, cp: int) -> int:
    """Data-parallel index of ``rank``: ranks that differ only in ``cp`` share it."""
    validate_cp_degree(cp)
    if not isinstance(rank, int) or isinstance(rank, bool):
        msg = f"rank must be an int, got {type(rank).__name__}"
        raise TypeError(msg)
    if rank < 0:
        msg = f"rank must be >= 0, got {rank}"
        raise ValueError(msg)
    return rank // cp


def validate_cp_ep_mix(ep: int, cp: int, world_size: int | None = None) -> None:
    """Require a legal expert carve-out of the shard group under CP.

    When ``ep > 1``: ``ep % cp == 0`` and, if given, ``world_size % ep == 0``.
    """
    validate_cp_degree(cp)
    if not isinstance(ep, int) or isinstance(ep, bool):
        msg = f"ep must be an int, got {type(ep).__name__}"
        raise TypeError(msg)
    if ep < 1:
        msg = f"ep must be >= 1, got {ep}"
        raise ValueError(msg)
    if ep <= 1:
        return
    if ep % cp != 0:
        msg = (
            f"ep={ep} is not divisible by cp={cp}: the expert degree must "
            "split evenly over the context-parallel group."
        )
        raise ValueError(msg)
    if world_size is None:
        return
    if world_size % ep != 0:
        msg = (
            f"world_size ({world_size}) must be divisible by ep ({ep}): "
            "the shard group times replicate must split evenly over ep."
        )
        raise ValueError(msg)


def validate_cp_heads(
    num_attention_heads: int,
    num_key_value_heads: int | None,
    cp: int,
) -> None:
    """Validate Ulysses head sharding for ``cp``.

    Query heads split evenly; key/value heads either split evenly or replicate
    (when fewer than ``cp``, each rank repeats them so every query slice finds
    its KV head).
    """
    validate_cp_degree(cp)
    if cp <= 1:
        return
    if num_attention_heads % cp != 0:
        msg = (
            f"Ulysses CP requires num_attention_heads ({num_attention_heads}) "
            f"divisible by cp ({cp})"
        )
        raise ValueError(msg)
    kv_heads = (
        num_key_value_heads if num_key_value_heads is not None else num_attention_heads
    )
    if kv_heads % cp != 0 and cp % kv_heads != 0:
        msg = (
            f"Ulysses GQA requires num_key_value_heads ({kv_heads}) divisible "
            f"by cp ({cp}) or cp divisible by num_key_value_heads "
            "(KV replicate path); got neither"
        )
        raise ValueError(msg)


def validate_cp_seq_len(seq_len: int, cp: int) -> None:
    """Fail fast when a sequence length cannot split evenly over ``cp``.

    Uneven shards hang CP collectives, so reject them.
    """
    validate_cp_degree(cp)
    if cp <= 1:
        return
    if seq_len % cp != 0:
        msg = (
            f"sequence length ({seq_len}) must be divisible by cp ({cp}): "
            "uneven shards deadlock CP collectives. Pad completions so the "
            "action-frame length is a multiple of cp."
        )
        raise ValueError(msg)


def validate_cp_config(
    *,
    cp: int,
    cp_style: str,
    fsdp_config: FSDPConfig | None,
    world_size: int,
    ep: int = 1,
    use_liger_loss: bool = False,
    liger_cp_level: str | None = None,
    use_sequence_packing: bool = False,
    attn_implementation: str | None = None,
    num_attention_heads: int | None = None,
    num_key_value_heads: int | None = None,
) -> str:
    """Fail fast on illegal CP configurations before collective work.

    :return: The normalized ``cp_style`` (``"ulysses"`` when ``cp == 1``).
    """
    validate_cp_degree(cp)
    if cp == 1:
        return CP_STYLE
    style = validate_cp_style(cp_style)
    if fsdp_config is None:
        msg = (
            f"cp={cp} requires fsdp_config: context parallel without FSDP "
            "keeps a full weight replica on every rank."
        )
        raise ValueError(msg)
    cp_data_parallel_size(world_size, cp)
    validate_cp_ep_mix(ep, cp, world_size)
    if use_liger_loss and liger_cp_level not in {"token", "turn", "trajectory"}:
        msg = (
            f"cp={cp} with use_liger_loss=True is supported only at the "
            f"token/turn/trajectory importance-sampling levels (got "
            f"{liger_cp_level!r}): the fused kernel runs sharded per rank at "
            "token level, while pooled levels gather fused per-token scalars "
            "and pool on the full sequence. Pass liger_cp_level='token' for "
            "a token-level GRPO run."
        )
        raise ValueError(msg)
    if attn_implementation is not None and attn_implementation != "flash_attention_2":
        msg = (
            f"cp={cp} requires attn_implementation='flash_attention_2', got "
            f"{attn_implementation!r}: the Ulysses substitution patches the "
            "flash-attention-2 forward. Unset defers to the live-model check "
            "at wrap time."
        )
        raise ValueError(msg)
    if num_attention_heads is not None:
        validate_cp_heads(num_attention_heads, num_key_value_heads, cp)
    return style


def shard_for_cp(
    data: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    seq_dim: int = 1,
) -> torch.Tensor:
    """Return this rank's even ``1/cp`` slice of ``data`` along ``seq_dim``."""
    validate_cp_degree(cp_size)
    if cp_size <= 1:
        return data
    length = data.shape[seq_dim]
    if length % cp_size != 0:
        msg = (
            f"CP requires sequence dimension {seq_dim} ({length}) divisible by "
            f"cp size ({cp_size}): uneven shards deadlock CP collectives "
            "(e.g. Ulysses all-to-all)."
        )
        raise ValueError(msg)
    return torch.chunk(data, cp_size, dim=seq_dim)[cp_rank]


class GatherForCp(torch.autograd.Function):
    """All-gather on dim 1; backward keeps this rank's slice and drops the rest."""

    @staticmethod
    def forward(
        ctx: GatherCpCtx,
        data: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        cp_size = dist.get_world_size(group)
        ctx.cp_rank = dist.get_rank(group)
        ctx.shard_len = data.shape[1]
        gathered = [torch.empty_like(data) for _ in range(cp_size)]
        dist.all_gather(gathered, data.contiguous(), group=group)
        return torch.cat(gathered, dim=1)

    @staticmethod
    def backward(
        ctx: GatherCpCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (grad_output,) = grad_outputs
        start = ctx.cp_rank * ctx.shard_len
        shard_grad = grad_output.narrow(1, start, ctx.shard_len).contiguous()
        return shard_grad, None


def gather_for_cp(data: torch.Tensor, cp_group: dist.ProcessGroup) -> torch.Tensor:
    """Differentiable all-gather of per-rank slices, concatenated on dim 1.

    Backward keeps this rank's slice and drops the rest.
    """
    return GatherForCp.apply(data, cp_group)


def gather_for_cp_wo_grad(
    data: torch.Tensor, cp_size: int, cp_group: dist.ProcessGroup
) -> torch.Tensor:
    """Non-differentiable all-gather of per-rank slices, concatenated on dim 1."""
    gathered = [torch.empty_like(data) for _ in range(cp_size)]
    dist.all_gather(gathered, data, group=cp_group)
    return torch.cat(gathered, dim=1)


class GatherForCpSumGrad(torch.autograd.Function):
    """All-gather on dim 1. Backward sums every rank's partial grad, then slices."""

    @staticmethod
    def forward(
        ctx: GatherCpSumCtx,
        data: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        cp_size = dist.get_world_size(group)
        ctx.cp_rank = dist.get_rank(group)
        ctx.shard_len = data.shape[1]
        ctx.group = group
        gathered = [torch.empty_like(data) for _ in range(cp_size)]
        dist.all_gather(gathered, data.contiguous(), group=group)
        return torch.cat(gathered, dim=1)

    @staticmethod
    def backward(
        ctx: GatherCpSumCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (grad_output,) = grad_outputs
        # Each rank's loss touches a different dense re-shard of this row, so
        # the packed owner needs the sum before the local slice is kept.
        grad = grad_output.detach().contiguous().clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)
        start = ctx.cp_rank * ctx.shard_len
        shard_grad = grad.narrow(1, start, ctx.shard_len).contiguous()
        return shard_grad, None


def gather_for_cp_sum_grad(
    data: torch.Tensor, cp_group: dist.ProcessGroup
) -> torch.Tensor:
    """All-gather on dim 1. Backward sums partial grads, then keeps this rank's slice."""
    return GatherForCpSumGrad.apply(data, cp_group)


@dataclass
class TrainingShardBundle:
    """One rank's sequence shards of the tensors a loss step consumes.

    The cut is computed once on the action frame (``N = T - 1``). The shift
    from token ids to next-token labels happens before the cut: query ids
    drop the last token (labels-only) and label ids drop the first, so
    each rank's label shard already holds its boundary targets.
    """

    query_ids: torch.Tensor
    label_ids: torch.Tensor
    mask: torch.Tensor
    advantages: torch.Tensor
    old_log_probs: torch.Tensor
    reference_log_probs: torch.Tensor
    turn_ids: torch.Tensor | None
    sampling_ratios: torch.Tensor | None


def shard_training_bundle(
    *,
    token_ids: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    turn_ids: torch.Tensor | None,
    sampling_ratios: torch.Tensor | None,
    cp_rank: int,
    cp_size: int,
    seq_dim: int = 1,
) -> TrainingShardBundle:
    """Cut the training tensors at one shared boundary.

    :param token_ids: ``(B, T)`` full-sequence token ids; the action frame is
        the last ``T - 1`` columns.
    :param mask: ``(B, N)`` action-token mask on the action frame.
    :param advantages: ``(B, N)`` or ``(B, 1)`` advantages; the ``(B, 1)``
        per-trajectory form broadcasts and passes through unsharded.
    :param old_log_probs: ``(B, N)`` old-policy log-probs.
    :param reference_log_probs: ``(B, N)`` reference-policy log-probs.
    :param turn_ids: ``(B, N)`` turn index per action token, or ``None``.
    :param sampling_ratios: ``(B, N)`` sampling-correction ratios, or ``None``.
    :param seq_dim: Sequence dimension shared by every framed tensor.
    :return: This rank's ``(B, N/cp)`` shards as a bundle.
    :raises ValueError: If the action frame does not split evenly, or any
        framed tensor disagrees on its length.
    """
    validate_cp_degree(cp_size)
    seq_len = token_ids.shape[seq_dim]
    action_len = seq_len - 1
    query_ids = token_ids.narrow(seq_dim, 0, action_len)
    label_ids = token_ids.narrow(seq_dim, 1, action_len)
    if cp_size <= 1:
        return TrainingShardBundle(
            query_ids=query_ids,
            label_ids=label_ids,
            mask=mask,
            advantages=advantages,
            old_log_probs=old_log_probs,
            reference_log_probs=reference_log_probs,
            turn_ids=turn_ids,
            sampling_ratios=sampling_ratios,
        )
    validate_cp_seq_len(action_len, cp_size)
    framed: dict[str, torch.Tensor | None] = {
        "mask": mask,
        "advantages": advantages,
        "old_log_probs": old_log_probs,
        "reference_log_probs": reference_log_probs,
        "turn_ids": turn_ids,
        "sampling_ratios": sampling_ratios,
    }
    for name, tensor in framed.items():
        if tensor is None:
            continue
        if tensor.shape[seq_dim] == 1:
            continue
        if tensor.shape[seq_dim] != action_len:
            msg = (
                f"CP shard boundary disagreement: '{name}' has sequence "
                f"length {tensor.shape[seq_dim]} on dim {seq_dim}, expected "
                f"the action-frame length {action_len}. All training tensors "
                "must share one cut."
            )
            raise ValueError(msg)
    width = action_len // cp_size

    def _cut(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[seq_dim] == 1:
            return tensor
        return tensor.narrow(seq_dim, cp_rank * width, width).contiguous()

    return TrainingShardBundle(
        query_ids=query_ids.narrow(seq_dim, cp_rank * width, width).contiguous(),
        label_ids=label_ids.narrow(seq_dim, cp_rank * width, width).contiguous(),
        mask=_cut(mask),
        advantages=_cut(advantages),
        old_log_probs=_cut(old_log_probs),
        reference_log_probs=_cut(reference_log_probs),
        turn_ids=None if turn_ids is None else _cut(turn_ids),
        sampling_ratios=None if sampling_ratios is None else _cut(sampling_ratios),
    )
