# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel sequence split, Ulysses attention, and mixer all-to-all.

``cp > 1`` shards the sequence: ranks in a CP group hold the same batch rows
and one even ``1/cp`` slice of each sequence. Pad, shard, and gather live in
:mod:`agilerl.distributed.context_parallel.sequence`. Dense attention
all-to-alls to head shards in
:mod:`agilerl.distributed.context_parallel.ulysses`. Mamba all-to-alls
projected states to ``[B, L, D/cp]`` in
:mod:`agilerl.distributed.context_parallel.mamba`. Gated DeltaNet keeps the
sequence shard and passes recurrent state through flash-linear-attention in
:mod:`agilerl.distributed.context_parallel.gdn`.
:mod:`agilerl.distributed.context_parallel.mixer` runs both through the family
plans. Mesh views live in :mod:`agilerl.distributed.mesh`.
"""

from agilerl.distributed.context_parallel.mamba import MixerInProjBypass
from agilerl.distributed.context_parallel.mixer import apply_mixer_context_parallel
from agilerl.distributed.context_parallel.sequence import (
    all_to_all_hidden_to_seq,
    all_to_all_seq_to_hidden,
    gather_for_cp,
    gather_for_cp_wo_grad,
    gather_sequence_for_cp,
    pad_right_for_cp,
    shard_for_cp,
    validate_cp_degree,
    validate_cp_heads,
    validate_cp_seq_len,
    validate_cp_style,
)
from agilerl.distributed.context_parallel.ulysses import (
    ULYSSES_PARAMS,
    PublishedUlyssesParams,
    attention_head_counts,
    clear_ulysses_params,
    model_attention_backend,
    replicate_kv_heads,
    substitute_hf_ulysses_attn,
    ulysses_checkpoint_context_fn,
    update_ulysses_params,
)

__all__ = [
    "ULYSSES_PARAMS",
    "MixerInProjBypass",
    "PublishedUlyssesParams",
    "all_to_all_hidden_to_seq",
    "all_to_all_seq_to_hidden",
    "apply_mixer_context_parallel",
    "attention_head_counts",
    "clear_ulysses_params",
    "gather_for_cp",
    "gather_for_cp_wo_grad",
    "gather_sequence_for_cp",
    "model_attention_backend",
    "pad_right_for_cp",
    "replicate_kv_heads",
    "shard_for_cp",
    "substitute_hf_ulysses_attn",
    "ulysses_checkpoint_context_fn",
    "update_ulysses_params",
    "validate_cp_degree",
    "validate_cp_heads",
    "validate_cp_seq_len",
    "validate_cp_style",
]
