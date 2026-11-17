# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 DSA attention: causal SDPA without the indexer when top-k keeps every visible key.

Without a cache or padding, the indexer scores the ``seq_len // index_kpool``
complete k-pools, keeps those ending at or before each query, and appends the
query's incomplete tail pool. When ``index_topk // index_kpool`` covers every
pool, that selection is exactly ``key <= query`` whatever the indexer scores,
so the dense boolean mask equals the causal mask SDPA applies with
``is_causal=True``. The indexer runs under ``no_grad`` and returns integer
indices, so its parameters get no gradient either way.

The condition depends only on the text config, the sequence length, the
padding mask, and the cache, all shared by every layer of one forward. The
model forward checks it once and hands the decoder layers a None mask; full
and shared DSA layers then both skip top-k.

Context parallel: queries stay on the rank's sequence shard. The compressed
latent KV is all-gathered over the context-parallel group and expanded for the
whole sequence. The first DSA layer of each forward gathers the padding mask
and checks top-k coverage on the whole sequence for every layer; the indexer
scores the shard's queries against every rank's packed key states at their
global causal positions.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed as dist

from agilerl import HAS_LIGER_KERNEL, HAS_LLM_DEPENDENCIES
from agilerl.architectures.hyper_connections import compile_hyper_connection_mix
from agilerl.architectures.runtime import PatchRuntimeConfig
from agilerl.utils.patching import class_is_patched, library_shim

if HAS_LLM_DEPENDENCIES or TYPE_CHECKING:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextTextAttention,
        Glm5NextTextHyperConnection,
        Glm5NextTextModel,
        eager_attention_forward,
    )
else:
    ALL_ATTENTION_FUNCTIONS: Any = None
    Glm5NextTextAttention: Any = None
    Glm5NextTextHyperConnection: Any = None
    Glm5NextTextModel: Any = None
    eager_attention_forward: Any = None

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedModel
    from transformers.cache_utils import Cache
    from transformers.modeling_outputs import MoeModelOutputWithPast
    from transformers.models.glm5_next.configuration_glm5_next import (
        Glm5NextTextConfig,
    )

__all__ = [
    "CAUSAL_SDPA_WHEN_TOPK_COVERS",
    "SEQUENCE_SHARD",
    "SequenceShard",
    "SequenceShardIndexerCache",
    "causal_sdpa_attention_forward",
    "expand_latent_kv",
    "install_glm5_next_patches",
    "sequence_shard_sparse_mask",
    "topk_attention_mask",
    "topk_covers_causal_prefix",
    "topk_skipping_model_forward",
]

CAUSAL_SDPA_WHEN_TOPK_COVERS = "agilerl_causal_sdpa_when_topk_covers"

# Attention kwarg carrying the SequenceShard while a sequence shard trains.
SEQUENCE_SHARD = "agilerl_sequence_shard"


def topk_covers_causal_prefix(
    config: Glm5NextTextConfig,
    seq_length: int,
    attention_mask: torch.Tensor | None,
) -> bool:
    """Whether the SDPA mask built from DSA top-k is exactly the causal mask, for a forward without a cache.

    :param config: GLM-5.3 text config.
    :param seq_length: Query length of this forward.
    :param attention_mask: Padding mask passed to the model, or None.
    :return: True when SDPA with ``is_causal=True`` and no mask gives the same attention.
    """
    if config._attn_implementation != "sdpa":
        return False
    kpool = config.index_kpool
    # Without the tail, a query whose prefix ends mid-pool cannot see its own tail tokens.
    if kpool > 1 and not config.index_kpool_always_select_tail:
        return False
    if seq_length // kpool > config.index_topk // kpool:
        return False
    return attention_mask is None or bool(attention_mask.all())


def topk_skipping_model_forward(forward: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``Glm5NextTextModel.forward`` to pass None masks when :func:`topk_covers_causal_prefix` holds.

    :param forward: Stock ``Glm5NextTextModel.forward``.
    :return: Forward that checks coverage once and otherwise defers to ``forward``.
    """

    @functools.wraps(forward)
    def model_forward(
        self: Glm5NextTextModel,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict[str, Any] | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> MoeModelOutputWithPast:
        caching = self.config.use_cache if use_cache is None else use_cache
        inputs = input_ids if inputs_embeds is None else inputs_embeds
        if (
            inputs is not None
            and not isinstance(attention_mask, dict)
            and past_key_values is None
            and not caching
            and topk_covers_causal_prefix(self.config, inputs.shape[1], attention_mask)
        ):
            # Linear attention only multiplies by the padding mask: the identity when all True.
            attention_mask = {"indexed_attention": None, "linear_attention": None}
        return forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

    return model_forward


class SequenceShardIndexerCache:
    """Cache stand-in that hands the DSA indexer every shard's packed key states."""

    def __init__(
        self, layer_idx: int, group: dist.ProcessGroup, shard_length: int
    ) -> None:
        self.layers = {layer_idx: self}
        self.group = group
        self.shard_length = shard_length
        self.keys = torch.empty(0)

    def update_indexer(
        self, packed_states: torch.Tensor, _layer_idx: int
    ) -> torch.Tensor:
        """Return the whole sequence's packed key states."""
        # Import cycle: agilerl.distributed imports agilerl.architectures, whose catalog imports this module.
        from agilerl.distributed.context_parallel.sequence import gather_for_cp_wo_grad

        self.keys = gather_for_cp_wo_grad(
            packed_states, dist.get_world_size(self.group), self.group
        )
        return self.keys

    def get_seq_length(self) -> int:
        """Key count up to and including this shard's last query."""
        return (dist.get_rank(self.group) + 1) * self.shard_length


class SequenceShard:
    """Context-parallel group of a model's DSA layers, with the running forward's whole-sequence top-k coverage.

    :param group: Context-parallel process group.
    """

    def __init__(self, group: dist.ProcessGroup) -> None:
        self.group = group
        self.cu_seqlens: torch.Tensor | None = None
        # Whole-sequence length and top-k coverage, or None before the forward's first DSA layer.
        self.coverage: tuple[int, bool] | None = None
        # Causal mask of the shard's queries when top-k covers, or None before its first use.
        self.causal_mask: torch.Tensor | None = None

    def begin_forward(self, cu_seqlens: torch.Tensor) -> None:
        """Drop the coverage when ``cu_seqlens`` belongs to a new forward.

        Every rank publishes fresh offsets for each sequence-shard forward and
        checkpoint recompute republishes the forward's own, so the mask
        all-gather that refills the coverage runs on every rank together.

        :param cu_seqlens: Varlen offsets published for the running forward.
        """
        if cu_seqlens is not self.cu_seqlens:
            self.cu_seqlens = cu_seqlens
            self.coverage = None
            self.causal_mask = None

    def whole_sequence_coverage(
        self, config: Glm5NextTextConfig, attention_mask: torch.Tensor
    ) -> tuple[int, bool]:
        """Whole-sequence length and :func:`topk_covers_causal_prefix`, gathered once per forward.

        :param config: GLM-5.3 text config shared by the model's DSA layers.
        :param attention_mask: The shard's ``[batch, shard_len]`` boolean padding mask.
        :return: Sequence length over every shard, and whether top-k keeps every causal key.
        """
        if self.coverage is None:
            # Import cycle: agilerl.distributed imports agilerl.architectures, whose catalog imports this module.
            from agilerl.distributed.context_parallel.sequence import (
                gather_for_cp_wo_grad,
            )

            # Gloo cannot all-gather bool tensors.
            full_mask = gather_for_cp_wo_grad(
                attention_mask.to(torch.int32),
                dist.get_world_size(self.group),
                self.group,
            ).bool()
            seq_length = full_mask.shape[1]
            self.coverage = (
                seq_length,
                topk_covers_causal_prefix(config, seq_length, full_mask),
            )
        return self.coverage

    def shard_causal_mask(
        self, shard_length: int, seq_length: int, device: torch.device
    ) -> torch.Tensor:
        """Boolean ``[1, 1, shard_len, seq_len]`` mask of ``key <= query`` for this rank's queries, built once per forward.

        :param shard_length: Query count on each rank.
        :param seq_length: Key count over every shard.
        :param device: Device of the shard's hidden states.
        :return: Causal mask at the shard's global query positions.
        """
        if self.causal_mask is None:
            query_positions = dist.get_rank(self.group) * shard_length + torch.arange(
                shard_length, device=device
            )
            keys = torch.arange(seq_length, device=device)
            self.causal_mask = (keys <= query_positions[:, None])[None, None]
        return self.causal_mask


def expand_latent_kv(
    attention: Glm5NextTextAttention, kv_nope: torch.Tensor, k_rot: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``expand_kv`` whose key is a view of the ``kv_b_proj`` output when there is no rope part.

    :param attention: DSA attention layer.
    :param kv_nope: Normalised latent KV ``[batch, 1, seq_len, kv_lora_rank]``.
    :param k_rot: Shared rope key ``[batch, 1, seq_len, qk_rope_head_dim]``.
    :return: Key and value states ``[batch, heads, seq_len, head_dim]``.
    """
    if attention.qk_rope_head_dim:
        return attention.expand_kv(kv_nope, k_rot)
    batch_size, _, seq_length, _ = kv_nope.shape
    kv_shape = (
        batch_size,
        seq_length,
        -1,
        attention.qk_nope_head_dim + attention.v_head_dim,
    )
    key_value = attention.kv_b_proj(kv_nope).view(kv_shape).transpose(1, 2)
    key_states, value_states = torch.split(
        key_value, [attention.qk_nope_head_dim, attention.v_head_dim], dim=-1
    )
    return key_states, value_states


def topk_attention_mask(
    attention: Glm5NextTextAttention,
    topk_indices: torch.Tensor,
    query_states: torch.Tensor,
    kv_length: int,
) -> torch.Tensor:
    """``build_attention_mask_from_topk`` scattered into a bool tensor.

    :param attention: DSA attention layer.
    :param topk_indices: ``[batch, q_len, topk]`` selected keys; -1 or ``>= kv_length`` selects none.
    :param query_states: Query states; their dtype sets the eager additive mask.
    :param kv_length: Key count.
    :return: ``[batch, 1, q_len, kv_length]`` bool mask for SDPA, additive mask for eager.
    """
    valid = topk_indices.ge(0) & topk_indices.lt(kv_length)
    # Invalid indices land in one extra column, dropped below.
    indices = torch.where(valid, topk_indices, kv_length).long()
    selected = torch.zeros(
        *topk_indices.shape[:2],
        kv_length + 1,
        dtype=torch.bool,
        device=topk_indices.device,
    )
    selected.scatter_(-1, indices, True)
    mask = selected[..., :kv_length].unsqueeze(1)
    if attention.config._attn_implementation == "sdpa":
        return mask
    zero = torch.zeros((), device=query_states.device, dtype=query_states.dtype)
    return torch.where(mask, zero, torch.finfo(query_states.dtype).min)


def sequence_shard_sparse_mask(
    attention: Glm5NextTextAttention,
    shard: SequenceShard,
    hidden_states: torch.Tensor,
    q_resid: torch.Tensor,
    query_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    prev_topk_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """SDPA mask of one sequence shard's queries over the whole sequence.

    :param attention: DSA attention layer.
    :param shard: Sequence shard of the running forward.
    :param hidden_states: The shard's hidden states ``[batch, shard_len, hidden_size]``.
    :param q_resid: The shard's ``q_a_layernorm(q_a_proj(hidden_states))``.
    :param query_states: The shard's query states.
    :param attention_mask: The shard's ``[batch, shard_len]`` padding mask, or
        None when every shard token is real.
    :param prev_topk_indices: Top-k indices from the previous full indexer layer.
    :return: Boolean mask ``[batch or 1, 1, shard_len, seq_len]`` and the
        top-k indices behind it, or None when top-k keeps every causal key.
    """
    batch_size, shard_length = hidden_states.shape[:2]
    device = hidden_states.device
    if attention_mask is None:
        attention_mask = torch.ones(
            batch_size, shard_length, dtype=torch.bool, device=device
        )
    seq_length, topk_covers = shard.whole_sequence_coverage(
        attention.config, attention_mask
    )
    # Shard queries sit at a rank offset, so SDPA's is_causal would align them to key 0.
    if topk_covers:
        return shard.shard_causal_mask(shard_length, seq_length, device), None
    if attention.indexer is not None:
        cache = SequenceShardIndexerCache(
            attention.layer_idx, shard.group, shard_length
        )
        topk_indices = attention.indexer(
            hidden_states=hidden_states,
            q_resid=q_resid,
            attention_mask=attention_mask,
            past_key_values=cast("Cache", cache),
        )
    elif prev_topk_indices is None:
        message = "Shared DSA layers require top-k indices from a previous full indexer layer."
        raise ValueError(message)
    else:
        topk_indices = prev_topk_indices
    sparse_mask = topk_attention_mask(attention, topk_indices, query_states, seq_length)
    return sparse_mask, topk_indices


def latent_kv(
    attention: Glm5NextTextAttention,
    hidden_states: torch.Tensor,
    shard: SequenceShard | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalised latent KV and shared rope key, over the whole sequence when given a shard.

    :param attention: DSA attention layer.
    :param hidden_states: Input hidden states ``[batch, seq_len, hidden_size]``.
    :param shard: This rank's sequence shard, or None.
    :return: Latent KV ``[batch, 1, kv_len, kv_lora_rank]`` and rope key
        ``[batch, 1, kv_len, qk_rope_head_dim]``.
    """
    compressed_kv = attention.kv_a_proj_with_mqa(hidden_states)
    if shard is not None:
        # Import cycle: agilerl.distributed imports agilerl.architectures, whose catalog imports this module.
        from agilerl.distributed.context_parallel.sequence import (
            gather_sequence_for_cp,
        )

        compressed_kv = gather_sequence_for_cp(compressed_kv, shard.group)
    batch_size, kv_length = compressed_kv.shape[:2]
    kv_pass, k_rot = torch.split(
        compressed_kv, [attention.kv_lora_rank, attention.qk_rope_head_dim], dim=-1
    )
    k_pass = attention.kv_a_layernorm(kv_pass).view(
        batch_size, 1, kv_length, attention.kv_lora_rank
    )
    k_rot = k_rot.view(batch_size, 1, kv_length, attention.qk_rope_head_dim)
    return k_pass, k_rot


def causal_sdpa_attention_forward(
    self: Glm5NextTextAttention,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    prev_topk_indices: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """``Glm5NextTextAttention.forward`` that runs causal SDPA without top-k when the mask is None.

    Given :data:`SEQUENCE_SHARD`, the queries are one sequence shard
    and the keys cover the whole sequence; see :func:`sequence_shard_sparse_mask`.

    :param hidden_states: Input hidden states ``[batch, seq_len, hidden_size]``.
    :param attention_mask: ``[batch, seq_len]`` boolean padding mask, or None
        when :func:`topk_covers_causal_prefix` held for this forward.
    :param past_key_values: Cache, or None.
    :param prev_topk_indices: Top-k indices from the previous full indexer layer.
    :return: Attention output, attention weights, and the top-k indices when the next layer shares them.
    """
    shard: SequenceShard | None = kwargs.pop(SEQUENCE_SHARD, None)
    batch_size, seq_length = hidden_states.shape[:-1]
    query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    query_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)

    k_pass, k_rot = latent_kv(self, hidden_states, shard)
    if past_key_values is not None:
        k_pass, k_rot = past_key_values.update(k_pass, k_rot, self.layer_idx)

    key_states, value_states = expand_latent_kv(self, k_pass, k_rot)

    if shard is not None:
        sparse_mask, topk_indices = sequence_shard_sparse_mask(
            self,
            shard,
            hidden_states,
            q_resid,
            query_states,
            attention_mask,
            prev_topk_indices,
        )
    # The SDPA backend sets is_causal=True when it gets no mask and seq_len > 1.
    elif attention_mask is None:
        topk_indices = None
        sparse_mask = None
    else:
        if self.indexer is not None:
            topk_indices = self.indexer(
                hidden_states=hidden_states,
                q_resid=q_resid,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
        else:
            if prev_topk_indices is None:
                message = "Shared DSA layers require top-k indices from a previous full indexer layer."
                raise ValueError(message)
            topk_indices = prev_topk_indices
        sparse_mask = topk_attention_mask(
            self, topk_indices, query_states, key_states.shape[2]
        )

    attention_interface: Callable[..., tuple[torch.Tensor, torch.Tensor | None]] = (
        ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        sparse_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, topk_indices if self.next_skip_topk else None


@library_shim("transformers", "==5.19.*")
def install_glm5_next_patches(
    _patch: PatchRuntimeConfig,
    _model: PreTrainedModel | PeftModel | None = None,
    _use_sequence_packing: bool = False,
) -> None:
    """Patch the GLM-5.3 text model and DSA attention to skip top-k, compile the hyper-connection mix, and register the Liger applier.

    :param _patch: Catalog patch config.
    :param _model: Already-built model, or None; the class-level patches cover it.
    :param _use_sequence_packing: Unused; the DSA mask ignores packed-document boundaries.
    """
    if HAS_LIGER_KERNEL:
        # Defers liger-kernel, an optional extra.
        from agilerl.architectures.glm5_next.liger import register_glm5_next_liger

        register_glm5_next_liger()
    compile_hyper_connection_mix(Glm5NextTextHyperConnection)
    if class_is_patched(Glm5NextTextAttention, CAUSAL_SDPA_WHEN_TOPK_COVERS):
        return
    model_cls: Any = Glm5NextTextModel
    attention_cls: Any = Glm5NextTextAttention
    model_cls.forward = topk_skipping_model_forward(model_cls.__dict__["forward"])
    attention_cls.forward = causal_sdpa_attention_forward
    setattr(attention_cls, CAUSAL_SDPA_WHEN_TOPK_COVERS, True)
